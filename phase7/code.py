from common import *
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

load_encoder()
P7_VARIANT = "lora_hybrid"
STAGE7 = f"phase7_{P7_VARIANT}"
P7_DIR = RUN_DIR / STAGE7
PCR_EPOCHS, PCR_LR, PCR_BATCH, PCR_TAU, PCR_METHOD = (4, 0.0003, 4, 0.05, "lora")
PCR_MAX_CHUNKS, PCR_HEAD_CHUNKS, PCR_TRAIN_CHUNKS = (8, 4, 4)
PCR_EMB_DOCS, PCR_HARD_POOL, PCR_K = (8, 30, 7)


def as_text(x):
    return "\n".join(map(str, x)) if isinstance(x, list) else "" if x is None else str(x)


def load_pcr_queries(path):
    raw = json.load(open(path, encoding="utf-8"))
    items = (
        raw
        if isinstance(raw, list)
        else [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in raw.items()]
    )
    out = []
    for i, r in enumerate(items):
        rel = r.get("relevant_candidates", r.get("relevant", r.get("labels")))
        out.append(
            {
                "id": str(r.get("id", r.get("query_id", i))),
                "text": as_text(r.get("text", r.get("document"))),
                "rel": None if rel is None else [str(x) for x in rel],
            }
        )
    return out


def load_pcr_cands(path):
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): as_text(v.get("text", v.get("document")) if isinstance(v, dict) else v) for k, v in raw.items()}
    return {
        str(r.get("id", r.get("candidate_id", i))): as_text(r.get("text", r.get("document"))) for i, r in enumerate(raw)
    }


def chunk_texts(texts, batch=128):
    C, out = (cfg.ft_max_len, [])
    for i in range(0, len(texts), batch):
        for ids in tok(texts[i : i + batch], add_special_tokens=False)["input_ids"]:
            ch = [ids[j : j + C] for j in range(0, len(ids), C)] or [[EOS_ID]]
            if len(ch) > PCR_MAX_CHUNKS:
                ch = ch[:PCR_HEAD_CHUNKS] + ch[-(PCR_MAX_CHUNKS - PCR_HEAD_CHUNKS) :]
            out.append(ch)
    return out


class BM25:

    def __init__(self, docs, k1=1.5, b=0.75):
        self.vec = CountVectorizer(
            token_pattern="(?u)\\b[a-zA-Z]{3,}\\b",
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.5,
            max_features=400000,
            dtype=np.float32,
        )
        tf = self.vec.fit_transform(docs).tocoo()
        N = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.bincount(tf.row, weights=tf.data, minlength=N)
        avg = dl.mean()
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avg))
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def scores(self, queries):
        q = self.vec.transform(queries).astype(np.float32)
        q.data = np.log1p(q.data)
        return (q @ self.W.T).toarray()


class DenseEncoder(nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        self.body = EncoderTaskModel(pretrained, cfg.d_model, "seq")

    def forward(self, ids, pad, doc_index, n_docs):
        v = self.body(ids, pad)
        out = torch.stack([v[doc_index == d].mean(0) for d in range(int(n_docs))])
        return F.normalize(out.float(), dim=-1)


def chunk_batch(doc_chunks):
    chunks, di = ([], [])
    for k, ch in enumerate(doc_chunks):
        for c in ch:
            chunks.append(c)
            di.append(k)
    L = max((len(c) for c in chunks))
    ids = torch.full((len(chunks), L), PAD_ID)
    pad = torch.zeros(len(chunks), L, dtype=torch.bool)
    for r, c in enumerate(chunks):
        ids[r, : len(c)] = torch.tensor(c)
        pad[r, : len(c)] = True
    return (ids, pad, torch.tensor(di), torch.tensor(len(doc_chunks)))


@torch.no_grad()
def embed_docs(m, doc_chunks):
    m.eval()
    out = []
    for i in range(0, len(doc_chunks), PCR_EMB_DOCS):
        ids, pad, di, nd = chunk_batch(doc_chunks[i : i + PCR_EMB_DOCS])
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            out.append(m(ids.to(DEVICE), pad.to(DEVICE), di.to(DEVICE), nd).float().cpu())
    return torch.cat(out).numpy()


def zrows(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-06)


def rank_topk(S, qids, cids, k=PCR_K):
    pos = {c: j for j, c in enumerate(cids)}
    out = {}
    for i, q in enumerate(qids):
        s = S[i].copy()
        if q in pos:
            s[pos[q]] = -np.inf
        out[q] = [cids[j] for j in np.argsort(-s)[:k]]
    return out


def micro_f1_at_k(ranked, gold):
    tp = sum((len(set(ranked[q]) & set(gold[q])) for q in gold))
    p = tp / max(1, sum((len(ranked[q]) for q in gold)))
    r = tp / max(1, sum((len(gold[q]) for q in gold)))
    return 2 * p * r / max(1e-09, p + r)


def train_short(ch):
    h = PCR_TRAIN_CHUNKS // 2
    return ch if len(ch) <= PCR_TRAIN_CHUNKS else ch[:h] + ch[-(PCR_TRAIN_CHUNKS - h) :]


def pcr_loss(emb, y):
    B = y.size(0)
    sim = emb[:B] @ emb[B:].t() / PCR_TAU
    return F.cross_entropy(sim.masked_fill(y == 2, -10000.0), torch.arange(B, device=sim.device))


if not stage_done(STAGE7):
    trq, dvq = (load_pcr_queries(P7["train"]), load_pcr_queries(P7["dev"]))
    tr_c, dv_c = (load_pcr_cands(P7["train_cands"]), load_pcr_cands(P7["dev_cands"]))
    tr_ids, dv_ids = (list(tr_c), list(dv_c))
    for q in trq:
        q["rel_in_pool"] = [c for c in q["rel"] or [] if c in tr_c and c != q["id"]]
    cover = np.mean([len(q["rel_in_pool"]) / max(1, len(q["rel"] or [])) for q in trq])
    print(
        f"PCR: train {len(trq)} queries / {len(tr_ids)} candidates | dev {len(dvq)} / {len(dv_ids)} | avg relevant {np.mean([len(q['rel'] or []) for q in trq]):.1f} (in pool {cover:.0%})"
    )
    t0 = time.time()
    tr_chunks = dict(zip(tr_ids, chunk_texts([tr_c[c] for c in tr_ids])))
    for q, ch in zip(trq, chunk_texts([q["text"] for q in trq])):
        q["chunks"] = ch
    dv_chunks = chunk_texts([dv_c[c] for c in dv_ids])
    dvq_chunks = chunk_texts([q["text"] for q in dvq])
    print(
        f"tokenized in {time.time() - t0:.0f}s | avg chunks/candidate {np.mean([len(c) for c in tr_chunks.values()]):.1f}"
    )
    bm_tr = BM25([tr_c[c] for c in tr_ids])
    S_tr = bm_tr.scores([q["text"] for q in trq])
    for i, q in enumerate(trq):
        rel = set(q["rel"] or []) | {q["id"]}
        q["hard"] = [tr_ids[j] for j in np.argsort(-S_tr[i])[: PCR_HARD_POOL + len(rel)] if tr_ids[j] not in rel][
            :PCR_HARD_POOL
        ]
    bm_dv = BM25([dv_c[c] for c in dv_ids])
    BM_DEV = bm_dv.scores([q["text"] for q in dvq])
    dv_gold = {q["id"]: q["rel"] or [] for q in dvq}
    dv_qids = [q["id"] for q in dvq]
    print(f"dev Micro-F1@7  BM25 only: {micro_f1_at_k(rank_topk(BM_DEV, dv_qids, dv_ids), dv_gold):.4f}")
    _prev_method = cfg.ft_method
    cfg.ft_method = PCR_METHOD
    pcr_model = DenseEncoder(fresh_pretrained()).to(DEVICE)
    cfg.ft_method = _prev_method
    ALPHA = {"a": 0.5}

    def dev_score(m):
        Eq, Ec = (embed_docs(m, dvq_chunks), embed_docs(m, dv_chunks))
        S = Eq @ Ec.T
        res = {
            a: micro_f1_at_k(rank_topk(a * zrows(BM_DEV) + (1 - a) * zrows(S), dv_qids, dv_ids), dv_gold)
            for a in np.round(np.arange(0, 1.01, 0.1), 1)
        }
        ALPHA["a"] = max(res, key=res.get)
        print(
            f"    dev Micro-F1@7  dense {res[0.0]:.4f} | BM25 {res[1.0]:.4f} | hybrid {res[ALPHA['a']]:.4f} (alpha {ALPHA['a']})"
        )
        return res[ALPHA["a"]]

    print("before fine-tuning:")
    dev_score(pcr_model)

    def pcr_batches():
        pairs = [(qi, c) for qi, q in enumerate(trq) for c in q["rel_in_pool"]]
        random.shuffle(pairs)
        for i in range(0, len(pairs) - PCR_BATCH + 1, PCR_BATCH):
            bp = pairs[i : i + PCR_BATCH]
            pos = [c for _, c in bp]
            neg = [random.choice(trq[qi]["hard"]) if trq[qi]["hard"] else random.choice(tr_ids) for qi, _ in bp]
            cand = pos + neg
            y = torch.zeros(len(bp), len(cand), dtype=torch.long)
            for r, (qi, c) in enumerate(bp):
                rel = set(trq[qi]["rel_in_pool"])
                for j, cid in enumerate(cand):
                    if cid in rel:
                        y[r, j] = 2
                y[r, r] = 1
            docs = [train_short(trq[qi]["chunks"]) for qi, _ in bp] + [train_short(tr_chunks[c]) for c in cand]
            yield (*chunk_batch(docs), y)

    n_pairs = sum((len(q["rel_in_pool"]) for q in trq))
    best7 = finetune(
        pcr_model,
        pcr_batches,
        dev_score,
        PCR_EPOCHS,
        PCR_LR,
        P7_DIR / "ckpt",
        f"PCR-{P7_VARIANT}",
        loss_fn=pcr_loss,
        batches_per_epoch=n_pairs // PCR_BATCH,
    )
    dev_score(pcr_model)
    A = ALPHA["a"]
    teq, te_c = (load_pcr_queries(P7["test"]), load_pcr_cands(P7["test_cands"]))
    if SMOKE:
        teq = teq[:4]
    te_ids = list(te_c)
    te_qids = [q["id"] for q in teq]
    Eq = embed_docs(pcr_model, chunk_texts([q["text"] for q in teq]))
    Ec = embed_docs(pcr_model, chunk_texts([te_c[c] for c in te_ids]))
    S_dense = Eq @ Ec.T
    S_bm = BM25([te_c[c] for c in te_ids]).scores([q["text"] for q in teq])
    hybrid = rank_topk(A * zrows(S_bm) + (1 - A) * zrows(S_dense), te_qids, te_ids)
    dense = rank_topk(S_dense, te_qids, te_ids)
    assert all((len(v) == PCR_K and len(set(v)) == PCR_K for v in hybrid.values())) and set(hybrid) == set(te_qids)
    print(f"test: {len(te_qids)} queries, pool {len(te_ids)} | alpha {A}")
    write_submission(hybrid, P7_DIR / "submission_phase7.zip")
    write_submission(dense, P7_DIR / "dense_only" / "submission_phase7.zip")
    mark_done(STAGE7, dev_micro_f1_at_7=best7, alpha=float(A))
    del pcr_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
else:
    print(f"{STAGE7} already done")
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

STAGE7F = "phase7_final"
P7F_DIR = RUN_DIR / STAGE7F
K1S, BS, QTOPS = ((0.9, 1.2, 1.5, 2.0), (0.4, 0.6, 0.75, 0.9), (None, 150, 400))
PASS_SIZES, LAMBDAS, WGRID = ((100, 200, 400), (0.0, 0.2, 0.5), (0, 0.5, 1, 2))
RR_TOPN, RR_GROUP, RR_GROUPS_PER_BATCH = (50, 8, 2)
RR_EPOCHS, RR_LR, RR_HEAD_MULT, RR_QTOK, RR_CTOK, RR_PASS_WORDS = (3, 2e-05, 50, 250, 250, 200)
LEGAL_STOP = set(
    "petitioner petitioners respondent respondents appellant appellants applicant applicants accused\ncomplainant learned counsel advocate advocates court courts high supreme hon honble ble judgment judgement order\ndated versus state india sri smt shri said aforesaid case cases para paragraph page present matter instant thereof\nherein therein whereas impugned filed".split()
)
STOP7 = list(ENGLISH_STOP_WORDS | LEGAL_STOP)
STAT_RE = [
    (re.compile("\\b(?:sections?|sec\\.?|u/s\\.?|s\\.)\\s*(\\d+[a-z]?)\\b"), " sec\\1 "),
    (re.compile("\\barticles?\\s*(\\d+[a-z]?)\\b"), " art\\1 "),
    (re.compile("\\b(?:order|o\\.)\\s*([ivxl]+|\\d+)\\s*,?\\s*(?:rule|r\\.)\\s*(\\d+)\\b"), " ord\\1r\\2 "),
]
TOKEN_PATTERN = "(?u)\\b[a-z][a-z0-9]{2,}\\b"


def clean7(t):
    t = re.sub("<[^>]+>", " ", t.lower())
    for rx, rep in STAT_RE:
        t = rx.sub(rep, t)
    t = re.sub("\\b\\d+\\b", " ", t)
    return re.sub("\\s+", " ", t)


def pass_split(text, n):
    w = text.split()
    if len(w) <= n:
        return [" ".join(w)]
    st = n // 2
    return [" ".join(w[i : i + n]) for i in range(0, len(w) - n + st, st)]


class BM25Index:

    def __init__(self, docs):
        self.vec = CountVectorizer(
            token_pattern=TOKEN_PATTERN,
            lowercase=False,
            stop_words=STOP7,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.5,
            max_features=500000,
            dtype=np.float32,
        )
        self.tf = self.vec.fit_transform(docs).tocoo()
        N = self.tf.shape[0]
        df = np.bincount(self.tf.col, minlength=self.tf.shape[1])
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.dl = np.bincount(self.tf.row, weights=self.tf.data, minlength=N)
        self.avg = max(self.dl.mean(), 1.0)

    def WT(self, k1, b):
        tf = self.tf
        w = self.idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * self.dl[tf.row] / self.avg))
        return sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape).T.tocsr()

    def Q(self, queries, qtop=None):
        q = self.vec.transform(queries).astype(np.float32).tocsr()
        q.data = np.log1p(q.data)
        if not qtop:
            return q
        rows, cols, vals = ([], [], [])
        for i in range(q.shape[0]):
            s, e = (q.indptr[i], q.indptr[i + 1])
            ix, v = (q.indices[s:e], q.data[s:e])
            keep = np.argsort(-(v * self.idf[ix]))[:qtop]
            rows += [i] * len(keep)
            cols += ix[keep].tolist()
            vals += v[keep].tolist()
        return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=q.shape)


def gold_and_self(qids, cids, gold):
    pos = {c: j for j, c in enumerate(cids)}
    G = np.zeros((len(qids), len(cids)), bool)
    M = np.zeros_like(G)
    for i, q in enumerate(qids):
        for c in gold.get(q, []):
            if c in pos:
                G[i, pos[c]] = True
        if q in pos:
            M[i, pos[q]] = True
    return (G, M)


def fast_f1(S, G, M, ng, rows=None):
    if rows is not None:
        S, G, M, ng = (S[rows], G[rows], M[rows], ng[rows])
    S = np.where(M, -np.inf, S)
    topk = np.argpartition(-S, PCR_K, axis=1)[:, :PCR_K]
    tp = G[np.arange(len(S))[:, None], topk].sum()
    p, r = (tp / (PCR_K * len(S)), tp / max(1, ng.sum()))
    return 2 * p * r / max(1e-09, p + r)


def scores_bm25(index, qtexts, k1, b, qt, starts=None):
    S = (index.Q(qtexts, qt) @ index.WT(k1, b)).toarray()
    return np.maximum.reduceat(S, starts, axis=1) if starts is not None else S


def tune_bm25(index, qtexts, G, M, ng, starts=None):
    best = (-1.0, None)
    Qs = {qt: index.Q(qtexts, qt) for qt in QTOPS}
    for k1, b in itertools.product(K1S, BS):
        WT = index.WT(k1, b)
        for qt in QTOPS:
            S = (Qs[qt] @ WT).toarray()
            if starts is not None:
                S = np.maximum.reduceat(S, starts, axis=1)
            f = fast_f1(S, G, M, ng)
            if f > best[0]:
                best = (f, (k1, b, qt))
    return best


def passage_index(ctexts_clean, n):
    ps, owner = ([], [])
    for j, t in enumerate(ctexts_clean):
        p = pass_split(t, n)
        ps += p
        owner += [j] * len(p)
    owner = np.array(owner)
    return (BM25Index(ps), np.r_[0, np.flatnonzero(np.diff(owner)) + 1])


@torch.no_grad()
def chunk_vecs(m, doc_chunks):
    m.eval()
    vecs, owner = ([], [])
    for i in range(0, len(doc_chunks), PCR_EMB_DOCS):
        ids, pad, di, nd = chunk_batch(doc_chunks[i : i + PCR_EMB_DOCS])
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            v = m.body(ids.to(DEVICE), pad.to(DEVICE))
        vecs.append(F.normalize(v.float(), dim=-1).cpu())
        owner += (di + i).tolist()
    return (torch.cat(vecs).numpy(), np.array(owner))


def dense_all(m, q_chunks, c_chunks):
    Qv, Qo = chunk_vecs(m, q_chunks)
    Cv, Co = chunk_vecs(m, c_chunks)

    def means(V, O, n):
        X = np.zeros((n, V.shape[1]), np.float32)
        np.add.at(X, O, V)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-06)

    Qm, Cm = (means(Qv, Qo, len(q_chunks)), means(Cv, Co, len(c_chunks)))
    c_start = np.r_[0, np.flatnonzero(np.diff(Co)) + 1]
    S_max = np.stack([np.maximum.reduceat(Qv[Qo == i] @ Cv.T, c_start, axis=1).mean(0) for i in range(len(q_chunks))])
    return (Qm @ Cm.T, S_max, Cm)


def knn_graph(Cm, k=5):
    sim = Cm @ Cm.T
    np.fill_diagonal(sim, -np.inf)
    nb = np.argpartition(-sim, k, axis=1)[:, :k]
    r = np.arange(len(Cm))[:, None]
    A = np.zeros_like(sim)
    A[r, nb] = np.maximum(sim[r, nb], 0)
    return A / (A.sum(1, keepdims=True) + 1e-06)


def fuse(Z, w, lam, A):
    S = sum((wi * z for wi, z in zip(w, Z)))
    return S + lam * zrows(S @ A.T) if lam else S


def search_fusion(Z, A, G, M, ng, rows=None):
    best = (-1.0, None)
    for lam in LAMBDAS:
        for w in itertools.product(WGRID, repeat=len(Z)):
            if sum(w) == 0:
                continue
            f = fast_f1(fuse(Z, w, lam, A), G, M, ng, rows)
            if f > best[0]:
                best = (f, (w, lam))
    return best


def first_stage_features(dm, queries, cands, cids, P_doc, PASS_N, P_pass):
    qc = [clean7(q["text"]) for q in queries]
    cc = [clean7(cands[c]) for c in cids]
    S_doc = scores_bm25(BM25Index(cc), qc, *P_doc)
    pidx, starts = passage_index(cc, PASS_N)
    S_pass = scores_bm25(pidx, qc, *P_pass, starts=starts)
    S_mean, S_max, Cm = dense_all(dm, chunk_texts([q["text"] for q in queries]), chunk_texts([cands[c] for c in cids]))
    return ([zrows(x) for x in (S_doc, S_pass, S_mean, S_max)], knn_graph(Cm))


class SplitPassages:

    def __init__(self, qtexts, ctexts, pass_params):
        self.qtexts, self.pass_params = (qtexts, pass_params)
        self.qp = [pass_split(t, RR_PASS_WORDS) for t in qtexts]
        cp = [pass_split(t, RR_PASS_WORDS) for t in ctexts]
        self.c_start = np.r_[0, np.cumsum([len(p) for p in cp])]
        self.flat_c = [p for ps in cp for p in ps]
        clean_c = [clean7(p) for p in self.flat_c]
        self.bm = BM25Index(clean_c)
        self.WT = self.bm.WT(pass_params[0], pass_params[1])
        self.tfidf = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            lowercase=False,
            stop_words=STOP7,
            sublinear_tf=True,
            max_features=300000,
            dtype=np.float32,
        )
        self.tfidf.fit(clean_c + [clean7(p) for ps in self.qp for p in ps])

    def pair_texts(self, pairs):
        out, by_q = ({}, defaultdict(list))
        for qi, cj in pairs:
            by_q[qi].append(cj)
        for qi, cjs in by_q.items():
            s = (self.bm.Q([clean7(self.qtexts[qi])], self.pass_params[2]) @ self.WT).toarray()[0]
            Xq = self.tfidf.transform([clean7(p) for p in self.qp[qi]])
            for cj in cjs:
                a, b = (self.c_start[cj], self.c_start[cj + 1])
                cpass = self.flat_c[a + int(np.argmax(s[a:b]))]
                qsim = (Xq @ self.tfidf.transform([clean7(cpass)]).T).toarray().ravel()
                out[qi, cj] = (self.qp[qi][int(np.argmax(qsim))], cpass)
        return out


def tokenize_pairs(texts):
    keys = list(texts)
    qs = [texts[k][0] for k in keys]
    cs = [texts[k][1] for k in keys]
    qi = tok(qs, add_special_tokens=False)["input_ids"]
    ci = tok(cs, add_special_tokens=False)["input_ids"]
    return {k: a[:RR_QTOK] + [EOS_ID] + b[:RR_CTOK] for k, a, b in zip(keys, qi, ci)}


def pad_batch(seqs):
    L = max((len(s) for s in seqs))
    ids = torch.full((len(seqs), L), PAD_ID)
    pad = torch.zeros(len(seqs), L, dtype=torch.bool)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s)
        pad[r, : len(s)] = True
    return (ids, pad)


class CrossEncoder(nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        self.body = EncoderTaskModel(pretrained, 1, "seq")

    def forward(self, ids, pad):
        return self.body(ids, pad).squeeze(-1)


@torch.no_grad()
def rr_score(m, seqs, bs=32):
    m.eval()
    out = []
    for i in range(0, len(seqs), bs):
        ids, pad = pad_batch(seqs[i : i + bs])
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            out.append(m(ids.to(DEVICE), pad.to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def rr_loss(logits, y):
    return F.cross_entropy(logits.view(len(y), RR_GROUP), y)


def topn(S, Mself, n=RR_TOPN):
    return np.argsort(-np.where(Mself, -np.inf, S), axis=1)[:, :n]


def blend(S0, top_idx, r, w):
    out = np.full(S0.shape, -np.inf, np.float32)
    rows = np.arange(len(S0))[:, None]
    out[rows, top_idx] = w * zrows(S0[rows, top_idx]) + (1 - w) * zrows(r)
    return out


def pair_seqs(sp_obj, top_idx, n_queries):
    tokd = tokenize_pairs(sp_obj.pair_texts([(i, int(c)) for i in range(n_queries) for c in top_idx[i]]))
    return [tokd[i, int(c)] for i in range(n_queries) for c in top_idx[i]]


if not stage_done(STAGE7F):
    t0 = time.time()
    dm = with_method(PCR_METHOD, lambda: DenseEncoder(fresh_pretrained())).to(DEVICE)
    dm.load_state_dict(torch.load(P7_DIR / "ckpt" / "best.pt", map_location=DEVICE, weights_only=False)["model"])
    dvq, dv_c = (load_pcr_queries(P7["dev"]), load_pcr_cands(P7["dev_cands"]))
    dv_ids = list(dv_c)
    dv_qids = [q["id"] for q in dvq]
    dv_gold = {q["id"]: q["rel"] or [] for q in dvq}
    G, M, ng = (*gold_and_self(dv_qids, dv_ids, dv_gold), np.array([len(dv_gold[q]) for q in dv_qids]))
    qc = [clean7(q["text"]) for q in dvq]
    cc = [clean7(dv_c[c]) for c in dv_ids]
    f_doc, P_doc = tune_bm25(BM25Index(cc), qc, G, M, ng)
    print(f"bm25_doc best {f_doc:.4f} with (k1, b, query_terms) = {P_doc}   [{time.time() - t0:.0f}s]")
    best_pass = (-1.0, None)
    for n in PASS_SIZES:
        pidx, starts = passage_index(cc, n)
        f, P = tune_bm25(pidx, qc, G, M, ng, starts)
        print(f"bm25_passage {n} words: best {f:.4f} with {P}   [{time.time() - t0:.0f}s]")
        if f > best_pass[0]:
            best_pass = (f, (n, P))
    PASS_N, P_pass = best_pass[1]
    Z, A = first_stage_features(dm, dvq, dv_c, dv_ids, P_doc, PASS_N, P_pass)
    for n, z in zip(("bm25_doc", "bm25_passage", "dense_mean", "dense_maxsim"), Z):
        print(f"  {n:14s} dev {fast_f1(z, G, M, ng):.4f}")
    fold = np.random.default_rng(SEED + 11).integers(0, 2, len(dv_qids))
    cv = []
    for f in (0, 1):
        _, (w, lam) = search_fusion(Z, A, G, M, ng, rows=fold != f)
        cv.append(fast_f1(fuse(Z, w, lam, A), G, M, ng, rows=fold == f))
    f_fuse, (W_BEST, LAM_BEST) = search_fusion(Z, A, G, M, ng)
    S0_dev = fuse(Z, W_BEST, LAM_BEST, A)
    print(f"first stage: dev {f_fuse:.4f} | 2-fold CV {np.mean(cv):.4f} | weights {W_BEST} lambda {LAM_BEST}")
    teq, te_c = (load_pcr_queries(P7["test"]), load_pcr_cands(P7["test_cands"]))
    te_ids = list(te_c)
    if SMOKE:
        teq = teq[:4]
    te_qids = [q["id"] for q in teq]
    Zt, At = first_stage_features(dm, teq, te_c, te_ids, P_doc, PASS_N, P_pass)
    S_test = fuse(Zt, W_BEST, LAM_BEST, At)
    write_submission(rank_topk(S_test, te_qids, te_ids), P7F_DIR / "first_stage" / "submission_phase7.zip")
    del dm
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    trq, tr_c = (load_pcr_queries(P7["train"]), load_pcr_cands(P7["train_cands"]))
    tr_ids = list(tr_c)
    tpos = {c: j for j, c in enumerate(tr_ids)}
    trq = [q for q in trq if any((c in tpos and c != q["id"] for c in q["rel"] or []))]
    S_tr = scores_bm25(BM25Index([clean7(tr_c[c]) for c in tr_ids]), [clean7(q["text"]) for q in trq], *P_doc)
    _, M_tr = gold_and_self([q["id"] for q in trq], tr_ids, {})
    top_tr = topn(S_tr, M_tr)
    groups, neg_pool, need = ([], {}, set())
    for qi, q in enumerate(trq):
        gold = {tpos[c] for c in q["rel"] if c in tpos and c != q["id"]}
        negs = [int(c) for c in top_tr[qi] if c not in gold]
        if not negs:
            continue
        neg_pool[qi] = negs
        need |= {(qi, c) for c in negs} | {(qi, c) for c in gold}
        groups += [(qi, c) for c in gold]
    sp_tr = SplitPassages([q["text"] for q in trq], [tr_c[c] for c in tr_ids], P_pass)
    tr_tok = tokenize_pairs(sp_tr.pair_texts(sorted(need)))
    print(f"rerank train: {len(trq)} queries, {len(groups)} groups, {len(tr_tok)} pairs   [{time.time() - t0:.0f}s]")
    top_dev = topn(S0_dev, M)
    dv_seqs = pair_seqs(SplitPassages([q["text"] for q in dvq], [dv_c[c] for c in dv_ids], P_pass), top_dev, len(dvq))

    def rr_batches():
        gs = list(groups)
        random.shuffle(gs)
        for i in range(0, len(gs) - RR_GROUPS_PER_BATCH + 1, RR_GROUPS_PER_BATCH):
            seqs, ys = ([], [])
            for qi, pc in gs[i : i + RR_GROUPS_PER_BATCH]:
                pool = neg_pool[qi]
                negs = (
                    random.sample(pool, RR_GROUP - 1)
                    if len(pool) >= RR_GROUP - 1
                    else [random.choice(pool) for _ in range(RR_GROUP - 1)]
                )
                seqs += [tr_tok[qi, c] for c in [pc] + negs]
                ys.append(0)
            yield (*pad_batch(seqs), torch.tensor(ys))

    WGRID_R = np.round(np.arange(0, 1.01, 0.1), 1)

    def dev_score(m):
        r = rr_score(m, dv_seqs).reshape(len(dvq), RR_TOPN)
        res = {w: fast_f1(blend(S0_dev, top_dev, r, w), G, M, ng) for w in WGRID_R}
        wb = max(res, key=res.get)
        print(
            f"    dev Micro-F1@7 rerank only {res[0.0]:.4f} | first stage {res[1.0]:.4f} | blend {res[wb]:.4f} (w {wb})"
        )
        return res[wb]

    rr_model = with_method("full", lambda: CrossEncoder(fresh_pretrained()))
    for p in rr_model.body.embed.parameters():
        p.requires_grad_(False)
    finetune(
        rr_model,
        rr_batches,
        dev_score,
        RR_EPOCHS,
        RR_LR,
        P7F_DIR / "ckpt",
        "RERANK",
        loss_fn=rr_loss,
        head_lr_mult=RR_HEAD_MULT,
        batches_per_epoch=len(groups) // RR_GROUPS_PER_BATCH,
    )
    r_dev = rr_score(rr_model, dv_seqs).reshape(len(dvq), RR_TOPN)
    fold = np.random.default_rng(SEED + 17).integers(0, 2, len(G))
    cv_r = []
    for f in (0, 1):
        w_f = max(WGRID_R, key=lambda w: fast_f1(blend(S0_dev, top_dev, r_dev, w), G, M, ng, fold != f))
        cv_r.append(fast_f1(blend(S0_dev, top_dev, r_dev, w_f), G, M, ng, fold == f))
    W_R = max(WGRID_R, key=lambda w: fast_f1(blend(S0_dev, top_dev, r_dev, w), G, M, ng))
    cv_base = np.mean([fast_f1(S0_dev, G, M, ng, fold == f) for f in (0, 1)])
    print(f"blend weight {W_R} | 2-fold CV {np.mean(cv_r):.4f} vs first stage {cv_base:.4f}")
    _, M_te = gold_and_self(te_qids, te_ids, {})
    top_te = topn(S_test, M_te)
    te_seqs = pair_seqs(SplitPassages([q["text"] for q in teq], [te_c[c] for c in te_ids], P_pass), top_te, len(teq))
    r_te = rr_score(rr_model, te_seqs).reshape(len(teq), RR_TOPN)
    ranked = rank_topk(blend(S_test, top_te, r_te, W_R), te_qids, te_ids)
    assert set(ranked) == set(te_qids) and all((len(v) == PCR_K and len(set(v)) == PCR_K for v in ranked.values()))
    write_submission(ranked, P7F_DIR / "submission_phase7.zip")
    mark_done(
        STAGE7F,
        first_stage_cv=float(np.mean(cv)),
        rerank_cv=float(np.mean(cv_r)),
        blend_w=float(W_R),
        bm25_doc=list(P_doc),
        passage_words=PASS_N,
        bm25_passage=list(P_pass),
        weights=list(W_BEST),
        lam=LAM_BEST,
    )
    print(f"done in {time.time() - t0:.0f}s")
    del rr_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
else:
    print(f"{STAGE7F} already done")
