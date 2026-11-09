from common import *

load_encoder()
P6_VARIANT = "salient_long"
STAGE6 = "phase6"
P6_DIR = RUN_DIR / STAGE6
SUMM_SRC_BUDGET, SUMM_TGT_LEN = (1024, 1024)
SUMM_EPOCHS, SUMM_LR, SUMM_BATCH, SUMM_VAL_FRAC, SUMM_CKPT_EVERY = (8, 5e-05, 4, 0.05, 1000)
SUMM_GEN_BATCH, SUMM_REP_PEN, SUMM_NO_REPEAT, SUMM_DEV_GEN = (8, 1.2, 4, 50)


def as_text(x):
    return "\n".join(map(str, x)) if isinstance(x, list) else "" if x is None else str(x)


STOPWORDS = set(
    "the of and to in a is that for be by on as with was it this which or are an at from not has have had shall any been such its under said were there their they also than into upon same other may would".split()
)
SENT_SPLIT = re.compile("(?<=[.?!;:])\\s+|\\n+")


def salient_src(text, budget):
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s and len(s.strip()) > 2]
    if not sents:
        return [EOS_ID]
    ids = tok([(" " if i else "") + s for i, s in enumerate(sents)], add_special_tokens=False)["input_ids"]
    flat = [t for x in ids for t in x]
    if len(flat) <= budget:
        return flat
    words = [[w for w in re.findall("[a-z]{3,}", s.lower()) if w not in STOPWORDS] for s in sents]
    tf = Counter((w for ws in words for w in ws))
    n = len(sents)
    score = []
    for i, ws in enumerate(words):
        u = set(ws)
        base = sum((tf[w] for w in u)) / (len(u) ** 0.5 + 1)
        score.append(base * (1.5 if i < 5 else 1.3 if i >= n - 10 else 1.0))
    chosen, used = ([], 0)
    for i in sorted(range(n), key=lambda i: -score[i]):
        L = len(ids[i])
        if L > budget // 4:
            continue
        if used + L <= budget:
            chosen.append(i)
            used += L
        if used >= budget - 8:
            break
    return [t for i in sorted(chosen) for t in ids[i]] or flat[:budget]


def load_summ(path):
    raw = json.load(open(path, encoding="utf-8"))
    items = (
        raw
        if isinstance(raw, list)
        else [{"id": k, **(v if isinstance(v, dict) else {"document": v})} for k, v in raw.items()]
    )
    return [
        {
            "id": str(r.get("id", i)),
            "document": as_text(r.get("document", r.get("text"))),
            "summary": None if r.get("summary") is None else as_text(r["summary"]),
        }
        for i, r in enumerate(items)
    ]


def tokenize_summ(docs, batch=128):
    t0 = time.time()
    for k, d in enumerate(docs):
        d["src"] = salient_src(d["document"], SUMM_SRC_BUDGET)
        d["doc_len"] = len(d["document"].split())
        if k and k % 2000 == 0:
            print(f"    selected sentences for {k}/{len(docs)} docs ({time.time() - t0:.0f}s)")
    with_sum = [d for d in docs if d["summary"] is not None]
    for i in range(0, len(with_sum), batch):
        part = with_sum[i : i + batch]
        for d, ids in zip(part, tok([d["summary"] for d in part], add_special_tokens=False)["input_ids"]):
            d["sum_len"] = len(ids)
            d["tgt"] = ids[: SUMM_TGT_LEN - 1]


class SummModel(nn.Module):

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, src, src_pad, tin, tpad):
        return self.m.decode(tin, self.m.encode(src, src_pad), src_pad, tpad)


def summ_batches(docs, batch, shuffle):
    idx = list(range(len(docs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch):
        ds = [docs[j] for j in idx[i : i + batch]]
        S = max((len(d["src"]) for d in ds))
        T = max((len(d["tgt"]) for d in ds)) + 1
        src = torch.full((len(ds), S), PAD_ID)
        spad = torch.zeros(len(ds), S, dtype=torch.bool)
        tin = torch.full((len(ds), T), PAD_ID)
        tpad = torch.zeros(len(ds), T, dtype=torch.bool)
        tout = torch.full((len(ds), T), -100)
        for r, d in enumerate(ds):
            s, t = (d["src"], d["tgt"])
            src[r, : len(s)] = torch.tensor(s)
            spad[r, : len(s)] = True
            tin[r, : len(t) + 1] = torch.tensor([BOS_ID] + t)
            tpad[r, : len(t) + 1] = True
            tout[r, : len(t) + 1] = torch.tensor(t + [EOS_ID])
        yield (src, spad, tin, tpad, tout)


@torch.no_grad()
def summ_dev_loss(m, docs):
    m.eval()
    tot = n = 0.0
    for src, spad, tin, tpad, tout in summ_batches(docs, SUMM_BATCH * 2, False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            lg = m(src.to(DEVICE), spad.to(DEVICE), tin.to(DEVICE), tpad.to(DEVICE))
        tout = tout.to(DEVICE)
        tot += F.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), tout.reshape(-1), ignore_index=-100, reduction="sum"
        ).item()
        n += (tout != -100).sum().item()
    return tot / max(1, n)


@torch.no_grad()
def summ_generate(m, docs, max_new, min_new):
    m.eval()
    out = {}
    for i in range(0, len(docs), SUMM_GEN_BATCH):
        part = docs[i : i + SUMM_GEN_BATCH]
        S = max((len(d["src"]) for d in part))
        src = torch.full((len(part), S), PAD_ID)
        pad = torch.zeros(len(part), S, dtype=torch.bool)
        for j, d in enumerate(part):
            src[j, : len(d["src"])] = torch.tensor(d["src"])
            pad[j, : len(d["src"])] = True
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            g = m.m.generate(
                src.to(DEVICE),
                pad.to(DEVICE),
                max_new,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=SUMM_REP_PEN,
                min_new=min_new,
                no_repeat_ngram=SUMM_NO_REPEAT,
            )
        for d, row in zip(part, g.tolist()):
            if EOS_ID in row:
                row = row[: row.index(EOS_ID)]
            out[d["id"]] = re.sub("\\s+", " ", tok.decode(row, skip_special_tokens=True)).strip() or "."
        if i // SUMM_GEN_BATCH % 10 == 0:
            print(f"    generated {min(i + SUMM_GEN_BATCH, len(docs))}/{len(docs)}")
    return out


def rouge_1_l(pred, ref):
    p, r = (pred.lower().split(), ref.lower().split())
    if not p or not r:
        return (0.0, 0.0)
    ov = sum((Counter(p) & Counter(r)).values())
    r1 = 2 * ov / (len(p) + len(r))
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    return (r1, 2 * prev[-1] / (len(p) + len(r)))


if not stage_done(STAGE6):
    all6 = load_summ(P6["train"])
    in_dev = lambda d: int(hashlib.sha1(f"summ:{d['id']}".encode()).hexdigest()[:8], 16) / 2**32 < SUMM_VAL_FRAC
    train6, dev6 = ([d for d in all6 if not in_dev(d)], [d for d in all6 if in_dev(d)])
    if cfg.ft_max_docs:
        train6, dev6 = (train6[: cfg.ft_max_docs], dev6[: max(4, cfg.ft_max_docs // 5)])
    tokenize_summ(train6 + dev6)
    sl = np.array([d["sum_len"] for d in train6])
    SUMM_MAX_NEW = int(min(SUMM_TGT_LEN, np.percentile(sl, 90)))
    SUMM_MIN_NEW = int(0.6 * np.median(sl))
    print(
        f"SUMM[{P6_VARIANT}]: {len(train6)} train, {len(dev6)} dev | summary tokens median {int(np.median(sl))}, p90 {int(np.percentile(sl, 90))}, {(sl > SUMM_TGT_LEN - 1).mean():.0%} truncated | generate {SUMM_MIN_NEW}-{SUMM_MAX_NEW} tokens"
    )
    summ_model = SummModel(fresh_pretrained())

    def dev_score(m):
        l = summ_dev_loss(m, dev6)
        print(f"    dev loss {l:.4f} (ppl {math.exp(l):.1f})")
        return -l

    best6 = finetune(
        summ_model,
        lambda: summ_batches(train6, SUMM_BATCH, True),
        dev_score,
        SUMM_EPOCHS,
        SUMM_LR,
        P6_DIR / "ckpt",
        f"SUMM-{P6_VARIANT}",
        batches_per_epoch=math.ceil(len(train6) / SUMM_BATCH),
        ckpt_every=SUMM_CKPT_EVERY,
    )
    sample = dev6[:SUMM_DEV_GEN]
    gen = summ_generate(summ_model, sample, SUMM_MAX_NEW, SUMM_MIN_NEW)
    scores = np.array([rouge_1_l(gen[d["id"]], d["summary"]) for d in sample])
    print(
        f"dev sample ({len(sample)}): ROUGE-1 F1 {scores[:, 0].mean():.4f} | ROUGE-L F1 {scores[:, 1].mean():.4f} | avg words {np.mean([len(gen[d['id']].split()) for d in sample]):.0f} vs ref {np.mean([len(d['summary'].split()) for d in sample]):.0f}"
    )
    try:
        from bert_score import score as bscore

        _, _, F1 = bscore([gen[d["id"]] for d in sample], [d["summary"] for d in sample], lang="en", verbose=False)
        print(f"dev sample BERTScore F1 {F1.mean().item():.4f}")
    except ImportError:
        print("(pip install bert-score for a local BERTScore estimate)")
    print(f"\nREF >> {sample[0]['summary'][:400]}\nGEN >> {gen[sample[0]['id']][:400]}")
    test6 = load_summ(P6["test"])
    if SMOKE:
        test6 = test6[:4]
    tokenize_summ(test6)
    gen_t = summ_generate(summ_model, test6, SUMM_MAX_NEW, SUMM_MIN_NEW)
    write_submission([{"id": d["id"], "summary": gen_t[d["id"]]} for d in test6], P6_DIR / "submission_phase6.zip")
    mark_done(STAGE6, dev_loss=-best6, rouge1=float(scores[:, 0].mean()), rougeL=float(scores[:, 1].mean()))
    del summ_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
else:
    print(f"{STAGE6} already done")
