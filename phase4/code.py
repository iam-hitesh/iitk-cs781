from common import *

load_encoder()
P4_DIR = RUN_DIR / "phase4"
CJPE_TRAIN_CAP, CJPE_EPOCHS, CJPE_LR, CJPE_EVAL_EVERY = (None, 2, 2e-05, 2000)


def load_cjpe(path):
    raw = json.load(open(path, encoding="utf-8"))
    docs = []
    items = (
        raw
        if isinstance(raw, list)
        else [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in raw.items()]
    )
    for i, r in enumerate(items):
        lab = r.get("label", r.get("labels", r.get("pred", r.get("decision"))))
        if isinstance(lab, list):
            lab = lab[0]
        docs.append(
            {
                "id": str(r.get("id", i)),
                "text": extract_text(r) if "text" not in r else r["text"],
                "label": None if lab is None else int(lab),
            }
        )
    return docs


def cjpe_chunks(text):
    ids = tok.encode(text, add_special_tokens=False)
    C = cfg.ft_max_len
    chunks = [ids[i : i + C] for i in range(0, max(1, len(ids)), C)] or [[EOS_ID]]
    if len(chunks) > cfg.cjpe_max_chunks:
        h = cfg.cjpe_head_chunks
        chunks = chunks[:h] + chunks[-(cfg.cjpe_max_chunks - h) :]
    return chunks


class HierarchicalClassifier(nn.Module):

    def __init__(self, pretrained, n_labels):
        super().__init__()
        self.body = EncoderTaskModel(pretrained, cfg.d_model, "seq")
        self.q = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.head = nn.Linear(cfg.d_model, n_labels)
        self.drop = nn.Dropout(0.1)

    def forward(self, ids, pad, doc_index, n_docs):
        v = torch.tanh(self.body(ids, pad))
        s = v @ self.q.to(v.dtype)
        outs = []
        for d in range(int(n_docs)):
            sel = doc_index == d
            w = F.softmax(s[sel].float(), 0).to(v.dtype)
            outs.append((w[:, None] * v[sel]).sum(0))
        return self.head(self.drop(torch.stack(outs)))


def cjpe_batches(docs, batch_docs, shuffle):
    idx = list(range(len(docs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_docs):
        ds = [docs[j] for j in idx[i : i + batch_docs]]
        chunks, doc_index = ([], [])
        for d_i, d in enumerate(ds):
            for c in d["chunks"]:
                chunks.append(c)
                doc_index.append(d_i)
        L = max((len(c) for c in chunks))
        ids = torch.full((len(chunks), L), PAD_ID)
        pad = torch.zeros(len(chunks), L, dtype=torch.bool)
        for r, c in enumerate(chunks):
            ids[r, : len(c)] = torch.tensor(c)
            pad[r, : len(c)] = True
        y = torch.tensor([d["label"] if d["label"] is not None else -100 for d in ds])
        yield (ids, pad, torch.tensor(doc_index), torch.tensor(len(ds)), y)


@torch.no_grad()
def cjpe_probs(task_model, docs):
    task_model.eval()
    probs = []
    for ids, pad, di, nd, _ in cjpe_batches(docs, cfg.cjpe_batch_docs, False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = task_model(ids.to(DEVICE), pad.to(DEVICE), di.to(DEVICE), nd)
        probs.extend(F.softmax(logits.float(), -1)[:, 1].cpu().tolist())
    return np.array(probs)


def best_threshold(probs, y):
    best = (0.5, -1.0)
    for t in np.linspace(0.05, 0.95, 91):
        f = macro_f1(list(y), list((probs >= t).astype(int)), [0, 1])
        if f > best[1]:
            best = (float(t), f)
    return best


if not stage_done("phase4"):
    train4, dev4 = (load_cjpe(P4["train"]), load_cjpe(P4["dev"]))
    random.Random(SEED).shuffle(train4)
    train4 = train4[:CJPE_TRAIN_CAP]
    if cfg.ft_max_docs:
        train4, dev4 = (train4[: cfg.ft_max_docs], dev4[: max(4, cfg.ft_max_docs // 5)])
    for d in train4 + dev4:
        d["chunks"] = cjpe_chunks(d["text"])
    print(
        f"CJPE: {len(train4)} train docs, {len(dev4)} dev | label dist train {Counter((d['label'] for d in train4))} | avg chunks {np.mean([len(d['chunks']) for d in train4]):.1f}"
    )
    cjpe_model = HierarchicalClassifier(fresh_pretrained(), 2)
    for p in cjpe_model.body.embed.parameters():
        p.requires_grad_(False)
    THRESH = {"t": 0.5}

    def dev_score(m):
        p = cjpe_probs(m, dev4)
        y = np.array([d["label"] for d in dev4])
        t, f = best_threshold(p, y)
        THRESH["t"] = t
        print(
            f"    dev macro-F1 {f:.4f} at threshold {t:.2f} (F1@0.5 = {macro_f1(list(y), list((p >= 0.5).astype(int)), [0, 1]):.4f})"
        )
        return f

    best4 = finetune(
        cjpe_model,
        lambda: cjpe_batches(train4, cfg.cjpe_batch_docs, True),
        dev_score,
        CJPE_EPOCHS,
        CJPE_LR,
        P4_DIR / "ckpt",
        "CJPE",
        batches_per_epoch=math.ceil(len(train4) / cfg.cjpe_batch_docs),
        eval_every=CJPE_EVAL_EVERY,
    )
    dev_score(cjpe_model)
    test4 = load_cjpe(P4["test"])
    if SMOKE:
        test4 = test4[:6]
    for d in test4:
        d["chunks"] = cjpe_chunks(d["text"])
    p = cjpe_probs(cjpe_model, test4)
    preds4 = [{"id": d["id"], "pred": int(pr >= THRESH["t"])} for d, pr in zip(test4, p)]
    print(f"test predictions: {Counter((x['pred'] for x in preds4))}")
    write_submission(preds4, P4_DIR / "submission_phase4.zip")
    mark_done("phase4", dev_f1=best4, threshold=THRESH["t"])
    del cjpe_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
else:
    print("phase4 already done")
