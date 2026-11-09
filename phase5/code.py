from common import *

load_encoder()
STAT_LIST, STAT_ID, N_STAT = ([], {}, 0)


def as_text(x):
    return "\n".join(map(str, x)) if isinstance(x, list) else "" if x is None else str(x)


def load_lsi(path):
    raw = json.load(open(path, encoding="utf-8"))
    items = (
        raw
        if isinstance(raw, list)
        else [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in raw.items()]
    )
    docs = []
    for i, r in enumerate(items):
        labs = r.get("labels", r.get("label"))
        if labs is not None and (not isinstance(labs, list)):
            labs = [labs]
        docs.append(
            {
                "id": str(r.get("id", i)),
                "text": as_text(r.get("text", r.get("document"))),
                "labels": None if labs is None else sorted({int(x) for x in labs}),
            }
        )
    return docs


def add_chunks(docs, max_chunks, head_chunks, batch=256):
    C = cfg.ft_max_len
    for i in range(0, len(docs), batch):
        part = docs[i : i + batch]
        for d, ids in zip(part, tok([d["text"] for d in part], add_special_tokens=False)["input_ids"]):
            ch = [ids[j : j + C] for j in range(0, len(ids), C)] or [[EOS_ID]]
            if len(ch) > max_chunks:
                ch = ch[:head_chunks] + ch[-(max_chunks - head_chunks) :]
            d["chunks"] = ch


def lsi_batches(docs, batch_docs, shuffle):
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
        y = torch.zeros(len(ds), N_STAT)
        for r, d in enumerate(ds):
            for l in d["labels"] or []:
                if l in STAT_ID:
                    y[r, STAT_ID[l]] = 1.0
        yield (ids, pad, torch.tensor(doc_index), torch.tensor(len(ds)), y)


@torch.no_grad()
def lsi_probs(m, docs):
    m.eval()
    out = []
    for ids, pad, di, nd, _ in lsi_batches(docs, LSI_BATCH_DOCS * 2, False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            lg = m(ids.to(DEVICE), pad.to(DEVICE), di.to(DEVICE), nd)
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_STAT))


def multi_hot(docs):
    Y = np.zeros((len(docs), N_STAT), np.float32)
    for r, d in enumerate(docs):
        for l in d["labels"] or []:
            if l in STAT_ID:
                Y[r, STAT_ID[l]] = 1.0
    return Y


def ml_macro_f1(Y, P):
    tp = (Y * P).sum(0)
    fp = ((1 - Y) * P).sum(0)
    fn = (Y * (1 - P)).sum(0)
    f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    present = Y.sum(0) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def ml_micro_f1(Y, P):
    tp = (Y * P).sum()
    fp = ((1 - Y) * P).sum()
    fn = (Y * (1 - P)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def decide(probs, th):
    P = (probs >= th[None, :]).astype(np.float32)
    empty = P.sum(1) == 0
    if empty.any():
        P[np.where(empty)[0], probs[empty].argmax(1)] = 1.0
    return P


def tune_thresholds(probs, Y, grid=np.round(np.arange(0.05, 0.95, 0.05), 2), min_pos=5):
    g = [ml_macro_f1(Y, decide(probs, np.full(probs.shape[1], t))) for t in grid]
    th = np.full(probs.shape[1], grid[int(np.argmax(g))])
    for j in np.where(Y.sum(0) >= min_pos)[0]:
        y, col = (Y[:, j], probs[:, j])
        f = [2 * ((col >= t) & (y == 1)).sum() / max(1, (col >= t).sum() + y.sum()) for t in grid]
        th[j] = grid[int(np.argmax(f))]
    return th


class LabelHead(nn.Module):

    def __init__(self, n_labels, d):
        super().__init__()
        self.u = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.w = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_labels))


class LabelAttentionClassifier(nn.Module):

    def __init__(self, pretrained, n_labels):
        super().__init__()
        self.body = EncoderTaskModel(pretrained, cfg.d_model, "token")
        self.head = LabelHead(n_labels, cfg.d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self, ids, pad, doc_index, n_docs):
        h = torch.tanh(self.body(ids, pad))
        out = []
        for d in range(int(n_docs)):
            sel = doc_index == d
            hd = h[sel][pad[sel]]
            att = torch.softmax((self.head.u.to(hd.dtype) @ hd.t()).float(), -1).to(hd.dtype)
            v = self.drop(att @ hd)
            out.append((v * self.head.w.to(v.dtype)).sum(-1) + self.head.b.to(v.dtype))
        return torch.stack(out)


from sklearn.linear_model import LogisticRegression

LSI_MAX_CHUNKS, LSI_HEAD_CHUNKS, LSI_BATCH_DOCS, LSI_POS_WEIGHT_CAP, LSI_DEV_EVAL_CAP = (8, 4, 4, 20.0, 2000)
RUNS5 = [
    {
        "stage": "phase5_lora_laat",
        "method": "lora",
        "lr": 0.0003,
        "epochs": 4,
        "head_mult": 5.0,
        "eval_every": 3000,
        "freeze_embed": False,
        "seed": 0,
    },
    {
        "stage": "phase5_full_laat",
        "method": "full",
        "lr": 2e-05,
        "epochs": 5,
        "head_mult": 30.0,
        "eval_every": 3000,
        "freeze_embed": True,
        "seed": 1,
    },
]


def run_laat(spec):
    global STAT_LIST, STAT_ID, N_STAT
    stage = spec["stage"]
    out_dir = RUN_DIR / stage
    if stage_done(stage):
        print(f"{stage} already done")
        return
    train5, dev5 = (load_lsi(P5["train"]), load_lsi(P5["dev"]))
    STAT_LIST = sorted({l for d in train5 + dev5 for l in d["labels"] or []})
    STAT_ID = {s: i for i, s in enumerate(STAT_LIST)}
    N_STAT = len(STAT_LIST)
    random.Random(SEED + spec["seed"]).shuffle(train5)
    if cfg.ft_max_docs:
        train5, dev5 = (train5[: cfg.ft_max_docs], dev5[: max(8, cfg.ft_max_docs // 3)])
    add_chunks(train5 + dev5, LSI_MAX_CHUNKS, LSI_HEAD_CHUNKS)
    freq = Counter((l for d in train5 for l in d["labels"]))
    print(
        f"LSI[{stage}]: {len(train5)} train, {len(dev5)} dev | {N_STAT} statutes | {np.mean([len(d['labels']) for d in train5]):.2f} labels/doc"
    )
    Ydev = multi_hot(dev5)
    dev_eval = dev5[:LSI_DEV_EVAL_CAP]
    Yev = Ydev[: len(dev_eval)]
    pos = np.array([freq.get(s, 0) for s in STAT_LIST], dtype=np.float32)
    pw = torch.tensor(
        np.clip((len(train5) - pos) / np.maximum(pos, 1), 1.0, LSI_POS_WEIGHT_CAP), dtype=torch.float32, device=DEVICE
    )
    lsi_loss = lambda logits, y: F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
    model = with_method(spec["method"], lambda: LabelAttentionClassifier(fresh_pretrained(), N_STAT))
    if spec["freeze_embed"]:
        for p in model.body.embed.parameters():
            p.requires_grad_(False)

    def dev_score(m):
        pr = lsi_probs(m, dev_eval)
        th = tune_thresholds(pr, Yev)
        f = ml_macro_f1(Yev, decide(pr, th))
        print(f"    dev macro-F1 {f:.4f} (tuned) | {ml_macro_f1(Yev, decide(pr, np.full(N_STAT, 0.5))):.4f} (all 0.5)")
        return f

    finetune(
        model,
        lambda: lsi_batches(train5, LSI_BATCH_DOCS, True),
        dev_score,
        spec["epochs"],
        spec["lr"],
        out_dir / "ckpt",
        stage,
        loss_fn=lsi_loss,
        head_lr_mult=spec["head_mult"],
        batches_per_epoch=math.ceil(len(train5) / LSI_BATCH_DOCS),
        eval_every=spec["eval_every"],
    )
    pr_dev = lsi_probs(model, dev5)
    th = tune_thresholds(pr_dev, Ydev)
    Pdev = decide(pr_dev, th)
    f_macro, f_micro = (ml_macro_f1(Ydev, Pdev), ml_micro_f1(Ydev, Pdev))
    print(f"{stage} full dev: macro-F1 {f_macro:.4f} | micro-F1 {f_micro:.4f}")
    test5 = load_lsi(P5["test"])
    if SMOKE:
        test5 = test5[:6]
    add_chunks(test5, LSI_MAX_CHUNKS, LSI_HEAD_CHUNKS)
    pr_test = lsi_probs(model, test5)
    Pt = decide(pr_test, th)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dev_probs.npy", pr_dev)
    np.save(out_dir / "test_probs.npy", pr_test)
    json.dump(
        {
            "stat_list": [int(s) for s in STAT_LIST],
            "dev_ids": [d["id"] for d in dev5],
            "test_ids": [d["id"] for d in test5],
        },
        open(out_dir / "label_index.json", "w"),
    )
    preds = [
        {"id": d["id"], "pred": sorted({int(STAT_LIST[j]) for j in np.where(row > 0)[0]})} for d, row in zip(test5, Pt)
    ]
    write_submission(preds, out_dir / "submission_phase5.zip")
    mark_done(stage, dev_macro_f1=f_macro, dev_micro_f1=f_micro)
    del model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


def tuned_f1(P, Y):
    return ml_macro_f1(Y, decide(P, tune_thresholds(P, Y)))


def logit(p):
    p = np.clip(p, 1e-05, 1 - 1e-05)
    return np.log(p / (1 - p))


def ensemble(stages, out_stage="phase5_ensemble", stack=True):
    global STAT_LIST, STAT_ID, N_STAT
    out_dir = RUN_DIR / out_stage
    avail = [s for s in stages if (RUN_DIR / s / "dev_probs.npy").exists()]
    assert avail, "no saved Phase 5 probabilities found"
    idx = json.load(open(RUN_DIR / avail[0] / "label_index.json"))
    for s in avail[1:]:
        other = json.load(open(RUN_DIR / s / "label_index.json"))
        assert other == idx, f"{s}: label or document order differs"
    STAT_LIST = idx["stat_list"]
    STAT_ID = {s: i for i, s in enumerate(STAT_LIST)}
    N_STAT = len(STAT_LIST)
    dev_by_id = {d["id"]: d for d in load_lsi(P5["dev"])}
    Ydev = multi_hot([dev_by_id[i] for i in idx["dev_ids"]])
    probs_dev = {s: np.load(RUN_DIR / s / "dev_probs.npy") for s in avail}
    probs_test = {s: np.load(RUN_DIR / s / "test_probs.npy") for s in avail}
    for s in avail:
        print(f"{s:20s} dev macro-F1 {tuned_f1(probs_dev[s], Ydev):.4f}")
    Pd = np.mean(list(probs_dev.values()), 0)
    Pt = np.mean(list(probs_test.values()), 0)
    f_avg = tuned_f1(Pd, Ydev)
    print(f"{'average':20s} dev macro-F1 {f_avg:.4f}")
    oof, St, f_stack = (Pd, Pt, f_avg)
    if stack:
        Xd, Xt = (logit(Pd), logit(Pt))
        fold = np.random.default_rng(SEED).integers(0, 2, len(Xd))
        oof, St = (Pd.copy(), Pt.copy())
        for j in range(N_STAT):
            if Ydev[:, j].sum() < 6:
                continue
            for f in (0, 1):
                tr, te = (fold != f, fold == f)
                if 0 < Ydev[tr, j].sum() < tr.sum():
                    oof[te, j] = (
                        LogisticRegression(C=0.5, max_iter=200, solver="liblinear")
                        .fit(Xd[tr], Ydev[tr, j])
                        .predict_proba(Xd[te])[:, 1]
                    )
            St[:, j] = (
                LogisticRegression(C=0.5, max_iter=200, solver="liblinear").fit(Xd, Ydev[:, j]).predict_proba(Xt)[:, 1]
            )
        f_stack = tuned_f1(oof, Ydev)
        print(f"{'stacked (cross-fit)':20s} dev macro-F1 {f_stack:.4f}")
    use_stack = f_stack > f_avg + 0.003
    P_dev_final, P_test_final = (oof, St) if use_stack else (Pd, Pt)
    Ptest = decide(P_test_final, tune_thresholds(P_dev_final, Ydev))
    print(
        f"using {('stacked' if use_stack else 'averaged')} probabilities | test {Ptest.sum(1).mean():.2f} statutes/doc"
    )
    preds = [
        {"id": i, "pred": sorted({int(STAT_LIST[j]) for j in np.where(row > 0)[0]})}
        for i, row in zip(idx["test_ids"], Ptest)
    ]
    write_submission(preds, out_dir / "submission_phase5.zip")
    json.dump(
        {"sources": avail, "avg_f1": f_avg, "stack_f1": f_stack, "used": "stack" if use_stack else "avg"},
        open(out_dir / "summary.json", "w"),
        indent=2,
    )


for spec in RUNS5:
    run_laat(spec)
ensemble([s["stage"] for s in RUNS5])
