from common import *

load_encoder()
P3_DIR = RUN_DIR / "phase3"
RR_EPOCHS, RR_LR = (6, 3e-05)
rr_vocab = json.load(open(P3["label_vocab"], encoding="utf-8"))
RR_LABELS = list(rr_vocab.keys()) if isinstance(rr_vocab, dict) else list(rr_vocab)
RR_ID = {l: i for i, l in enumerate(RR_LABELS)}
print(len(RR_LABELS), "rhetorical roles:", RR_LABELS)


def load_rr(path):
    raw = json.load(open(path, encoding="utf-8"))
    docs = []
    items = (
        raw
        if isinstance(raw, list)
        else [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in raw.items()]
    )
    for i, r in enumerate(items):
        docs.append({"id": str(r.get("id", i)), "text": r["text"], "labels": r.get("labels")})
    return docs


def rr_examples(docs, domain):
    exs = []
    dom = tok.encode(f"[{domain}] ", add_special_tokens=False)
    for d in docs:
        sents = [tok.encode(s, add_special_tokens=False) for s in d["text"]]
        for i, cur in enumerate(sents):
            prev = sents[i - 1] if i > 0 else []
            nxt = sents[i + 1] if i + 1 < len(sents) else []
            budget = cfg.ft_max_len - len(dom) - 2
            cur = cur[: budget * 2 // 3]
            side = (budget - len(cur)) // 2
            prev = prev[-side:] if side > 0 else []
            nxt = nxt[:side] if side > 0 else []
            ids = dom + prev + [EOS_ID] + cur + [EOS_ID] + nxt
            pool = [0] * (len(dom) + len(prev) + 1) + [1] * len(cur) + [0] * (1 + len(nxt))
            y = RR_ID[d["labels"][i]] if d.get("labels") else -100
            exs.append({"doc": d["id"], "i": i, "ids": ids, "pool": pool, "y": y})
    return exs


def rr_batches(exs, batch, shuffle):
    idx = list(range(len(exs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch):
        es = [exs[j] for j in idx[i : i + batch]]
        L = max((len(e["ids"]) for e in es))
        ids = torch.full((len(es), L), PAD_ID)
        pad = torch.zeros(len(es), L, dtype=torch.bool)
        pool = torch.zeros(len(es), L, dtype=torch.bool)
        for r, e in enumerate(es):
            n = len(e["ids"])
            ids[r, :n] = torch.tensor(e["ids"])
            pad[r, :n] = True
            pool[r, :n] = torch.tensor(e["pool"], dtype=torch.bool)
        yield (ids, pad, pool, torch.tensor([e["y"] for e in es]))


@torch.no_grad()
def rr_predict(task_model, exs):
    task_model.eval()
    preds = []
    for ids, pad, pool, _ in rr_batches(exs, cfg.ft_batch * 2, False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            preds.extend(task_model(ids.to(DEVICE), pad.to(DEVICE), pool.to(DEVICE)).argmax(-1).cpu().tolist())
    out = defaultdict(dict)
    for e, p in zip(exs, preds):
        out[e["doc"]][e["i"]] = RR_LABELS[p]
    return {d: [v[i] for i in range(len(v))] for d, v in out.items()}


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for l in labels:
        tp = sum((1 for t, p in zip(y_true, y_pred) if t == l and p == l))
        fp = sum((1 for t, p in zip(y_true, y_pred) if t != l and p == l))
        fn = sum((1 for t, p in zip(y_true, y_pred) if t == l and p != l))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


if not stage_done("phase3"):
    train_docs = {"CL": load_rr(P3["CL_train"]), "IT": load_rr(P3["IT_train"])}
    dev_docs = {"CL": load_rr(P3["CL_dev"]), "IT": load_rr(P3["IT_dev"])}
    if cfg.ft_max_docs:
        for k in train_docs:
            train_docs[k] = train_docs[k][: cfg.ft_max_docs]
            dev_docs[k] = dev_docs[k][: max(2, cfg.ft_max_docs // 5)]
    train_exs = rr_examples(train_docs["CL"], "CL") + rr_examples(train_docs["IT"], "IT")
    dev_exs = {k: rr_examples(dev_docs[k], k) for k in dev_docs}
    counts = Counter((e["y"] for e in train_exs))
    class_w = [
        (len(train_exs) / (len(RR_LABELS) * counts[i])) ** 0.5 if counts[i] else 1.0 for i in range(len(RR_LABELS))
    ]
    print(
        f"RR: {len(train_exs)} train sentences | dev CL {len(dev_exs['CL'])} IT {len(dev_exs['IT'])} | label dist: {[(RR_LABELS[i], c) for i, c in counts.most_common(6)]}"
    )
    rr_model = EncoderTaskModel(fresh_pretrained(), len(RR_LABELS), "seq")

    def dev_score(m):
        scores = {}
        for k, exs in dev_exs.items():
            pred = rr_predict(m, exs)
            y_true = [RR_LABELS[e["y"]] for e in exs]
            y_pred = [pred[e["doc"]][e["i"]] for e in exs]
            scores[k] = macro_f1(y_true, y_pred, RR_LABELS)
        print(f"    dev macro-F1 CL {scores['CL']:.4f} IT {scores['IT']:.4f}")
        return float(np.mean(list(scores.values())))

    best3 = finetune(
        rr_model,
        lambda: rr_batches(train_exs, cfg.ft_batch, True),
        dev_score,
        RR_EPOCHS,
        RR_LR,
        P3_DIR / "ckpt",
        "RR",
        class_weight=class_w,
    )
    preds3 = {}
    for k in ("CL", "IT"):
        tdocs = load_rr(P3[f"{k}_test"])
        if SMOKE:
            tdocs = tdocs[:3]
        p = rr_predict(rr_model, rr_examples(tdocs, k))
        for d in tdocs:
            preds3[d["id"]] = p.get(d["id"], ["None"] * len(d["text"]))
            assert len(preds3[d["id"]]) == len(d["text"]), f"sentence count mismatch for {d['id']}"
    write_submission(preds3, P3_DIR / "submission_phase3.zip")
    mark_done("phase3", dev_f1=best3)
    del rr_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
else:
    print("phase3 already done")
