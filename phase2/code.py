from common import *

load_encoder()
P2_DIR = RUN_DIR / "phase2"
NER_EPOCHS, NER_LR, NER_O_WEIGHT = (10, 5e-05, 0.15)
NER_LABELS = [l.strip() for l in open(P2["labels"], encoding="utf-8") if l.strip()]
if NER_LABELS and NER_LABELS[0].startswith("["):
    NER_LABELS = json.load(open(P2["labels"]))
BIO = ["O"] + [f"{p}-{l}" for l in NER_LABELS for p in ("B", "I")]
BIO_ID = {t: i for i, t in enumerate(BIO)}
print(len(NER_LABELS), "entity labels ->", len(BIO), "BIO tags")


def load_texts(path):
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): v if isinstance(v, str) else extract_text(v) for k, v in raw.items()}
    return {str(r.get("id", i)): extract_text(r) for i, r in enumerate(raw)}


def load_gold(path):
    raw = json.load(open(path, encoding="utf-8"))
    out = {}

    def norm_ent(e):
        if isinstance(e, (list, tuple)):
            return (int(e[0]), int(e[1]), str(e[2]), str(e[3]) if len(e) > 3 else None)
        if isinstance(e, dict):
            s = e.get("start", e.get("start_char", e.get("begin")))
            t = e.get("end", e.get("end_char"))
            l = e.get("label", e.get("type", e.get("entity")))
            txt = e.get("text", e.get("entity_text"))
            if s is None and "span" in e:
                s, t = e["span"]
            return (int(s), int(t), str(l), str(txt) if txt is not None else None)
        raise ValueError(f"unknown entity format {e!r}")

    items = (
        raw.items()
        if isinstance(raw, dict)
        else [
            (str(r.get("id", i)), r.get("entities", r.get("annotations", r.get("gold", r)))) for i, r in enumerate(raw)
        ]
    )
    for k, ents in items:
        if isinstance(ents, dict):
            ents = ents.get("entities", ents.get("annotations", []))
        out[str(k)] = [norm_ent(e) for e in ents]
    return out


def check_alignment(texts, gold):
    ok = tot = 0
    bad = []
    for d, ents in gold.items():
        if d not in texts:
            continue
        for s, e, l, t in ents:
            if t is None:
                continue
            tot += 1
            same = texts[d][s:e].strip() == t.strip()
            ok += same
            if not same and len(bad) < 3:
                bad.append((d, s, e, t[:40], texts[d][s:e][:40]))
    if tot:
        print(
            f"gold/text offset alignment: {ok / tot:.1%} of {tot} entities"
            + ("" if ok / tot > 0.95 else f"  <-- MISALIGNED, e.g. {bad}")
        )


def ner_windows(doc_id, text, ents=None):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = (enc["input_ids"], enc["offset_mapping"])
    if not ids:
        return []
    labels = [0] * len(ids)
    if ents is not None:
        for s, e, lab, _ in ents:
            if f"B-{lab}" not in BIO_ID:
                continue
            first = True
            for i, (ts, te) in enumerate(offs):
                if te <= s or ts >= e:
                    continue
                labels[i] = BIO_ID[("B-" if first else "I-") + lab]
                first = False
    W, stride = (cfg.ft_max_len, cfg.ft_max_len // 2)
    wins = []
    for st in range(0, max(1, len(ids)), stride):
        wins.append(
            {
                "doc": doc_id,
                "start": st,
                "ids": ids[st : st + W],
                "offs": offs[st : st + W],
                "labels": labels[st : st + W],
            }
        )
        if st + W >= len(ids):
            break
    return wins


def ner_batches(windows, batch, shuffle):
    idx = list(range(len(windows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch):
        ws = [windows[j] for j in idx[i : i + batch]]
        L = max((len(w["ids"]) for w in ws))
        ids = torch.full((len(ws), L), PAD_ID)
        pad = torch.zeros(len(ws), L, dtype=torch.bool)
        y = torch.full((len(ws), L), -100)
        for r, w in enumerate(ws):
            n = len(w["ids"])
            ids[r, :n] = torch.tensor(w["ids"])
            pad[r, :n] = True
            y[r, :n] = torch.tensor(w["labels"])
        yield (ids, pad, y)


@torch.no_grad()
def ner_predict(task_model, docs):
    task_model.eval()
    out = {}
    for doc_id, text in docs.items():
        wins = ner_windows(doc_id, text)
        n_tok = wins[-1]["start"] + len(wins[-1]["ids"]) if wins else 0
        tag = np.zeros(n_tok, dtype=np.int64)
        conf = np.full(n_tok, -1000000000.0)
        for b_start in range(0, len(wins), cfg.ft_batch):
            ws = wins[b_start : b_start + cfg.ft_batch]
            ids, pad, _ = next(ner_batches(ws, len(ws), False))
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                lp = F.log_softmax(task_model(ids.to(DEVICE), pad.to(DEVICE)).float(), -1).cpu().numpy()
            for r, w in enumerate(ws):
                n = len(w["ids"])
                arg = lp[r, :n].argmax(-1)
                centre = -np.abs(np.arange(n) - n / 2)
                for i in range(n):
                    if centre[i] > conf[w["start"] + i]:
                        conf[w["start"] + i] = centre[i]
                        tag[w["start"] + i] = arg[i]
        offs = (
            wins[0]["offs"]
            if len(wins) == 1
            else tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        )
        spans, i = ([], 0)
        while i < n_tok:
            t = BIO[tag[i]]
            if t == "O":
                i += 1
                continue
            lab = t[2:]
            j = i + 1
            while j < n_tok and BIO[tag[j]] == f"I-{lab}":
                j += 1
            s, e = (offs[i][0], offs[j - 1][1])
            while s < e and text[s].isspace():
                s += 1
            if e > s:
                spans.append([int(s), int(e), lab])
            i = j
        out[doc_id] = spans
    return out


def strict_macro_f1(pred, gold):
    tp, fp, fn = (Counter(), Counter(), Counter())
    for d in gold:
        g = {tuple(x[:3]) for x in gold[d]}
        p = {tuple(x[:3]) for x in pred.get(d, [])}
        for x in p & g:
            tp[x[2]] += 1
        for x in p - g:
            fp[x[2]] += 1
        for x in g - p:
            fn[x[2]] += 1
    f1s = {}
    for l in NER_LABELS:
        pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        rc = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        f1s[l] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return (float(np.mean(list(f1s.values()))), f1s)


if not stage_done("phase2"):
    texts = {**load_texts(P2["fold1_text"]), **load_texts(P2["fold2_text"])}
    gold = {**load_gold(P2["fold1_gold"]), **load_gold(P2["fold2_gold"])}
    check_alignment(texts, gold)
    ids_all = [d for d in texts if d in gold]
    random.Random(SEED).shuffle(ids_all)
    if cfg.ft_max_docs:
        ids_all = ids_all[: cfg.ft_max_docs]
    n_dev = max(1, len(ids_all) // 10)
    dev_ids, tr_ids = (ids_all[:n_dev], ids_all[n_dev:])
    train_wins = [w for d in tr_ids for w in ner_windows(d, texts[d], gold[d])]
    tag_counts = Counter((t for w in train_wins for t in w["labels"]))
    print(
        f"NER: {len(tr_ids)} train docs -> {len(train_wins)} windows | {len(dev_ids)} dev docs | {sum((len(gold[d]) for d in tr_ids))} entities | B-tags {sum((c for t, c in tag_counts.items() if BIO[t].startswith('B-')))}"
    )
    class_w = [NER_O_WEIGHT] + [1.0] * (len(BIO) - 1)
    ner_model = EncoderTaskModel(fresh_pretrained(), len(BIO), "token")
    dev_docs = {d: texts[d] for d in dev_ids}
    dev_gold = {d: gold[d] for d in dev_ids}
    n_gold_dev = sum((len(v) for v in dev_gold.values()))

    def dev_score(m):
        pred = ner_predict(m, dev_docs)
        f, _ = strict_macro_f1(pred, dev_gold)
        print(
            f"    dev: predicted {sum((len(v) for v in pred.values()))} spans vs {n_gold_dev} gold, strict macro-F1 {f:.4f}"
        )
        return f

    best2 = finetune(
        ner_model,
        lambda: ner_batches(train_wins, cfg.ft_batch, True),
        dev_score,
        NER_EPOCHS,
        NER_LR,
        P2_DIR / "ckpt",
        "NER",
        class_weight=class_w,
    )
    f1, per = strict_macro_f1(ner_predict(ner_model, dev_docs), dev_gold)
    print(
        f"dev strict macro-F1 {f1:.4f}\n"
        + "\n".join((f"  {l:10s} {v:.3f}" for l, v in sorted(per.items(), key=lambda x: -x[1])))
    )
    test_docs = load_texts(P2["fold3_text"])
    if SMOKE:
        test_docs = dict(list(test_docs.items())[:5])
    preds2 = ner_predict(ner_model, test_docs)
    for d in test_docs:
        preds2.setdefault(d, [])
    assert set(preds2) == set(test_docs)
    print(f"test: {sum((len(v) for v in preds2.values()))} spans over {len(preds2)} docs")
    write_submission(preds2, P2_DIR / "submission_phase2.zip")
    mark_done("phase2", dev_f1=f1)
    del ner_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
else:
    print("phase2 already done")
