from common import *

try:
    import ijson
except ImportError:
    os.system(f"{sys.executable} -m pip install -q ijson")
    import ijson
CLEAN_DIR = DATA_DIR / "clean"
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
TEXT_KEYS = ("text", "content", "judgment", "judgement", "body", "document", "raw_text", "full_text")
ID_KEYS = ("id", "doc_id", "case_id", "docid", "filename", "name", "title")


def detect_format(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        while not first.strip():
            first = f.readline()
            if not first:
                return "jsonl"
        if first.lstrip().startswith("["):
            return "array"
        try:
            json.loads(first)
        except json.JSONDecodeError:
            return "object"
        second = ""
        while not second.strip():
            second = f.readline()
            if not second:
                return "object"
        try:
            json.loads(second)
            return "jsonl"
        except json.JSONDecodeError:
            return "object"


def iter_raw_records(path):
    fmt = detect_format(path)
    with open(path, "rb") as f:
        if fmt == "array":
            for rec in ijson.items(f, "item"):
                yield (None, rec)
        elif fmt == "object":
            for k, v in ijson.kvitems(f, ""):
                yield (k, v)
        else:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                if line.strip():
                    yield (None, json.loads(line))


def extract_text(rec):
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        for k in TEXT_KEYS:
            v = rec.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v and all((isinstance(x, str) for x in v)):
                return "\n".join(v)
        strs = [v for v in rec.values() if isinstance(v, str)]
        if strs:
            return max(strs, key=len)
    if isinstance(rec, list) and rec and all((isinstance(x, str) for x in rec)):
        return "\n".join(rec)
    return ""


def extract_id(key, rec, fallback):
    if key is not None:
        return str(key)
    if isinstance(rec, dict):
        for k in ID_KEYS:
            if k in rec and rec[k] not in (None, ""):
                return str(rec[k])
    return fallback


CHAR_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "\xa0": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u3000": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\xad": "",
        "…": "...",
    }
)
CTRL_RE = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]")
HYPHEN_BREAK_RE = re.compile("(\\w)-\\n(?=[a-z])")
MULTISPACE_RE = re.compile("[ \\t]+")
MULTINEWLINE_RE = re.compile("\\n{3,}")
DROP_LINE_RE = re.compile(
    "|".join(
        [
            "^\\s*[\\(\\[]?\\s*\\d{1,4}\\s*[\\)\\]]?\\s*$",
            "^\\s*[-–—_=\\.\\*·]{0,4}\\s*\\d{1,4}\\s*[-–—_=\\.\\*·]{0,4}\\s*$",
            "^\\s*page\\s*(?:no\\.?)?\\s*\\d{1,4}\\s*(?:of\\s*\\d{1,4})?\\s*$",
            "^\\s*[ivxlcdm]{1,6}\\s*$",
            "^\\s*[\\.\\-_=\\*~·]{6,}\\s*$",
            "^\\s*(?:signature\\s+not\\s+verified|digitally\\s+signed\\s+by\\b.*|signing\\s+date:?.*|reason:\\s*.*)\\s*$",
            "^\\s*(?:\\d{1,2}[/.-]){2}\\d{2,4}\\s+\\d{1,2}:\\d{2}(?::\\d{2})?\\s*$",
            "^\\s*(?:www\\.|https?://)\\S+\\s*$",
        ]
    ),
    re.I,
)
EXTRA_DROP_LINES = set()
ALPHA_RE = re.compile("[^\\W\\d_]")


def drop_line(ln):
    return bool(ln) and (ln.lower() in EXTRA_DROP_LINES or DROP_LINE_RE.match(ln) is not None)


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").translate(CHAR_MAP)
    text = HYPHEN_BREAK_RE.sub("\\1", CTRL_RE.sub(" ", text))
    lines = [MULTISPACE_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not drop_line(ln)]
    return MULTINEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()


def quality(text):
    n_words = text.count(" ") + text.count("\n") + 1
    sample = text[:4000]
    nonspace = len(sample) - sample.count(" ") - sample.count("\n")
    return (n_words, len(ALPHA_RE.findall(sample)) / nonspace if nonspace else 0.0)


def split_of(doc_id):
    return (
        "val"
        if int(hashlib.sha1(f"{SEED}:{doc_id}".encode()).hexdigest()[:8], 16) / 2**32 < cfg.val_fraction
        else "train"
    )


def source_marker(source):
    return CLEAN_DIR / f"{source}.done.json"


def source_files(source):
    return {s: CLEAN_DIR / s / f"{source}.jsonl.gz" for s in ("train", "val")}


def iter_gz_tolerant(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return
    except (EOFError, OSError, gzip.BadGzipFile):
        return


def recover_source(source):
    st = defaultdict(int)
    t0 = time.time()
    for split, path in source_files(source).items():
        if not path.exists():
            continue
        tmp = path.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            for rec in iter_gz_tolerant(path):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st["kept_" + split] += 1
                st["words_" + split] += rec["n_words"]
        os.replace(tmp, path)
    st["read"] = st["kept_train"] + st["kept_val"]
    st["recovered"] = 1
    print(
        f"{source:14s} recovered: {st['read']:,} docs ({st['words_train'] / 1000000.0:.1f}M train words) {time.time() - t0:.0f}s"
    )
    return st


def clean_source(source, path, exact_keys):
    st = defaultdict(int)
    t0 = time.time()
    seen_ids = set()
    files = source_files(source)
    for p in files.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.unlink(missing_ok=True)
    writers = {s: gzip.open(p, "wt", encoding="utf-8") for s, p in files.items()}
    try:
        for idx, (key, rec) in enumerate(iter_raw_records(path)):
            if cfg.max_docs_per_source and idx >= cfg.max_docs_per_source:
                break
            st["read"] += 1
            raw = extract_text(rec)
            if not raw:
                st["empty"] += 1
                continue
            raw_id = extract_id(key, rec, str(idx))
            if raw_id in seen_ids:
                st["id_collisions"] += 1
            seen_ids.add(raw_id)
            doc_id = f"{source}:{idx}:{raw_id}"
            text = clean_text(raw)
            n_words, alpha = quality(text)
            if n_words < cfg.min_words:
                st["too_short"] += 1
                continue
            if alpha < cfg.min_alpha_ratio:
                st["low_alpha"] += 1
                continue
            k = hashlib.sha1(text[:20000].lower().encode()).hexdigest()
            if k in exact_keys:
                st["dup_exact"] += 1
                continue
            exact_keys.add(k)
            split = split_of(doc_id)
            writers[split].write(
                json.dumps({"id": doc_id, "source": source, "text": text, "n_words": n_words}, ensure_ascii=False)
                + "\n"
            )
            st["kept_" + split] += 1
            st["words_" + split] += n_words
            if st["read"] % 10000 == 0:
                print(
                    f"{source:14s} read {st['read']:>9,} kept {st['kept_train'] + st['kept_val']:>9,} {time.time() - t0:6.0f}s"
                )
    finally:
        for w in writers.values():
            w.close()
    print(
        f"{source:14s} done: read {st['read']:,} kept {st['kept_train'] + st['kept_val']:,} ({st['words_train'] / 1000000.0:.1f}M train words, {st['dup_exact']} exact dups, {st['id_collisions']} id collisions) {time.time() - t0:.0f}s"
    )
    return st


def run_clean():
    if (CLEAN_DIR / "manifest.json").exists():
        print("clean/ complete, skipping")
        return json.load(open(CLEAN_DIR / "manifest.json"))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    exact_keys = set()
    stats = {}
    for source in SOURCES:
        if source_marker(source).exists():
            stats[source] = json.load(open(source_marker(source)))
            print(f"{source:14s} already done")
            continue
        files = source_files(source)
        if P1_TRAIN.get(source) is None and files["train"].exists():
            st = recover_source(source)
        elif files["train"].exists() and os.path.getsize(files["train"]) > 50000000:
            st = recover_source(source)
        else:
            st = clean_source(source, P1_TRAIN[source], exact_keys)
        stats[source] = dict(st)
        json.dump(stats[source], open(source_marker(source), "w"), indent=2)
    man = {"stats": stats, "config": asdict(cfg)}
    json.dump(man, open(CLEAN_DIR / "manifest.json", "w"), indent=2)
    return man


clean_manifest = run_clean()
print({s: f"{v.get('words_train', 0) / 1000000.0:.0f}M words" for s, v in clean_manifest["stats"].items()})


def iter_clean(split, sources=None):
    for path in sorted((CLEAN_DIR / split).glob("*.jsonl.gz")):
        if sources and path.name.split(".")[0] not in sources:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


TOKENS_DIR = DATA_DIR / "tokens"


def tokenize_corpus():
    if (TOKENS_DIR / "manifest.json").exists():
        man = json.load(open(TOKENS_DIR / "manifest.json"))
        for split in man:
            for v in man[split].values():
                for f in v["files"]:
                    f["bin"] = str(TOKENS_DIR / split / Path(f["bin"]).name)
                    f["idx"] = str(TOKENS_DIR / split / Path(f["idx"]).name)
        print("tokens/ exists, skipping")
        return man
    man = {}
    for split in ("val", "train"):
        (TOKENS_DIR / split).mkdir(parents=True, exist_ok=True)
        man[split] = {}
        t0 = time.time()
        for source in P1_TRAIN:
            buf, buf_len, starts, shard, files, n_docs = ([], 0, [], 0, [], 0)

            def flush():
                nonlocal buf, buf_len, starts, shard
                if not buf:
                    return
                stem = TOKENS_DIR / split / f"{source}_{shard:03d}"
                np.concatenate(buf).tofile(str(stem) + ".bin")
                np.asarray(starts, dtype=np.uint64).tofile(str(stem) + ".idx")
                files.append(
                    {"bin": str(stem) + ".bin", "idx": str(stem) + ".idx", "tokens": int(buf_len), "docs": len(starts)}
                )
                buf, buf_len, starts, shard = ([], 0, [], shard + 1)

            batch = []

            def enc_batch():
                nonlocal buf_len, n_docs
                for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                    ids = ids + [EOS_ID]
                    starts.append(buf_len)
                    buf.append(np.asarray(ids, dtype=np.uint16))
                    buf_len += len(ids)
                    n_docs += 1
                    if buf_len >= cfg.tokens_per_shard:
                        flush()
                batch.clear()

            for rec in iter_clean(split, [source]):
                batch.append(rec["text"])
                if len(batch) >= 256:
                    enc_batch()
            enc_batch()
            flush()
            man[split][source] = {"docs": n_docs, "tokens": sum((f["tokens"] for f in files)), "files": files}
            print(
                f"{split:5s} {source:14s} {n_docs:>8,} docs {man[split][source]['tokens'] / 1000000.0:9.1f}M tokens {time.time() - t0:5.0f}s"
            )
    json.dump(man, open(TOKENS_DIR / "manifest.json", "w"), indent=2)
    return man


token_manifest = tokenize_corpus()
N_TRAIN_TOKENS = sum((v["tokens"] for v in token_manifest["train"].values()))
print(
    f"train tokens: {N_TRAIN_TOKENS / 1000000000.0:.3f}B | val tokens: {sum((v['tokens'] for v in token_manifest['val'].values())) / 1000000.0:.1f}M"
)
if PREP_ONLY:
    raise SystemExit("Data prep complete; set PREP_ONLY=False to train.")
model = EncoderDecoder(cfg, VOCAB_SIZE).to(DEVICE)
print(f"parameters: {n_params(model) / 1000000.0:.1f}M (embedding {model.embed.weight.numel() / 1000000.0:.1f}M)")
if not SMOKE:
    assert 50000000.0 <= n_params(model) <= 300000000.0


class TokenStore:

    def __init__(self, split):
        self.shards = []
        for source, v in token_manifest[split].items():
            for f in v["files"]:
                self.shards.append((np.memmap(f["bin"], dtype=np.uint16, mode="r"), source))
        self.sizes = np.array([len(a) for a, _ in self.shards], dtype=np.float64)
        self.p = self.sizes / self.sizes.sum()

    def window(self, rng, length):
        i = rng.choice(len(self.shards), p=self.p)
        arr, _ = self.shards[i]
        if len(arr) <= length:
            return np.array(arr, dtype=np.int64)
        s = rng.integers(0, len(arr) - length)
        return np.array(arr[s : s + length], dtype=np.int64)


def make_continuation(rng, store):
    src_len = int(rng.integers(cfg.max_src_len // 4, cfg.max_src_len + 1))
    tgt_len = int(rng.integers(cfg.max_tgt_len // 2, cfg.max_tgt_len + 1))
    w = store.window(rng, src_len + tgt_len)
    if len(w) < src_len + 2:
        src_len = max(1, len(w) // 2)
    src, tgt = (w[:src_len].tolist(), w[src_len:].tolist())
    return (src, [BOS_ID] + tgt[:-1], tgt)


def make_span_corruption(rng, store):
    L = int(rng.integers(cfg.max_src_len // 2, cfg.max_src_len + 1))
    w = store.window(rng, L).tolist()
    L = len(w)
    n_mask = max(1, int(L * cfg.mask_ratio))
    n_spans = max(1, n_mask // cfg.mean_span)
    starts = sorted(rng.choice(L, size=min(n_spans, L), replace=False).tolist())
    spans, last_end = ([], -1)
    for s in starts:
        if s <= last_end:
            continue
        ln = max(1, int(rng.poisson(cfg.mean_span)))
        e = min(L, s + ln)
        spans.append((s, e))
        last_end = e
        if len(spans) >= 99:
            break
    src, tgt, pos = ([], [], 0)
    for i, (s, e) in enumerate(spans):
        src += w[pos:s] + [SENTINEL_IDS[i]]
        tgt += [SENTINEL_IDS[i]] + w[s:e]
        pos = e
    src += w[pos:]
    tgt += [SENTINEL_IDS[len(spans)]]
    return (src, [BOS_ID] + tgt, tgt + [EOS_ID])


def collate(examples):
    S = max((len(s) for s, _, _ in examples))
    T = max((len(t) for _, t, _ in examples))
    src = torch.full((len(examples), S), PAD_ID)
    src_pad = torch.zeros(len(examples), S, dtype=torch.bool)
    tin = torch.full((len(examples), T), PAD_ID)
    tout = torch.full((len(examples), T), -100)
    tpad = torch.zeros(len(examples), T, dtype=torch.bool)
    for i, (s, ti, to) in enumerate(examples):
        src[i, : len(s)] = torch.tensor(s)
        src_pad[i, : len(s)] = True
        tin[i, : len(ti)] = torch.tensor(ti)
        tout[i, : len(to)] = torch.tensor(to)
        tpad[i, : len(ti)] = True
    return (src, src_pad, tin, tout, tpad)


def make_batch(store, step, batch_size, seed_offset=0):
    rng = np.random.default_rng(SEED + seed_offset + step * 7919)
    exs = [
        make_span_corruption(rng, store) if rng.random() < cfg.span_corruption_prob else make_continuation(rng, store)
        for _ in range(batch_size)
    ]
    return collate(exs)


train_store, val_store = (TokenStore("train"), TokenStore("val"))
src, src_pad, tin, tout, tpad = make_batch(train_store, 0, 2)
print("src:", src.shape, "tgt:", tin.shape)
print("SRC >>", tok.decode(src[0][src_pad[0]].tolist())[:300])
print("TGT >>", tok.decode([t for t in tout[0].tolist() if t >= 0])[:300])
import csv

DECAY_FRACTION = 0.15


def lr_at(step):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_start = int(cfg.max_steps * (1 - DECAY_FRACTION))
    if step < decay_start:
        return cfg.lr
    prog = min(1.0, (step - decay_start) / max(1, cfg.max_steps - decay_start))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, lr, wd):
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        fused=DEVICE.type == "cuda",
    )


def save_ckpt(path, model, opt, scaler, step, best_val, extra=None):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict() if opt else None,
            "scaler": scaler.state_dict() if scaler else None,
            "step": step,
            "best_val": best_val,
            "cfg": asdict(cfg),
            "extra": extra or {},
        },
        tmp,
    )
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, store, n_batches):
    model.eval()
    losses = []
    for i in range(n_batches):
        b = [t.to(DEVICE) for t in make_batch(store, i, cfg.batch_size, seed_offset=10000000)]
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            losses.append(model(*b).item())
    model.train()
    return float(np.mean(losses))


def pretrain(model, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt = make_optimizer(model, cfg.lr, cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and AMP_DTYPE == torch.float16)
    step, best_val = (0, float("inf"))
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best_val = (ck["step"], ck["best_val"])
        if ck["scaler"]:
            scaler.load_state_dict(ck["scaler"])
        print(f"resumed from step {step} (best val {best_val:.3f}); training to max_steps={cfg.max_steps}")
    if step >= cfg.max_steps:
        print(f"already at step {step} >= max_steps {cfg.max_steps}; raise cfg.max_steps to continue")
        return best_val
    log_path = ckpt_dir / "log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "lr", "train_loss", "val_loss", "tok_per_s"])
    model.train()
    t0 = time.time()
    tokens_seen = 0
    run_loss = []
    while step < cfg.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum):
            b = [
                t.to(DEVICE, non_blocking=True)
                for t in make_batch(train_store, step * cfg.grad_accum + micro, cfg.batch_size)
            ]
            tokens_seen += int(b[1].sum() + b[4].sum())
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                loss = model(*b) / cfg.grad_accum
            scaler.scale(loss).backward()
            run_loss.append(loss.item() * cfg.grad_accum)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % cfg.log_every == 0:
            tps = tokens_seen / (time.time() - t0)
            print(
                f"step {step:6d} | lr {lr_at(step):.2e} | loss {np.mean(run_loss[-cfg.log_every * cfg.grad_accum:]):.4f} | {tps / 1000.0:6.1f}k tok/s"
            )
        val = ""
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val = evaluate(model, val_store, cfg.eval_batches)
            print(f"           val loss {val:.4f}  ppl {math.exp(val):.1f}")
            if val < best_val:
                best_val = val
                save_ckpt(ckpt_dir / "best.pt", model, None, None, step, best_val)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [step, lr_at(step), np.mean(run_loss[-cfg.grad_accum :]), val, tokens_seen / (time.time() - t0)]
            )
        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            save_ckpt(latest, model, opt, scaler, step, best_val)
    return best_val


def plot_log(ckpt_dir):
    import pandas as pd, matplotlib.pyplot as plt

    df = pd.read_csv(ckpt_dir / "log.csv")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df.step, df.train_loss.rolling(max(1, cfg.log_every)).mean(), label="train")
    v = df[df.val_loss.notna()]
    if len(v):
        ax.plot(v.step, v.val_loss.astype(float), "o-", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ckpt_dir / "loss.png", dpi=120)
    plt.show()


def sanity_tests():
    torch.manual_seed(0)
    tiny = Config(**{**asdict(cfg), "dropout": 0.0})
    m = EncoderDecoder(tiny, VOCAB_SIZE).to(DEVICE)
    m.eval()
    src, src_pad, tin, tout, tpad = [t.to(DEVICE) for t in make_batch(train_store, 1, 2)]
    with torch.no_grad():
        mem = m.encode(src, src_pad)
        base = m.decode(tin, mem, src_pad, tpad)
        tin2 = tin.clone()
        j = min(5, tin.shape[1] - 1)
        tin2[:, j] = (tin2[:, j] + 7) % VOCAB_SIZE
        pert = m.decode(tin2, mem, src_pad, tpad)
    assert torch.allclose(base[:, :j], pert[:, :j], atol=0.0001), "causal leak in decoder!"
    assert not torch.allclose(base[:, j:], pert[:, j:], atol=0.0001), "decoder ignores its input?"
    with torch.no_grad():
        src2 = src.clone()
        src2[:, 1] = (src2[:, 1] + 7) % VOCAB_SIZE
        assert not torch.allclose(
            m.encode(src, src_pad)[:, 0], m.encode(src2, src_pad)[:, 0], atol=1e-05
        ), "encoder not bidirectional?"
    with torch.no_grad():
        padded = torch.cat([src, torch.full((src.size(0), 3), PAD_ID, device=DEVICE)], 1)
        ppad = torch.cat([src_pad, torch.zeros(src.size(0), 3, dtype=torch.bool, device=DEVICE)], 1)
        assert torch.allclose(
            m.encode(src, src_pad), m.encode(padded, ppad)[:, : src.size(1)], atol=0.0001
        ), "padding leaks into attention"
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=0.0003)
    first = None
    for i in range(100):
        loss = m(src, src_pad, tin, tout, tpad)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    print(f"overfit one batch: {first:.3f} -> {loss.item():.3f}")
    assert loss.item() < first * 0.5, "cannot overfit a single batch: check the loss/targets"
    print("all sanity tests passed")


sanity_tests()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory after sanity tests: {torch.cuda.memory_allocated() / 1000000000.0:.2f} GB allocated")
P1_DIR = RUN_DIR / "phase1"
CKPT_DIR = P1_DIR / "ckpt"
if not stage_done("phase1_train"):
    best = pretrain(model, CKPT_DIR)
    plot_log(CKPT_DIR)
    mark_done("phase1_train", best_val=best)
else:
    print("phase1 training already done")
best_ck = torch.load(CKPT_DIR / "best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(best_ck["model"])
print(f"loaded best checkpoint (step {best_ck['step']}, val {best_ck['best_val']:.3f})")


def load_test_prompts(path):
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): v["text"] if isinstance(v, dict) else v for k, v in raw.items()}
    return {str(r.get("id", i)): r["text"] for i, r in enumerate(raw)}


def cut_words(text, n):
    return " ".join(text.split()[:n])


def generate_continuations(model, prompts, batch_size=8):
    out = {}
    ids = list(prompts)
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        enc = [tok.encode(prompts[k], add_special_tokens=False)[-cfg.max_src_len :] for k in chunk]
        S = max((len(e) for e in enc))
        src = torch.full((len(enc), S), PAD_ID)
        pad = torch.zeros(len(enc), S, dtype=torch.bool)
        for j, e in enumerate(enc):
            src[j, : len(e)] = torch.tensor(e)
            pad[j, : len(e)] = True
        stop = lambda o: torch.tensor(
            [len(tok.decode(row.tolist(), skip_special_tokens=True).split()) >= cfg.target_words for row in o],
            device=o.device,
        )
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            gen = model.generate(
                src.to(DEVICE),
                pad.to(DEVICE),
                cfg.max_gen_tokens,
                cfg.temperature,
                cfg.top_p,
                cfg.repetition_penalty,
                stop_fn=stop,
                ban_eos=True,
            )
        for k, row in zip(chunk, gen):
            text = tok.decode(row.tolist(), skip_special_tokens=True).strip()
            out[k] = cut_words(re.sub("\\s+", " ", text), cfg.target_words)
        if i // batch_size % 25 == 0:
            print(f"generated {min(i + batch_size, len(ids))}/{len(ids)}")
    return out


def write_submission(pred, path_zip):
    path_zip.parent.mkdir(parents=True, exist_ok=True)
    pj = path_zip.parent / "prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    print("wrote", path_zip)


if not stage_done("phase1"):
    prompts = load_test_prompts(P1_TEST)
    if SMOKE:
        prompts = dict(list(prompts.items())[:4])
    preds1 = {k: {"text": v} for k, v in generate_continuations(model, prompts).items()}
    assert set(preds1) == set(prompts), "prediction ids must match test ids exactly"
    wc = [len(v["text"].split()) for v in preds1.values()]
    print(
        f"words per continuation: min {min(wc)} max {max(wc)} | short (<{cfg.target_words}): {sum((w < cfg.target_words for w in wc))}"
    )
    k0 = next(iter(preds1))
    print(f"\n[{k0}] ...{prompts[k0][-200:]}\n>>> {preds1[k0]['text'][:400]}")
    write_submission(preds1, P1_DIR / "submission_phase1.zip")
    mark_done("phase1", n=len(preds1))
else:
    print("phase1 submission already done")
