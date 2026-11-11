import os, sys, re, io, json, gzip, hashlib, random, time, math, zipfile, shutil, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
import numpy as np

SMOKE = False
PREP_ONLY = False
RUN_NAME = "encdec_v1"
SEED = 42
RESET_STAGES = []
PREV_DATA = "/kaggle/input/notebooks/ihitesh/project-phase-1/cs781/data/clean"
PREPARED_DATA = os.environ.get("CS781_PREPARED_DATA", "/content/drive/MyDrive/cs781/data")
PREV_RUNS = "/kaggle/input/notebooks/ihitesh/project-phase-1/cs781/runs"
LOCAL_DATA_ROOT = os.environ.get("CS781_LOCAL_DATA")


def detect_environment():
    if os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "Kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "Google Colab"
    return "Local"


ENVIRONMENT = detect_environment()
if ENVIRONMENT == "Kaggle":
    OUTPUT_DIR = Path("/kaggle/working/cs781")
elif ENVIRONMENT == "Google Colab":
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path("/content/drive/MyDrive/cs781")
else:
    OUTPUT_DIR = Path(os.environ.get("CS781_OUT", "cs781")).resolve()
if not SMOKE and os.path.exists(os.path.join(PREPARED_DATA, "tokens", "manifest.json")):
    DATA_DIR = Path(PREPARED_DATA)
    print("using prepared data from", DATA_DIR)
else:
    DATA_DIR = OUTPUT_DIR / ("data_smoke" if SMOKE else "data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SMOKE and os.path.exists(PREV_DATA) and (not (DATA_DIR / "clean").exists()):
        print("copying partial clean/ from previous run ...")
        t0 = time.time()
        shutil.copytree(PREV_DATA, DATA_DIR / "clean")
        print(f"copied in {time.time() - t0:.0f}s")
RUN_DIR = OUTPUT_DIR / ("smoke" if SMOKE else "runs") / RUN_NAME
if ENVIRONMENT == "Kaggle" and os.path.exists(PREV_RUNS) and (not (OUTPUT_DIR / "runs").exists()):
    shutil.copytree(PREV_RUNS, OUTPUT_DIR / "runs")
    print("restored previous runs/ from", PREV_RUNS)
    for d in (OUTPUT_DIR / "runs").rglob("submission_phase*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    max_docs_per_source: int = None
    val_fraction: float = 0.01
    min_words: int = 50
    min_alpha_ratio: float = 0.6
    near_dedup: bool = False
    tokens_per_shard: int = 100000000
    d_model: int = 768
    n_heads: int = 12
    n_enc_layers: int = 8
    n_dec_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_src_len: int = 1024
    max_tgt_len: int = 256
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 0.0003
    min_lr: float = 3e-05
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 15000
    eval_every: int = 500
    eval_batches: int = 20
    ckpt_every: int = 250
    log_every: int = 20
    span_corruption_prob: float = 0.5
    mask_ratio: float = 0.15
    mean_span: int = 3
    target_words: int = 128
    max_gen_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    ft_method: str = "full"
    lora_rank: int = 16
    lora_alpha: int = 32
    ft_lr: float = 3e-05
    ft_epochs: int = 3
    ft_batch: int = 8
    ft_max_len: int = 512
    ft_max_docs: int = None
    cjpe_max_chunks: int = 12
    cjpe_head_chunks: int = 1
    cjpe_batch_docs: int = 4


cfg = Config()
if SMOKE:
    cfg = Config(
        max_docs_per_source=300,
        tokens_per_shard=2000000,
        d_model=128,
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=2,
        d_ff=256,
        dropout=0.0,
        max_src_len=256,
        max_tgt_len=64,
        batch_size=8,
        grad_accum=1,
        warmup_steps=10,
        max_steps=60,
        eval_every=20,
        eval_batches=2,
        ckpt_every=20,
        log_every=5,
        max_gen_tokens=60,
        ft_epochs=1,
        ft_batch=8,
        ft_max_len=256,
        ft_max_docs=30,
        cjpe_max_chunks=3,
        cjpe_batch_docs=2,
    )


def seed_everything(s=SEED):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch

        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


seed_everything()
print(f"env={ENVIRONMENT} smoke={SMOKE} prep_only={PREP_ONLY}\ndata={DATA_DIR}\nrun ={RUN_DIR}")
json.dump(asdict(cfg), open(RUN_DIR / "config.json", "w"), indent=2)


def marker(stage):
    return RUN_DIR / stage / "done.json"


def stage_done(stage):
    return marker(stage).exists()


def mark_done(stage, **info):
    marker(stage).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"time": time.time(), **info}, open(marker(stage), "w"), indent=2)


for s in RESET_STAGES:
    marker(s).unlink(missing_ok=True)
    if s in ("phase2", "phase3", "phase4"):
        shutil.rmtree(RUN_DIR / s / "ckpt", ignore_errors=True)
if RESET_STAGES:
    print("reset stages:", RESET_STAGES)
HF_REPO = "Exploration-Lab/CS-781-Capstone"
PHASE_PREFIX = {
    1: "Phase_1_Pretraining/",
    2: "Phase_2_LNER/",
    3: "Phase_3_RR/",
    4: "Phase_4_CJPE/",
    5: "Phase_5_LSI/",
    6: "Phase_6_Summ/",
    7: "Phase_7_PCR/",
}


def get_hf_token():
    if ENVIRONMENT == "Kaggle":
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF")
    if ENVIRONMENT == "Google Colab":
        from google.colab import userdata

        return userdata.get("HF")
    return os.environ.get("HF_TOKEN")


RAW = {ph: {} for ph in PHASE_PREFIX}
if LOCAL_DATA_ROOT:
    root = Path(LOCAL_DATA_ROOT)
    for ph, pre in PHASE_PREFIX.items():
        for p in (root / pre.rstrip("/")).rglob("*"):
            if p.is_file():
                RAW[ph][str(p.relative_to(root))] = str(p)
else:
    from huggingface_hub import hf_hub_download, list_repo_files, login

    token = get_hf_token()
    if not token:
        raise ValueError("No Hugging Face token: set Kaggle secret 'HF', Colab userdata 'HF', or env HF_TOKEN.")
    login(token=token)
    repo_files = list_repo_files(HF_REPO, repo_type="dataset", token=token)
    skip_phase1_train = (DATA_DIR / "clean" / "manifest.json").exists() or (
        DATA_DIR / "tokens" / "manifest.json"
    ).exists()
    for ph, pre in PHASE_PREFIX.items():
        for f in [x for x in repo_files if x.startswith(pre)]:
            if ph == 1 and skip_phase1_train and ("/Train/" in f):
                continue
            t0 = time.time()
            RAW[ph][f] = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
            print(f"P{ph} {f:55s} {os.path.getsize(RAW[ph][f]) / 1000000.0:8.1f} MB  {time.time() - t0:4.0f}s")


def find(ph, *needles, optional=False):
    hits = [p for rel, p in RAW[ph].items() if all((n.lower() in rel.lower() for n in needles))]
    if optional and (not hits):
        return None
    assert len(hits) == 1, f"phase {ph} {needles}: expected 1 file, found {hits}"
    return hits[0]


P1_TRAIN = {
    s: find(1, "train", n, optional=True)
    for s, n in [("supreme_court", "supreme"), ("high_courts", "high"), ("tribunals", "tribunal"), ("acts", "acts")]
}
P1_TEST = find(1, "test")
P2 = {
    "fold1_text": find(2, "fold1-text"),
    "fold1_gold": find(2, "fold1-gold"),
    "fold2_text": find(2, "fold2-text"),
    "fold2_gold": find(2, "fold2-gold"),
    "fold3_text": find(2, "fold3-text"),
    "labels": find(2, "labels"),
}
P3 = {k: find(3, k) for k in ("CL_train", "CL_dev", "IT_train", "IT_dev", "CL_test", "IT_test", "label_vocab")}
P4 = {"train": find(4, "train.json"), "dev": find(4, "dev.json"), "test": find(4, "test.json")}
P5 = {"train": find(5, "train.json"), "dev": find(5, "dev.json"), "test": find(5, "test.json")}
P6 = {"train": find(6, "train.json"), "test": find(6, "test.json")}
P7 = {
    "train": find(7, "train.json"),
    "train_cands": find(7, "train_candidates"),
    "dev": find(7, "dev.json"),
    "dev_cands": find(7, "dev_candidates"),
    "test": find(7, "test.json"),
    "test_cands": find(7, "test_candidates"),
}
print(
    "\nPhase 1 train:",
    {k: Path(v).name if v else "skipped" for k, v in P1_TRAIN.items()},
    "| test:",
    Path(P1_TEST).name,
)
for name, d in (("Phase 2", P2), ("Phase 3", P3), ("Phase 4", P4), ("Phase 5", P5), ("Phase 6", P6), ("Phase 7", P7)):
    print(f"{name}:", {k: Path(v).name for k, v in d.items()})
from transformers import AutoTokenizer

TOK_DIR = DATA_DIR / "tokenizer"
if (TOK_DIR / "tokenizer_config.json").exists():
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
else:
    tok = AutoTokenizer.from_pretrained(os.environ.get("CS781_TOKENIZER", "openai-community/gpt2"))
    tok.add_special_tokens(
        {
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "mask_token": "<mask>",
            "additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)],
        }
    )
    tok.save_pretrained(str(TOK_DIR))
PAD_ID, BOS_ID, EOS_ID = (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id)
SENTINEL_IDS = tok.convert_tokens_to_ids([f"<extra_id_{i}>" for i in range(100)])
VOCAB_SIZE = len(tok)
assert VOCAB_SIZE < 65536
print(f"vocab {VOCAB_SIZE} | pad {PAD_ID} bos {BOS_ID} eos {EOS_ID} | sentinel_0 {SENTINEL_IDS[0]}")
print(tok.convert_ids_to_tokens(tok.encode("Section 302 of the Indian Penal Code, 1860.")))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
AMP_DTYPE = torch.bfloat16 if BF16_OK else torch.float16
USE_AMP = DEVICE.type == "cuda"
print(
    "device:",
    DEVICE,
    torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "",
    "| amp:",
    AMP_DTYPE if USE_AMP else "off",
)


class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-06):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_cos_sin(T, head_dim, device, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos()[None, None], emb.sin()[None, None])


def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):

    def __init__(self, d, h, dropout, cross=False):
        super().__init__()
        self.h, self.dk, self.cross, self.p = (h, d // h, cross, dropout)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, kv=None, mask=None):
        B, T, _ = x.shape
        src = x if kv is None else kv
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(src).view(B, src.shape[1], self.h, self.dk).transpose(1, 2)
        v = self.v(src).view(B, src.shape[1], self.h, self.dk).transpose(1, 2)
        if not self.cross:
            cos, sin = rope_cos_sin(T, self.dk, x.device)
            q, k = (apply_rope(q, cos.to(q.dtype), sin.to(q.dtype)), apply_rope(k, cos.to(k.dtype), sin.to(k.dtype)))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):

    def __init__(self, d, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


class EncoderBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.n1 = RMSNorm(c.d_model)
        self.attn = Attention(c.d_model, c.n_heads, c.dropout)
        self.n2 = RMSNorm(c.d_model)
        self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.n1(x), mask=mask))
        return x + self.drop(self.ff(self.n2(x)))


class DecoderBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.n1 = RMSNorm(c.d_model)
        self.self_attn = Attention(c.d_model, c.n_heads, c.dropout)
        self.n2 = RMSNorm(c.d_model)
        self.cross_attn = Attention(c.d_model, c.n_heads, c.dropout, cross=True)
        self.n3 = RMSNorm(c.d_model)
        self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        x = x + self.drop(self.self_attn(self.n1(x), mask=self_mask))
        x = x + self.drop(self.cross_attn(self.n2(x), kv=memory, mask=cross_mask))
        return x + self.drop(self.ff(self.n3(x)))


def key_mask(pad_mask):
    return pad_mask[:, None, None, :]


def self_mask_from(pad_mask, causal):
    B, T = pad_mask.shape
    m = key_mask(pad_mask).expand(B, 1, T, T)
    if causal:
        m = m & torch.tril(torch.ones(T, T, dtype=torch.bool, device=pad_mask.device))[None, None]
    eye = torch.eye(T, dtype=torch.bool, device=pad_mask.device)[None, None]
    return m | eye


class EncoderDecoder(nn.Module):

    def __init__(self, c, vocab_size):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(vocab_size, c.d_model, padding_idx=PAD_ID)
        self.enc = nn.ModuleList((EncoderBlock(c) for _ in range(c.n_enc_layers)))
        self.dec = nn.ModuleList((DecoderBlock(c) for _ in range(c.n_dec_layers)))
        self.enc_norm = RMSNorm(c.d_model)
        self.dec_norm = RMSNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.lm_head = nn.Linear(c.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, src, src_pad):
        x = self.drop(self.embed(src))
        m = self_mask_from(src_pad, causal=False)
        for blk in self.enc:
            x = blk(x, m)
        return self.enc_norm(x)

    def decode(self, tgt_in, memory, src_pad, tgt_pad=None):
        if tgt_pad is None:
            tgt_pad = torch.ones_like(tgt_in, dtype=torch.bool)
        x = self.drop(self.embed(tgt_in))
        sm = self_mask_from(tgt_pad, causal=True)
        cm = key_mask(src_pad)
        for blk in self.dec:
            x = blk(x, memory, sm, cm)
        return self.lm_head(self.dec_norm(x))

    def forward(self, src, src_pad, tgt_in, tgt_out, tgt_pad):
        logits = self.decode(tgt_in, self.encode(src, src_pad), src_pad, tgt_pad)
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), tgt_out.view(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(
        self,
        src,
        src_pad,
        max_new,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        stop_fn=None,
        ban_eos=False,
        min_new=0,
        no_repeat_ngram=0,
    ):
        self.eval()
        memory = self.encode(src, src_pad)
        B = src.size(0)
        out = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for step in range(max_new):
            logits = self.decode(out, memory, src_pad)[:, -1].float()
            if repetition_penalty != 1.0:
                for b in range(B):
                    prev = out[b].unique()
                    logits[b, prev] = torch.where(
                        logits[b, prev] > 0, logits[b, prev] / repetition_penalty, logits[b, prev] * repetition_penalty
                    )
            if no_repeat_ngram and out.size(1) >= no_repeat_ngram:
                n = no_repeat_ngram
                for b in range(B):
                    seq = out[b].tolist()
                    prefix = seq[-(n - 1) :]
                    banned = [seq[i + n - 1] for i in range(len(seq) - n + 1) if seq[i : i + n - 1] == prefix]
                    if banned:
                        logits[b, banned] = -float("inf")
            logits[:, PAD_ID] = -float("inf")
            logits[:, BOS_ID] = -float("inf")
            logits[:, SENTINEL_IDS] = -float("inf")
            if ban_eos or step < min_new:
                logits[:, EOS_ID] = -float("inf")
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = F.softmax(logits / temperature, -1)
                if top_p < 1.0:
                    sp, si = probs.sort(-1, descending=True)
                    cum = sp.cumsum(-1)
                    sp[cum - sp > top_p] = 0
                    sp = sp / sp.sum(-1, keepdim=True)
                    nxt = si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)
                else:
                    nxt = torch.multinomial(probs, 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt == EOS_ID
            if stop_fn is not None and step % 8 == 7:
                done |= stop_fn(out)
            if done.all():
                break
        return out[:, 1:]


def n_params(m):
    return sum((p.numel() for p in m.parameters() if p.requires_grad))


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


FT_CKPT_EVERY = 300


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r, alpha, dropout=0.05):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scale


class LoRMALinear(nn.Module):

    def __init__(self, base: nn.Linear, r, alpha, dropout=0.05):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        out = base.out_features
        self.A = nn.Parameter(torch.randn(r, out) * 0.01)
        self.B = nn.Parameter(torch.zeros(out, r))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.base(x)
        return h + self.drop(h) @ self.A.t() @ self.B.t() * self.scale


def apply_lora(encoder_blocks, r, alpha):
    for blk in encoder_blocks:
        for name in ("q", "k", "v", "o"):
            setattr(blk.attn, name, LoRALinear(getattr(blk.attn, name), r, alpha))


def apply_lorma(encoder_blocks, r, alpha):
    for blk in encoder_blocks:
        for name in ("q", "k", "v", "o"):
            setattr(blk.attn, name, LoRMALinear(getattr(blk.attn, name), r, alpha))


class EncoderTaskModel(nn.Module):

    def __init__(self, pretrained, n_labels, task):
        super().__init__()
        self.embed, self.enc, self.enc_norm = (pretrained.embed, pretrained.enc, pretrained.enc_norm)
        self.task = task
        self.head = nn.Linear(cfg.d_model, n_labels)
        self.drop = nn.Dropout(0.1)
        frozen = list(self.embed.parameters()) + list(self.enc.parameters()) + list(self.enc_norm.parameters())
        if cfg.ft_method == "head_only":
            for p in frozen:
                p.requires_grad_(False)
        elif cfg.ft_method == "lora":
            for p in frozen:
                p.requires_grad_(False)
            apply_lora(self.enc, cfg.lora_rank, cfg.lora_alpha)
        elif cfg.ft_method == "lorma":
            for p in frozen:
                p.requires_grad_(False)
            apply_lorma(self.enc, cfg.lora_rank, cfg.lora_alpha)

    def forward(self, ids, pad, pool_mask=None):
        x = self.embed(ids)
        m = self_mask_from(pad, causal=False)
        for blk in self.enc:
            x = blk(x, m)
        x = self.drop(self.enc_norm(x))
        if self.task == "token":
            return self.head(x)
        pm = (pool_mask if pool_mask is not None else pad).float()[..., None]
        return self.head((x * pm).sum(1) / pm.sum(1).clamp(min=1))


best_ck = None


def load_encoder():
    global best_ck
    best_ck = torch.load(RUN_DIR / "phase1" / "ckpt" / "best.pt", map_location=DEVICE, weights_only=False)
    print(f"loaded pretrained encoder (step {best_ck['step']}, val {best_ck['best_val']:.3f})")
    return best_ck


def fresh_pretrained():
    m = EncoderDecoder(cfg, VOCAB_SIZE)
    m.load_state_dict(best_ck["model"])
    return m


def with_method(method, build):
    prev = cfg.ft_method
    cfg.ft_method = method
    try:
        return build()
    finally:
        cfg.ft_method = prev


def finetune(
    task_model,
    train_batches,
    dev_fn,
    epochs,
    lr,
    ckpt_dir,
    tag,
    class_weight=None,
    head_lr_mult=5.0,
    warmup_batches=200,
    batches_per_epoch=None,
    eval_every=None,
    loss_fn=None,
    ckpt_every=None,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    task_model.to(DEVICE)
    ckpt_every = ckpt_every or FT_CKPT_EVERY
    head_p = [p for n, p in task_model.named_parameters() if p.requires_grad and ("head" in n.split(".") or n == "q")]
    enc_p = [
        p for n, p in task_model.named_parameters() if p.requires_grad and (not ("head" in n.split(".") or n == "q"))
    ]
    print(
        f"{tag}: trainable {sum((p.numel() for p in enc_p + head_p)) / 1000000.0:.2f}M of {sum((p.numel() for p in task_model.parameters())) / 1000000.0:.1f}M (pretrained {sum((p.numel() for p in enc_p)) / 1000000.0:.1f}M at lr {lr:.1e}, head {sum((p.numel() for p in head_p)) / 1000.0:.0f}k at lr {lr * head_lr_mult:.1e})"
    )
    groups = [
        {"params": enc_p, "lr": lr, "base": lr},
        {"params": head_p, "lr": lr * head_lr_mult, "base": lr * head_lr_mult},
    ]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)
    total = batches_per_epoch * epochs if batches_per_epoch else None
    ep, skip, best, gb = (0, 0, -float("inf"), 0)
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        task_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ep, skip, best = (ck["step"], ck["extra"].get("batch", 0), float(ck["best_val"]))
        gb = ck["extra"].get("global", 0)
        print(f"resumed at epoch {ep + 1}, batch {skip} (best {best:.4f})")

    def check(where):
        nonlocal best
        score = float(dev_fn(task_model))
        task_model.train()
        print(f"{tag} {where}: dev {score:.4f}" + ("  <- best" if score > best else ""))
        if score > best:
            best = score
            save_ckpt(ckpt_dir / "best.pt", task_model, None, None, ep + 1, best)

    while ep < epochs:
        random.seed(SEED + ep)
        task_model.train()
        losses = []
        t0 = time.time()
        n_done = 0
        for b in train_batches():
            if n_done < skip:
                n_done += 1
                continue
            scale = min(1.0, (gb + 1) / warmup_batches)
            if total:
                scale *= max(0.0, 1 - gb / total)
            for g in opt.param_groups:
                g["lr"] = g["base"] * scale
            *inputs, y = [t.to(DEVICE) for t in b]
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = task_model(*inputs)
            if loss_fn is not None:
                loss = loss_fn(logits.float(), y)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), weight=w, ignore_index=-100
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_p + head_p, 1.0)
            opt.step()
            losses.append(loss.item())
            n_done += 1
            gb += 1
            if eval_every and n_done % eval_every == 0:
                check(f"epoch {ep + 1} batch {n_done}")
            if n_done % ckpt_every == 0:
                save_ckpt(latest, task_model, opt, None, ep, best, extra={"batch": n_done, "global": gb})
                print(
                    f"    {tag} epoch {ep + 1} batch {n_done}: loss {np.mean(losses[-ckpt_every:]):.4f} lr {opt.param_groups[0]['lr']:.1e} ({time.time() - t0:.0f}s)"
                )
        skip = 0
        check(
            f"epoch {ep + 1}/{epochs} end (loss {(np.mean(losses) if losses else float('nan')):.4f}, {time.time() - t0:.0f}s)"
        )
        ep += 1
        save_ckpt(latest, task_model, opt, None, ep, best, extra={"batch": 0, "global": gb})
    task_model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=False)["model"])
    return best


def write_submission(pred, path_zip):
    path_zip.parent.mkdir(parents=True, exist_ok=True)
    pj = path_zip.parent / "prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    print("wrote", path_zip)


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


def as_text(x):
    return "\n".join(map(str, x)) if isinstance(x, list) else "" if x is None else str(x)


def extract_text(rec):
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        for k in ("text", "content", "judgment", "judgement", "body", "document", "raw_text", "full_text"):
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
