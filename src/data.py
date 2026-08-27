"""
data.py — SlimPajama streamed, DOCUMENT-LEVEL segmentation with exact resume.

Segmentation (per Issue 3): tokenize ONE document -> split into complete
MAX_LENGTH sequences -> every segment keeps that document's source label -> DROP
the short remainder -> next document. No cross-document buffering, so shuffled
multi-source order does NOT distort domain frequencies. The routing unit is one
document -> one hidden source domain, cleanly.

Per-source audit: sequences produced and tokens dropped, by source, so domain
imbalance is visible/auditable and supplies the empirical prior for the MI
random control.

Resume: consumed_sequences is a single integer; same seed + .skip() replays the
exact stream. Only completed (trained) sequences are ever saved (see train.py).
"""
import torch
import config as C

SLIMPAJAMA_DOMAINS = [
    "RedPajamaCommonCrawl", "RedPajamaC4", "RedPajamaGithub",
    "RedPajamaBook", "RedPajamaArXiv", "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(C.TOKENIZER_NAME)
    tok.pad_token = tok.eos_token
    return tok


def _raw_stream(split="train"):
    """Yields {"text": ...} dicts. LOCAL mode reads a WikiText .txt file line by
    line (no HF Hub); HF mode streams. Controlled by C.DATA_SOURCE.

    WikiText articles are delimited by heading lines like " = Title = ". We treat
    each article as one document so document-level segmentation is meaningful.
    """
    import os, random
    if getattr(C, "DATA_SOURCE", "local") == "local":
        path = C.TRAIN_FILE if split == "train" else C.VALID_FILE
        fmt = getattr(C, "LOCAL_FORMAT", "wikitext").lower()
        if fmt in ("jsonl", "json", "slimpajama"):
            return _local_jsonl_docs(path)
        return _local_wikitext_docs(path)
    # HF streaming. Use the requested split if the dataset provides it; else fall
    # back to the train split (the eval/train disjointness is then handled by
    # EVAL_SKIP_DOCS in build_eval_set / batch_iterator).
    from datasets import load_dataset
    cfg = getattr(C, "DATASET_CONFIG", None)
    hf_split = "validation" if split == "valid" else C.DATASET_SPLIT
    try:
        if cfg:
            ds = load_dataset(C.DATASET_NAME, cfg, split=hf_split, streaming=True)
        else:
            ds = load_dataset(C.DATASET_NAME, split=hf_split, streaming=True)
    except (ValueError, Exception):
        # dataset has no such split (e.g. train-only) -> use train; disjointness
        # via EVAL_SKIP_DOCS handled by the caller.
        if cfg:
            ds = load_dataset(C.DATASET_NAME, cfg, split=C.DATASET_SPLIT, streaming=True)
        else:
            ds = load_dataset(C.DATASET_NAME, split=C.DATASET_SPLIT, streaming=True)
    return ds.shuffle(seed=C.SHUFFLE_SEED, buffer_size=C.SHUFFLE_BUFFER)


def _local_wikitext_docs(path):
    """Generator of {"text": article} from a WikiText .txt file.

    Robust to formatting: matches a top-level heading ' = Title = ' with an
    OPTIONAL leading space and exactly one '=' each side (subsection headings
    use '==' / '===' and are kept inside the article). A hard cap on document
    size (MAX_DOC_CHARS) guarantees no single tokenize call ever receives a
    multi-MB blob, even if heading detection fails on an odd file."""
    import re
    # optional leading space, single '=', a title not starting with '=', single '='
    heading = re.compile(r'^\s?= [^=].*[^=] = \s*$')
    max_chars = getattr(C, "MAX_DOC_CHARS", 200_000)
    buf, buf_len = [], 0

    def _flush(b):
        t = "".join(b).strip()
        return {"text": t} if t else None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            is_head = bool(heading.match(line))
            # start a new doc on a heading, OR force-flush if the buffer grew too
            # large (guards against a file with no detectable headings)
            if (is_head and buf) or buf_len >= max_chars:
                d = _flush(buf)
                if d:
                    yield d
                buf, buf_len = [], 0
            buf.append(line); buf_len += len(line)
        d = _flush(buf)
        if d:
            yield d


def _local_jsonl_docs(path):
    """Generator of full example dicts from a local SlimPajama-style JSONL file
    (one JSON object per line). CRUCIAL vs the WikiText reader: this PRESERVES the
    example's `meta` so _source_of recovers the true redpajama_set_name — otherwise
    every local doc collapses to one source label and the multi-domain routing
    diagnostics are meaningless.

    Accepts either a single .jsonl file or a directory of *.jsonl / *.json shards
    (read in sorted order for reproducibility). Text field is `text` (SlimPajama);
    meta field is `meta` (kept verbatim). Lines that don't parse are skipped."""
    import os, json, glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl"))
                       + glob.glob(os.path.join(path, "*.json")))
        if not files:
            raise FileNotFoundError(
                f"LOCAL_FORMAT=jsonl but no *.jsonl/*.json shards found in {path}")
    else:
        files = [path]
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                txt = ex.get("text", "")
                if not txt or not txt.strip():
                    continue
                # yield the whole example so _source_of sees meta.redpajama_set_name
                yield ex

def _source_of(ex):
    # SlimPajama tags source in meta.redpajama_set_name. Single-domain datasets
    # (e.g. WikiText) have no meta -> one label. MI is only meaningful on multi-
    # domain data; the dense baseline reports N/A regardless.
    meta = ex.get("meta") or {}
    if isinstance(meta, dict) and (meta.get("redpajama_set_name") or meta.get("source")):
        return meta.get("redpajama_set_name") or meta.get("source")
    return getattr(C, "SINGLE_DOMAIN_LABEL", "wikitext")


class DocSegmentStream:
    """Document-level segmentation. Yields (input_ids[MAX_LENGTH], source).
    Tracks consumed_sequences and a per-source audit (kept / dropped tokens)."""
    def __init__(self, tokenizer, skip_sequences=0, split="train"):
        self.tok = tokenizer
        self.consumed = 0
        self._skip = skip_sequences
        self._split = split
        self.audit_kept = {}      # source -> sequences produced
        self.audit_dropped = {}   # source -> tokens dropped (short remainders)

    def _bump(self, d, k, n=1):
        d[k] = d.get(k, 0) + n

    def __iter__(self):
        L = C.MAX_LENGTH
        produced = 0
        for ex in _raw_stream(self._split):
            src = _source_of(ex)
            ids = self.tok(ex.get("text", ""), add_special_tokens=False)["input_ids"]
            n_full = len(ids) // L
            # drop remainder (audit it)
            remainder = len(ids) - n_full * L
            if remainder:
                self._bump(self.audit_dropped, src, remainder)
            for s in range(n_full):
                seg = ids[s * L:(s + 1) * L]
                produced += 1
                if produced <= self._skip:
                    continue
                self.consumed = produced
                self._bump(self.audit_kept, src)
                yield torch.tensor(seg, dtype=torch.long), src


def batch_iterator(tokenizer, skip_sequences=0):
    """Yields (input_ids[MB,L], sources[MB], consumed_sequences, stream).
    Training starts AFTER the held-out eval region (the first EVAL_SAMPLES
    sequences of the same shuffled stream) so eval and train never overlap.
    consumed_sequences remains relative to the training region for clean resume."""
    # Held-out eval occupies the first EVAL_SAMPLES sequences of the shuffled
    # stream; training must start after them. consumed_sequences is ABSOLUTE
    # (counts from stream start), so on resume skip_sequences already exceeds the
    # guard and max() is idempotent — never double-skips.
    eval_guard = C.EVAL_SAMPLES if getattr(C, "DATA_SOURCE", "local") == "hf" else 0
    start = max(skip_sequences, eval_guard) if skip_sequences == 0 else skip_sequences
    start = max(start, eval_guard)
    # [FIX #10] no balanced-eval reservation to skip past. Project B has no
    # hand-assigned domains, so training starts right after the eval guard — the
    # first ~250k sequences are no longer thrown away.
    stream = DocSegmentStream(tokenizer, skip_sequences=start)
    batch, srcs = [], []
    for seg, src in stream:
        batch.append(seg); srcs.append(src)
        if len(batch) == C.MICRO_BATCH:
            yield torch.stack(batch), srcs, stream.consumed, stream
            batch, srcs = [], []


def build_eval_set(tokenizer):
    """Fixed held-out eval set with source labels retained (MI ground truth).
    Local files are FINITE, so stop at EVAL_SAMPLES OR end-of-file, whichever
    first — never loop. If the valid file yields fewer than EVAL_SAMPLES
    full-length sequences, use what it has."""
    stream = DocSegmentStream(tokenizer, skip_sequences=0, split="valid")
    seqs, labels = [], []
    for seg, src in stream:
        seqs.append(seg); labels.append(src)
        if len(seqs) >= C.EVAL_SAMPLES:
            break
    if len(seqs) == 0:
        raise RuntimeError(
            "Eval set empty — no full-length sequences from the valid file. "
            "Lower MAX_LENGTH or check VALID_FILE.")
    if len(seqs) < C.EVAL_SAMPLES:
        print(f"[eval] valid file produced only {len(seqs)} sequences "
              f"(< EVAL_SAMPLES={C.EVAL_SAMPLES}); using all of them.")
    audit = {"kept": dict(stream.audit_kept), "dropped": dict(stream.audit_dropped)}
    return torch.stack(seqs), labels, audit


def domain_prior(labels):
    """Empirical domain prior from eval labels — the MI random-control baseline."""
    from collections import Counter
    c = Counter(labels)
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


def build_source_labeled_diag(tokenizer):
    """[FIX #10] SOURCE-LABELED selector diagnostic set — no hand-assigned paths.

    Collects a per-source-balanced set (equal exposure per SOURCE domain, so no
    source dominates the heatmap), streamed from a modest skip past the eval guard.

    IMPORTANT — NOT held out. [review #2] Because Project B no longer reserves a
    disjoint region, training eventually streams through this same range, so these
    sequences can later become training examples. Treat the resulting MI / heatmap
    / oracle-share numbers as DESCRIPTIVE TRAINING-DISTRIBUTION diagnostics (is the
    selector self-organizing?), NOT as a held-out generalization measurement. If a
    paper-quality held-out number is wanted later, carve a genuinely disjoint region
    and skip training past it (the old reservation mechanism, but WITHOUT any
    domain->path target).

    Returns NO 'target path' — Project B specialists have no correct domain.
    Downstream (evaluate_selector) reports source -> predicted-path heatmap, mutual
    information, and oracle-path shares by source, but never scores agreement with a
    fixed domain->path map.

    Returns (ids[N,L], sources[N]). The old signature's third element (target paths)
    is gone; callers must drop it.
    """
    import torch
    per_source = max(1, C.SELECTOR_DIAG_SAMPLES // len(SLIMPAJAMA_DOMAINS))
    need = {s: per_source for s in SLIMPAJAMA_DOMAINS}
    got = {s: 0 for s in need}
    seqs, sources = [], []
    stream = DocSegmentStream(tokenizer, skip_sequences=C.SELECTOR_DIAG_SKIP)
    scanned, scan_cap = 0, max(C.SELECTOR_DIAG_SAMPLES * 500, 100_000)
    for seg, src in stream:
        scanned += 1
        if src in need and got[src] < need[src]:
            seqs.append(seg); sources.append(src); got[src] += 1
            if all(got[k] >= need[k] for k in need):
                break
        if scanned >= scan_cap:
            break        # single-domain data (e.g. wikitext) won't fill every source
    if not seqs:
        raise RuntimeError("source-labeled diagnostic set empty — check DATA_SOURCE.")
    # partial is acceptable here (single-domain corpora only have one source); the
    # diagnostic just reports over whatever sources appeared.
    filled = {k: v for k, v in got.items() if v > 0}
    ids = torch.stack(seqs)
    print(f"[selector-diag] collected per source: {filled} (scanned {scanned}) "
          f"— training-distribution diagnostic, NOT held-out")
    return ids, sources


# Back-compat shim: old name returned a third (target_paths) element built from the
# now-removed DOMAIN_TO_PATH map. Keep the name working but WITHOUT domain targets —
# callers that still unpack three values should migrate to build_source_labeled_diag.
def build_balanced_selector_eval(tokenizer):
    ids, sources = build_source_labeled_diag(tokenizer)
    return ids, sources
