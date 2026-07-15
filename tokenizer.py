"""BPE tokenizer trained on the provided train_corpus.txt ONLY.

Design notes (why this beats the byte baseline):
  * The corpus is ~14% Devanagari CHARACTERS, but each is 3 UTF-8 bytes, so a
    vocab-256 byte tokenizer spends ~3 tokens per Hindi char. BPE merges those
    3-byte sequences into single tokens.
  * Score is bits per BYTE = bits_per_token / bytes_per_token. Raising
    bytes_per_token (compression) directly divides the loss.
  * LOSSLESS BY CONSTRUCTION: every merge is built up from raw bytes, and any
    byte that never got merged still has its own id 0..255. Arbitrary UTF-8
    (even unseen scripts / invalid sequences) always encodes via the byte
    fallback. decode(encode(t)) == t exactly.

Interface contract kept: load() takes no required args and returns an object
with .encode(str)->list[int], .decode(list[int])->str, .vocab_size.
Merge table is resolved relative to __file__ so grading cwd does not matter.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "bpe.json")


def _split_words(text):
    """Same boundary rule used at BPE training time. Lossless: ''.join == text."""
    words, cur = [], []
    for ch in text:
        if ch == " ":
            if cur:
                words.append("".join(cur)); cur = []
            cur.append(ch)
        elif ch in "\n\t\r":
            if cur:
                words.append("".join(cur)); cur = []
            words.append(ch)
        else:
            cur.append(ch)
    if cur:
        words.append("".join(cur))
    return words


class BPETokenizer:
    def __init__(self, merges):
        pairs = [(tuple(p), n) for p, n in merges]
        self.ranks = {p: i for i, (p, _) in enumerate(pairs)}
        self.merge_id = {p: n for p, n in pairs}
        self.vocab_size = 256 + len(pairs)
        self.id2bytes = {i: bytes([i]) for i in range(256)}
        for (a, b), nid in pairs:
            self.id2bytes[nid] = self.id2bytes[a] + self.id2bytes[b]
        self._cache = {}

    def _encode_word(self, w):
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ids = list(w.encode("utf-8"))          # byte fallback floor
        while len(ids) >= 2:
            best, best_rank = None, None
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            a, b = ids[best], ids[best + 1]
            ids[best:best + 2] = [self.merge_id[(a, b)]]
        if len(w) < 64:
            self._cache[w] = ids
        return ids

    def encode(self, text):
        out = []
        for w in _split_words(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids):
        b = b"".join(self.id2bytes[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "vocab_size": self.vocab_size}, f)


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


def load(path=None):
    """Return the tokenizer. No args required (train.py / evaluate.py call
    load() bare). Falls back to raw bytes if no merge table is present."""
    p = path or _DEFAULT
    if not os.path.exists(p):
        return ByteTokenizer()
    with open(p) as f:
        d = json.load(f)
    if d.get("type") == "byte":
        return ByteTokenizer()
    return BPETokenizer(d["merges"])
