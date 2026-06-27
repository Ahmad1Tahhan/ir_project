"""Text pre-processing — shared by documents AND queries.

Using the *same* pipeline on both sides is a requirement: queries must be
processed with the same techniques as documents so their representations are
comparable. The pipeline is configurable (stemming vs. lemmatization, etc.)
and the chosen config is persisted with the artifacts so queries at search
time match exactly how documents were indexed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# nltk is imported lazily inside ensure_nltk_data / TextPreprocessor so that
# importing this module never fails before deps are installed.

# Matches word tokens: letters/digits, keeps intra-word apostrophes.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NONASCII_RE = re.compile(r"[^\x00-\x7f]+")


def ensure_nltk_data() -> None:
    """Download the small NLTK corpora we need (idempotent, cached locally)."""
    import nltk

    from ..config import RAW_DIR

    nltk_dir = RAW_DIR / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))

    for pkg, probe in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(probe)
        except LookupError:
            nltk.download(pkg, download_dir=str(nltk_dir), quiet=True)


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    strip_urls: bool = True
    strip_nonascii: bool = True
    remove_stopwords: bool = True
    # mutually meaningful: "lemmatize" (default) | "stem" | "none"
    normalizer: str = "lemmatize"
    min_token_len: int = 2

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class TextPreprocessor:
    """Turns raw text into a list of normalized tokens (or a joined string)."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._stopwords: set[str] = set()
        self._stemmer = None
        self._lemmatizer = None
        self._init_resources()

    def _init_resources(self) -> None:
        ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        if self.config.remove_stopwords:
            self._stopwords = set(stopwords.words("english"))
        if self.config.normalizer == "stem":
            self._stemmer = PorterStemmer()
        elif self.config.normalizer == "lemmatize":
            self._lemmatizer = WordNetLemmatizer()

    # -- normalization of a single token (cached: huge speedup over 500K docs) --
    @lru_cache(maxsize=200_000)
    def _normalize_token(self, tok: str) -> str:
        if self._stemmer is not None:
            return self._stemmer.stem(tok)
        if self._lemmatizer is not None:
            return self._lemmatizer.lemmatize(tok)
        return tok

    def tokens(self, text: str) -> list[str]:
        if not text:
            return []
        if self.config.lowercase:
            text = text.lower()
        if self.config.strip_urls:
            text = _URL_RE.sub(" ", text)
        if self.config.strip_nonascii:
            text = _NONASCII_RE.sub(" ", text)

        out: list[str] = []
        mn = self.config.min_token_len
        sw = self._stopwords
        for tok in _TOKEN_RE.findall(text):
            if len(tok) < mn:
                continue
            if tok in sw:
                continue
            tok = self._normalize_token(tok)
            if len(tok) < mn or tok in sw:
                continue
            out.append(tok)
        return out

    def process(self, text: str) -> str:
        """Return a single normalized string (space-joined tokens)."""
        return " ".join(self.tokens(text))

    def process_many(self, texts, as_tokens: bool = False):
        fn = self.tokens if as_tokens else self.process
        return [fn(t) for t in texts]
