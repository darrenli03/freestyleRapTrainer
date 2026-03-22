from functools import lru_cache
import nltk
from nltk.corpus import cmudict
from rhymescore import rhyme_score, is_vowel, SIMILARITY_GROUPS


def _ensure_brown_corpus():
    """Download brown corpus if not available."""
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)


@lru_cache(maxsize=1)
def get_frequency_dist():
    """
    Get cached frequency distribution from Brown corpus.
    Downloads corpus on first call if needed.
    """
    _ensure_brown_corpus()
    from nltk import FreqDist
    from nltk.corpus import brown

    words = brown.words()
    fd = FreqDist(w.lower() for w in words)
    return fd


def get_rhyme_tail(phones: list[str]) -> tuple[str, ...]:
    """Extract the rhyme tail: from the last stressed vowel to end."""
    for i in range(len(phones) - 1, -1, -1):
        if is_vowel(phones[i]) and phones[i][-1] == "1":
            return tuple(phones[i:])
    for i in range(len(phones) - 1, -1, -1):
        if is_vowel(phones[i]):
            return tuple(phones[i:])
    return tuple(phones)


def get_similar_phonemes(
    phoneme: str, min_score: float = 0.5
) -> list[tuple[str, float]]:
    """Return phonemes similar to the given phoneme with their similarity scores."""
    base = phoneme.rstrip("012")
    similar = []
    for group, score in SIMILARITY_GROUPS:
        if base in group and score >= min_score:
            for p in group:
                if p != base:
                    similar.append((p, score))
    return similar


def generate_pattern_variations(
    tail: tuple[str, ...], min_score: float = 0.5
) -> list[tuple[tuple[str, ...], float]]:
    """Generate similar rhyme patterns by varying vowels and consonants."""
    if not tail:
        return [(tail, 1.0)]

    results = []

    vowel_base = tail[0]
    vowel_variations = [(vowel_base, 1.0)]
    for sim, score in get_similar_phonemes(vowel_base, min_score):
        vowel_variations.append((sim + vowel_base[-1], score))

    consonant_positions = list(range(1, len(tail)))
    cons_variations_list = []
    for pos in consonant_positions:
        pos_vars = [(tail[pos], 1.0)]
        for sim, score in get_similar_phonemes(tail[pos], min_score):
            pos_vars.append((sim, score))
        cons_variations_list.append(pos_vars)

    for vowel, v_score in vowel_variations:

        def combine(pos_idx: int, current: tuple) -> list[tuple]:
            if pos_idx >= len(cons_variations_list):
                return [tuple([vowel] + list(current))]
            combos = []
            for cons, _ in cons_variations_list[pos_idx]:
                combos.extend(combine(pos_idx + 1, current + (cons,)))
            return combos

        new_tails = combine(0, ())
        for new_tail in new_tails:
            scores = [v_score]
            for i in range(1, len(tail)):
                from rhymescore import phoneme_similarity

                scores.append(phoneme_similarity(tail[i], new_tail[i]))
            pattern_score = sum(scores) / len(scores)
            results.append((new_tail, pattern_score))

    return results


class RhymeIndex:
    """Rhyme index with pattern-based search."""

    def __init__(self):
        self._tail_index: dict[tuple[str, ...], list[str]] = {}
        self._word_phones: dict[str, list[str]] = {}
        self._build()

    def _build(self):
        """Build the tail index from CMU dictionary."""
        cmu = cmudict.dict()
        for word, pronunciations in cmu.items():
            tail = get_rhyme_tail(pronunciations[0])
            self._tail_index.setdefault(tail, []).append(word)
            if word not in self._word_phones:
                self._word_phones[word] = pronunciations[0]

    def find_rhymes(
        self,
        word: str,
        min_score: float = 0.5,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Find words that rhyme with the given word.

        Args:
            word: The word to find rhymes for.
            min_score: Minimum rhyme score threshold (0.0 to 1.0).
            limit: Maximum number of results to return.

        Returns:
            List of (word, score) tuples sorted by score descending.
            Excludes the input word from results.
        """
        if word not in self._word_phones:
            return []

        original_phones = self._word_phones[word]
        original_tail = get_rhyme_tail(original_phones)
        variations = generate_pattern_variations(original_tail, min_score)

        results = []
        for tail, _ in variations:
            if tail in self._tail_index:
                for match_word in self._tail_index[tail]:
                    if match_word != word:
                        score = rhyme_score(
                            original_phones, self._word_phones[match_word]
                        )
                        if score >= min_score:
                            results.append((match_word, score))

        seen = set()
        unique_results = []
        for w, score in sorted(results, key=lambda x: -x[1]):
            if w not in seen:
                seen.add(w)
                unique_results.append((w, score))

        if limit is not None:
            return unique_results[:limit]
        return unique_results


_index: RhymeIndex | None = None


def get_index() -> RhymeIndex:
    """Get the singleton rhyme index, building it on first call."""
    global _index
    if _index is None:
        _index = RhymeIndex()
    return _index


@lru_cache(maxsize=1024)
def find_rhymes(
    word: str,
    min_score: float = 0.5,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """
    Find words that rhyme with the given word.

    Args:
        word: The word to find rhymes for.
        min_score: Minimum rhyme score threshold (0.0 to 1.0).
        limit: Maximum number of results to return.

    Returns:
        List of (word, score) tuples sorted by score descending.
        Excludes the input word from results.
    """
    return get_index().find_rhymes(word, min_score, limit)


def build_perfect_rhyme_index() -> dict[tuple, list[str]]:
    """
    DEPRECATED: This function is no longer supported.

    Use find_rhymes() for pattern-based rhyme matching with scores.
    """
    import warnings

    warnings.warn(
        "build_perfect_rhyme_index() is deprecated. "
        "Use find_rhymes() for scored partial rhymes.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {}


def diverse_rhymes(
    word: str,
    n: int = 5,
    min_score: float = 0.5,
    freq_weight: float = 0.3,
) -> list[tuple[str, float, int]]:
    """
    Find diverse rhyming words biased towards common usage.

    Combines rhyme score with word frequency from the Brown corpus
    to return words that both rhyme well and are commonly used.

    Args:
        word: The word to find rhymes for.
        n: Maximum number of rhymes to return.
        min_score: Minimum rhyme score threshold (0.0 to 1.0).
        freq_weight: Blend factor for frequency (0.0=rhymes only, 1.0=freq only).
                    Default 0.3 biases towards common words while preserving rhyme quality.

    Returns:
        List of (word, rhyme_score, frequency) tuples sorted by combined score.
        Excludes the input word from results.
    """
    candidates = find_rhymes(word, min_score=min_score, limit=None)
    if not candidates:
        return []

    freq_dist = get_frequency_dist()
    max_freq = max(freq_dist.values()) if freq_dist else 1

    scored_candidates = []
    for rhyme_word, rhyme_score_val in candidates:
        freq = freq_dist.get(rhyme_word, 0)
        freq_normalized = (freq + 1) / (max_freq + 1)
        freq_log = freq / (max_freq + 1)

        combined = (1 - freq_weight) * rhyme_score_val + freq_weight * freq_log
        scored_candidates.append((rhyme_word, rhyme_score_val, freq, combined))

    scored_candidates.sort(key=lambda x: -x[3])

    return [(w, s, f) for w, s, f, _ in scored_candidates[:n]]
