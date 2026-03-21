# ARPABET vowels always end in 0, 1, or 2 (stress markers)
def is_vowel(phoneme: str) -> bool:
    return phoneme[-1] in ('0', '1', '2')

# Phonetic similarity groups — sounds within a group are "close"
# Each entry is (set_of_phonemes, partial_score)
SIMILARITY_GROUPS: list[tuple[frozenset[str], float]] = [
    # Voiced/unvoiced stop pairs
    (frozenset({'P', 'B'}), 0.7),
    (frozenset({'T', 'D'}), 0.7),
    (frozenset({'K', 'G'}), 0.7),

    # Voiced/unvoiced fricative pairs
    (frozenset({'F', 'V'}), 0.7),
    (frozenset({'S', 'Z'}), 0.7),
    (frozenset({'SH', 'ZH'}), 0.7),
    (frozenset({'TH', 'DH'}), 0.7),

    # Affricate/fricative near-pairs
    (frozenset({'CH', 'SH'}), 0.5),
    (frozenset({'JH', 'ZH'}), 0.5),
    (frozenset({'CH', 'JH'}), 0.6),   # voiced/unvoiced affricate pair

    # Nasal group
    (frozenset({'M', 'N', 'NG'}), 0.5),
    (frozenset({'M', 'N'}), 0.6),

    # Liquid/glide group
    (frozenset({'L', 'R'}), 0.5),
    (frozenset({'W', 'Y'}), 0.4),
    (frozenset({'L', 'R', 'W', 'Y'}), 0.3),

    # Vowel height groups (close vowels score higher)
    # High vowels
    (frozenset({'IY', 'IH'}), 0.6),
    (frozenset({'UW', 'UH'}), 0.6),
    # Mid vowels
    (frozenset({'EY', 'EH'}), 0.6),
    (frozenset({'OW', 'AO'}), 0.6),
    (frozenset({'AH', 'AE'}), 0.5),
    # Cross-height near-pairs
    (frozenset({'EH', 'AE'}), 0.5),
    (frozenset({'IH', 'EH'}), 0.5),
    # Reduced vowels
    (frozenset({'AH', 'ER'}), 0.4),
    (frozenset({'IH', 'AH'}), 0.4),
]

def phoneme_similarity(a: str, b: str) -> float:
    """
    Returns similarity score between two phonemes in [0.0, 1.0].
    Strips stress digits before comparing vowels.
    1.0 = identical, 0.0 = no relation found.
    """
    # Normalize: strip stress markers for comparison
    a_base = a.rstrip('012')
    b_base = b.rstrip('012')

    if a_base == b_base:
        return 1.0

    # Walk groups from most to least specific — return first match
    for group, score in SIMILARITY_GROUPS:
        if a_base in group and b_base in group:
            return score

    return 0.0


def rhyme_score(
    phones_a: list[str],
    phones_b: list[str],
    partial_weight: float = 1.0,   # scale down partial matches; 0.0 = exact only
    min_phoneme_score: float = 0.3, # similarities below this count as 0
) -> float:
    """
    Score the rhyme quality of two phoneme sequences in [0.0, 1.0].

    Compares backwards from the end of each word, stopping at
    (and including) the last primary-stressed vowel. Only the
    rhyme tail — from that stressed vowel to the end — is scored.

    Args:
        phones_a, phones_b: Full ARPABET phoneme lists for each word.
        partial_weight:      How much partial phoneme matches contribute.
                             1.0 = full credit, 0.5 = half credit, 0.0 = exact only.
        min_phoneme_score:   Threshold below which a partial match scores 0.
    """
    def rhyme_tail(phones: list[str]) -> list[str]:
        """Slice from last primary-stressed vowel to end."""
        for i in range(len(phones) - 1, -1, -1):
            if is_vowel(phones[i]) and phones[i][-1] == '1':
                return phones[i:]
        # Fallback: last vowel of any stress
        for i in range(len(phones) - 1, -1, -1):
            if is_vowel(phones[i]):
                return phones[i:]
        return phones  # no vowel found — score the whole thing

    tail_a = rhyme_tail(phones_a)
    tail_b = rhyme_tail(phones_b)

    # Align from the end (rhymes match at the tail)
    rev_a = tail_a[::-1]
    rev_b = tail_b[::-1]

    scored = 0.0
    possible = 0.0

    for a_ph, b_ph in zip(rev_a, rev_b):
        sim = phoneme_similarity(a_ph, b_ph)

        # Apply partial_weight to any non-perfect match
        adjusted = sim if sim == 1.0 else sim * partial_weight

        # Threshold: treat weak similarities as 0
        if adjusted < min_phoneme_score:
            adjusted = 0.0

        scored += adjusted
        possible += 1.0

    # Penalize length mismatch — unmatched phonemes in the longer tail
    length_diff = abs(len(tail_a) - len(tail_b))
    possible += length_diff  # unmatched positions score 0

    if possible == 0:
        return 0.0

    return scored / possible
