# Legal English onsets (subset — extend as needed)
LEGAL_ONSETS = {
    ('P',), ('B',), ('T',), ('D',), ('K',), ('G',),
    ('F',), ('V',), ('TH',), ('DH',), ('S',), ('Z',),
    ('SH',), ('ZH',), ('HH',), ('CH',), ('JH',),
    ('M',), ('N',), ('L',), ('R',), ('W',), ('Y',),
    ('P', 'L'), ('B', 'L'), ('K', 'L'), ('G', 'L'), ('F', 'L'),
    ('S', 'L'), ('P', 'R'), ('B', 'R'), ('T', 'R'), ('D', 'R'),
    ('K', 'R'), ('G', 'R'), ('F', 'R'), ('TH', 'R'),
    ('S', 'P'), ('S', 'T'), ('S', 'K'), ('S', 'W'), ('S', 'N'),
    ('S', 'M'), ('S', 'P', 'L'), ('S', 'P', 'R'), ('S', 'T', 'R'),
    ('S', 'K', 'R'), ('S', 'K', 'W'),
}

# ARPABET vowels always end in 0, 1, or 2 (stress markers)
def is_vowel(phoneme: str) -> bool:
    return phoneme[-1] in ('0', '1', '2')

def split_cluster(cluster: list[str]) -> tuple[list[str], list[str]]:
    """
    Given consonants between two vowels, find the best coda/onset split
    using the Maximum Onset Principle: maximize the legal onset.
    """
    # Try to take as many consonants as possible into the onset
    for split in range(len(cluster) + 1):
        coda = cluster[:split]
        onset = cluster[split:]
        if not onset or tuple(onset) in LEGAL_ONSETS:
            return coda, onset
    # Fallback: everything goes to coda
    return cluster, []

def syllabify(phones: list[str]) -> list[list[str]]:
    vowel_indices = [i for i, p in enumerate(phones) if is_vowel(p)]

    if not vowel_indices:
        return [phones]

    syllables = []
    # Leading consonants before first vowel = onset of syllable 1
    carry_onset = phones[:vowel_indices[0]]

    for s, v_idx in enumerate(vowel_indices):
        is_last = (s == len(vowel_indices) - 1)

        if is_last:
            # Everything after the vowel is the coda
            coda = phones[v_idx + 1:]
            next_onset = []
        else:
            next_v_idx = vowel_indices[s + 1]
            cluster = phones[v_idx + 1 : next_v_idx]
            coda, next_onset = split_cluster(cluster)

        syllables.append(carry_onset + [phones[v_idx]] + coda)
        carry_onset = next_onset

    return syllables

def get_rhyme_key(phones: list[str]) -> list[str] | None:
    """
    Returns the substring from the last *stressed* vowel to the end.
    Falls back to the last vowel if no stressed vowel exists.
    """
    last_stressed = None
    last_vowel = None

    for i, p in enumerate(phones):
        if is_vowel(p):
            last_vowel = i
            if p[-1] == '1':  # primary stress
                last_stressed = i

    nucleus = last_stressed if last_stressed is not None else last_vowel
    if nucleus is None:
        return None  # no vowel — skip

    return phones[nucleus:]  # stressed vowel through end of word
