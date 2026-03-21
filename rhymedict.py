from nltk.corpus import cmudict

# ARPABET vowels always end in 0, 1, or 2 (stress markers)
def is_vowel(phoneme: str) -> bool:
    return phoneme[-1] in ('0', '1', '2')

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


def build_perfect_rhyme_index() -> dict[tuple, list[str]]:
    cmu = cmudict.dict()
    index = {}
    for word, pronunciations in cmu.items():
        for phones in pronunciations:
            key = get_rhyme_key(phones)
            if key and len(key) >= 2:  # require at least vowel + one consonant
                index.setdefault(key, []).append(word)
    return index
