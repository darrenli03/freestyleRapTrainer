from nltk.corpus import cmudict
from syllable import syllabify
from random import shuffle

# Load the CMU Pronouncing Dictionary
d = cmudict.dict()
target_word = "theory"

def find_rhymes(word):
    word = word.lower()
    if word in d:
        pronunciations = d[word]
        rhymes = []
        for pron in pronunciations:
            ending = pron[-2:]  # Last two phonemes
            for w, p in d.items():
                if any(ending == p[-2:] for p in p):
                    rhymes.append(w)
        return set(rhymes)
    else:
        return "Word not found in cmudict."

def find_words_with_primary_stress(stress_pattern):
    """
    Find words that match a specific primary stress pattern.
    
    Args:
        stress_pattern (list): A list of stress levels to match (e.g., ['1', '0', '2']).
    
    Returns:
        list: Words that match the given stress pattern.
    """
    matches = []
    for word, pronunciations in d.items():
        for pron in pronunciations:
            # Extract the stress pattern from the pronunciation
            stresses = [phoneme[-1] for phoneme in pron if phoneme[-1].isdigit()]
            if stresses == stress_pattern:
                matches.append(word)
    return matches

phones = d[target_word][0]
print(target_word, phones)

print("syllables:", syllabify(phones))

rhymes = find_rhymes(target_word)
print("found", len(rhymes), "rhymes")

if len(rhymes) > 20:
    print(list(rhymes)[-20:])
else:
    print(rhymes)

# Example: Find words with primary stress on the first syllable followed by unstressed syllables
pattern = ['1', '0']
matching_words = find_words_with_primary_stress(pattern)

print(f"Words matching the stress pattern {pattern}:")
print(matching_words[:20])  # Print the first 20 matches
