import pytest

from rhymedict import (
    find_rhymes,
    get_rhyme_tail,
    generate_pattern_variations,
    RhymeIndex,
    get_index,
    random_diverse_rhymes,
    get_word_phonemes,
    get_line_rhyme_tail,
    find_rhymes_by_phonemes,
    random_diverse_rhymes_by_phonemes,
)


class TestGetRhymeTail:
    def test_two_phoneme_tail(self):
        from nltk.corpus import cmudict

        cmu = cmudict.dict()
        assert get_rhyme_tail(cmu["cat"][0]) == ("AE1", "T")
        assert get_rhyme_tail(cmu["time"][0]) == ("AY1", "M")
        assert get_rhyme_tail(cmu["back"][0]) == ("AE1", "K")

    def test_three_phoneme_tail(self):
        from nltk.corpus import cmudict

        cmu = cmudict.dict()
        assert get_rhyme_tail(cmu["world"][0]) == ("ER1", "L", "D")

    def test_stressed_vowel_extraction(self):
        from nltk.corpus import cmudict

        cmu = cmudict.dict()
        tail = get_rhyme_tail(cmu["attack"][0])
        assert "AE1" in tail


class TestGeneratePatternVariations:
    def test_exact_match_included(self):
        tail = ("AE1", "T")
        variations = generate_pattern_variations(tail, min_score=0.5)
        variation_tails = [v[0] for v in variations]
        assert tail in variation_tails

    def test_similar_consonants_included(self):
        tail = ("AE1", "T")
        variations = generate_pattern_variations(tail, min_score=0.5)
        variation_tails = [v[0] for v in variations]
        assert ("AE1", "D") in variation_tails

    def test_length_preserved(self):
        tail = ("AE1", "T")
        variations = generate_pattern_variations(tail, min_score=0.5)
        for var, _ in variations:
            assert len(var) == len(tail)


class TestFindRhymes:
    def test_returns_list_of_tuples(self):
        result = find_rhymes("cat")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_word_not_in_dictionary(self):
        result = find_rhymes("xyzzyznotaword")
        assert result == []

    def test_excludes_input_word(self):
        result = find_rhymes("cat", limit=100)
        words = [w for w, _ in result]
        assert "cat" not in words

    def test_sorted_by_score_descending(self):
        result = find_rhymes("cat", limit=100)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_limit_parameter(self):
        result = find_rhymes("cat", limit=10)
        assert len(result) <= 10

    def test_min_score_filtering(self):
        result_low = find_rhymes("cat", min_score=0.5, limit=200)
        result_high = find_rhymes("cat", min_score=0.8, limit=200)
        assert all(score >= 0.5 for _, score in result_low)
        assert all(score >= 0.8 for _, score in result_high)
        assert len(result_high) <= len(result_low)

    def test_perfect_rhymes_for_cat(self):
        result = find_rhymes("cat", min_score=1.0)
        words = [w for w, _ in result]
        expected = {"bat", "back", "hat", "flat", "chat", "attack"}
        assert any(w in words for w in expected)

    def test_partial_rhymes_for_cat(self):
        result = find_rhymes("cat", min_score=0.5, limit=200)
        partial = [(w, s) for w, s in result if s < 1.0]
        assert len(partial) > 0
        assert all(0.5 <= s < 1.0 for _, s in partial)

    def test_world_perfect_rhymes(self):
        result = find_rhymes("world", min_score=1.0)
        words = [w for w, _ in result]
        expected = {"curled", "whirled", "twirled", "swirled"}
        for word in expected:
            assert word in words, f"{word} should be a perfect rhyme for world"

    def test_sing_perfect_rhymes(self):
        result = find_rhymes("sing", min_score=1.0)
        words = [w for w, _ in result]
        expected = {"ring", "thing", "bring", "king", "wing"}
        for word in expected:
            assert word in words, f"{word} should be a perfect rhyme for sing"

    def test_time_partial_rhymes(self):
        result = find_rhymes("time", min_score=0.5, limit=100)
        partial = [(w, s) for w, s in result if s < 1.0]
        assert len(partial) > 0, "Should have partial rhymes for time"
        assert any(w in ["align", "assign", "affine"] for w, _ in partial)

    def test_score_range(self):
        result = find_rhymes("cat", min_score=0.5, limit=100)
        for word, score in result:
            assert 0.0 <= score <= 1.0, f"Score for {word} out of range: {score}"


class TestRhymeIndex:
    def test_singleton(self):
        idx1 = get_index()
        idx2 = get_index()
        assert idx1 is idx2

    def test_find_rhymes_returns_same_results(self):
        from rhymedict import find_rhymes as module_find

        idx = get_index()
        result1 = idx.find_rhymes("cat", min_score=0.5, limit=20)
        result2 = module_find("cat", min_score=0.5, limit=20)
        assert result1 == result2


class TestLRUCache:
    def test_cached_results_are_identical(self):
        result1 = find_rhymes("cat", min_score=0.5, limit=10)
        result2 = find_rhymes("cat", min_score=0.5, limit=10)
        assert result1 == result2


class TestDeprecationWarning:
    def test_build_perfect_rhyme_index_warns(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from rhymedict import build_perfect_rhyme_index

            build_perfect_rhyme_index()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestDiverseRhymes:
    def test_returns_list_of_tuples(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("cat", n=5)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_excludes_input_word(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("cat", n=20)
        words = [w for w, _, _ in result]
        assert "cat" not in words

    def test_limit_parameter(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("cat", n=5)
        assert len(result) <= 5

    def test_common_words_appear(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("cat", n=5)
        words = [w for w, _, _ in result]
        assert "at" in words or "that" in words, "Common words should appear"

    def test_rhyme_scores_included(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("cat", n=5)
        for word, rhyme_score, freq in result:
            assert 0.0 <= rhyme_score <= 1.0
            assert freq >= 0

    def test_freq_weight_affects_results(self):
        from rhymedict import diverse_rhymes

        result_low = diverse_rhymes("cat", n=5, freq_weight=0.0)
        result_high = diverse_rhymes("cat", n=5, freq_weight=1.0)
        words_low = [w for w, _, _ in result_low]
        words_high = [w for w, _, _ in result_high]
        assert words_low != words_high, "freq_weight should affect results"

    def test_word_not_in_dictionary(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("xyzzyznotaword")
        assert result == []

    def test_min_score_filtering(self):
        from rhymedict import diverse_rhymes

        result = diverse_rhymes("cat", n=50, min_score=0.8)
        for word, rhyme_score, freq in result:
            assert rhyme_score >= 0.8


class TestPerformance:
    def test_query_speed(self):
        import time

        words = ["cat", "time", "world", "love", "flow", "money", "street", "freestyle"]

        start = time.time()
        for _ in range(100):
            for word in words:
                find_rhymes(word, min_score=0.5, limit=20)
        elapsed = time.time() - start

        queries_per_sec = (100 * len(words)) / elapsed
        assert queries_per_sec > 100, f"Too slow: {queries_per_sec:.0f} queries/sec"

    def test_cached_speed(self):
        import time

        find_rhymes("cat", min_score=0.5, limit=20)

        start = time.time()
        for _ in range(1000):
            find_rhymes("cat", min_score=0.5, limit=20)
        elapsed = time.time() - start

        queries_per_sec = 1000 / elapsed
        assert queries_per_sec > 100000, (
            f"Cache too slow: {queries_per_sec:.0f} queries/sec"
        )


class TestRandomDiverseRhymes:
    def test_returns_list_of_tuples(self):
        result = random_diverse_rhymes("cat", n=5)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_excludes_input_word(self):
        result = random_diverse_rhymes("cat", n=20)
        words = [w for w, _, _ in result]
        assert "cat" not in words

    def test_excludes_specified_words(self):
        exclude = {"hat", "bat", "flat"}
        result = random_diverse_rhymes("cat", exclude=exclude, n=5)
        words = [w for w, _, _ in result]
        assert "hat" not in words
        assert "bat" not in words
        assert "flat" not in words

    def test_limit_parameter(self):
        result = random_diverse_rhymes("cat", n=5)
        assert len(result) <= 5

    def test_returns_different_results_across_calls(self):
        words1 = random_diverse_rhymes("cat", n=10)
        words2 = random_diverse_rhymes("cat", n=10)
        assert words1 != words2, "Results should vary across calls due to shuffling"

    def test_handles_exclude_removing_all_candidates(self):
        exclude = {"hat", "bat", "flat", "chat", "back", "that", "at", "what", "attack"}
        result = random_diverse_rhymes("cat", exclude=exclude, n=5)
        words = [w for w, _, _ in result]
        assert "cat" not in words
        for w in exclude:
            assert w not in words

    def test_rhyme_scores_included(self):
        result = random_diverse_rhymes("cat", n=5)
        for word, rhyme_score_val, freq in result:
            assert 0.0 <= rhyme_score_val <= 1.0
            assert freq >= 0

    def test_word_not_in_dictionary(self):
        result = random_diverse_rhymes("xyzzyznotaword")
        assert result == []

    def test_min_score_filtering(self):
        result = random_diverse_rhymes("cat", n=50, min_score=0.8)
        for word, rhyme_score_val, freq in result:
            assert rhyme_score_val >= 0.8


class TestGetWordPhonemes:
    def test_returns_phoneme_list(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        assert isinstance(phones, list)
        assert len(phones) > 0

    def test_word_not_found_returns_none(self):
        phones = get_word_phonemes("xyzzyznotaword")
        assert phones is None

    def test_phonemes_match_rhyme_tail(self):
        phones = get_word_phonemes("cat")
        tail = get_rhyme_tail(phones)
        assert tail == ("AE1", "T")


class TestGetLineRhymeTail:
    def test_single_word(self):
        phones = get_word_phonemes("cat")
        assert isinstance(phones, list)
        tail = get_line_rhyme_tail(["cat"], n=1)
        assert tail == get_rhyme_tail(phones)  # type: ignore[arg-type]

    def test_multiple_words(self):
        tail = get_line_rhyme_tail(["locking", "in"], n=2)
        assert tail is not None
        assert isinstance(tail, tuple)
        assert len(tail) >= 2

    def test_insufficient_syllables_returns_none(self):
        tail = get_line_rhyme_tail(["in"], n=3)
        assert tail is None


class TestFindRhymesByPhonemes:
    def test_returns_list_of_tuples(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        result = find_rhymes_by_phonemes(tuple(phones))
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_sorted_by_score_descending(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        result = find_rhymes_by_phonemes(tuple(phones), limit=100)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_includes_common_rhymes(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        result = find_rhymes_by_phonemes(tuple(phones), min_score=0.8)
        words = [w for w, _ in result]
        assert any(w in ["hat", "bat", "flat", "that"] for w in words)


class TestRandomDiverseRhymesByPhonemes:
    def test_returns_list_of_tuples(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        result = random_diverse_rhymes_by_phonemes(tuple(phones), n=5)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_excludes_specified_words(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        exclude = {"hat", "bat", "flat"}
        result = random_diverse_rhymes_by_phonemes(tuple(phones), exclude=exclude, n=5)
        words = [w for w, _, _ in result]
        assert "hat" not in words
        assert "bat" not in words
        assert "flat" not in words

    def test_returns_different_results_across_calls(self):
        phones = get_word_phonemes("cat")
        assert phones is not None
        words1 = random_diverse_rhymes_by_phonemes(tuple(phones), n=10)
        words2 = random_diverse_rhymes_by_phonemes(tuple(phones), n=10)
        assert words1 != words2


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not installed. Run: pip install pytest")
