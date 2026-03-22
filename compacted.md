## Project Summary: Freestyle Rap Trainer

### Project Location
`/home/eric/Projects/freestyleRapTrainer/`

### Core Modules

| File | Purpose |
|------|---------|
| `rhymedict.py` | Rhyme word lookup with phoneme-based matching |
| `rhymescore.py` | Phonetic similarity scoring between phonemes |
| `main.py` | Tkinter UI for the rap trainer application |
| `test_rhymedict.py` | Unit tests for rhyme functionality |
| `test_beat.py` | Unit tests for beat playback |
| `requirements.txt` | Python dependencies |

---

### Key Features Implemented

#### 1. Pattern-Based Rhyme Search
- Phoneme-based lookup instead of canonical word scoring
- Generates similar rhyme tails by varying vowels and consonants
- Uses CMU dictionary for pronunciation data

#### 2. Random Diverse Rhymes with Exclusion
- `random_diverse_rhymes()` - returns random rhymes with frequency weighting
- `random_diverse_rhymes_by_phonemes()` - phoneme-based version
- Exclusion support to prevent repeated suggestions
- Shuffle top-N candidates before returning

#### 3. Multi-Syllable Rhyme Fallback
- `_find_rhyme_source()` in `main.py` - tries 1→2→3 syllables until ≥3 rhymes found
- `get_line_rhyme_tail()` - extracts rhyme tail from last N syllables of a line
- Handles lines like "locking in" → rhymes on "-ocking in" phonemes

#### 4. Phoneme-Based Core Architecture
All rhyme functions now use phoneme-based lookup internally:
- `find_rhymes_by_phonemes()` - core phoneme search
- `diverse_rhymes_by_phonemes()` - frequency-weighted phoneme search  
- `random_diverse_rhymes_by_phonemes()` - random + exclude phoneme search
- Word-based functions are thin wrappers

#### 5. Cache Pre-Warming
- `prewarm_caches()` - loads rhyme index and Brown corpus
- Background thread in `_prewarm_caches()` with status updates
- LRU caching on all rhyme lookup functions

#### 6. Beat Playback
- Load MP3/WAV files via file dialog
- Play/Pause controls
- Speed slider (0.5x to 1.5x) using samplerate manipulation
- Continuous loop playback
- Uses `soundfile` for reading and `sounddevice` for playback

#### 7. UI Enhancements
- Quit button added
- Start Recording button disabled until caches warm
- Status line shows loading progress
- Beat Controls section in UI

---

### Current API

```python
# Word-based (wrappers)
find_rhymes(word, min_score=0.5, limit=None)
diverse_rhymes(word, n=5, freq_weight=0.3)
random_diverse_rhymes(word, exclude=set(), n=5, freq_weight=0.3, pool_size=50)

# Phoneme-based (core)
find_rhymes_by_phonemes(phones, min_score=0.5, limit=None)
diverse_rhymes_by_phonemes(phones, n=5, freq_weight=0.3, exclude_word=None)
random_diverse_rhymes_by_phonemes(phones, exclude=set(), n=5, freq_weight=0.3, pool_size=50, exclude_word=None)

# Helpers
get_word_phonemes(word) -> list[str] | None
get_line_rhyme_tail(line_words, n) -> tuple[str, ...] | None
prewarm_caches(progress_callback=None)
```

---

### Test Status
- **74 tests passing** (64 rhyme tests + 8 beat tests + 2 prewarm tests)
- Pre-existing LSP type warnings about `None` vs `list` in `get_rhyme_tail` calls (non-blocking)

---

### Recent Bug Fix
**Bug:** Input words appearing as rhyme suggestions (e.g., "mike" showing when querying rhymes for "mike")

**Root Causes Found:**
1. `main.py` had `self.exclude_words.update(last_word)` which added characters instead of the word
2. Phoneme-based exclusion was too aggressive - "world" and "whirled" share same phonemes

**Fixes Applied:**
- Changed `update()` to `add()` for proper word exclusion
- Added `exclude_word` parameter to phoneme-based functions (excludes by word name, not phonemes)

---

### Files Modified (Recent Session)

**rhymedict.py:**
- Added `exclude_word` parameter to `diverse_rhymes_by_phonemes` and `random_diverse_rhymes_by_phonemes`
- Excludes input word by name rather than by phoneme matching

**main.py:**
- Fixed `exclude_words.update()` → `exclude_words.add()`
- Added `exclude_word=last_word` to `random_diverse_rhymes_by_phonemes` call

**test_rhymedict.py:**
- Added tests for `exclude_word` parameter

---

### Dependencies (requirements.txt)
```
numpy, soundfile, sounddevice, pytest-mock  (recent additions)
nltk, vosk, pygame-free alternatives       (existing)
```

---

### Known Issues / Future Work

1. **Loop gaps:** Beat playback may have small gaps between loops (crossfading planned)

2. **Test warnings:** RuntimeError in threading during pytest (cache warming attempts to update UI from background thread) - cosmetic only, doesn't affect functionality

3. **Type hints:** LSP warnings on `get_rhyme_tail` receiving `list | None` (pre-existing, tests pass)

4. **User exclusion bug:** Fixed - input words should no longer appear as suggestions
