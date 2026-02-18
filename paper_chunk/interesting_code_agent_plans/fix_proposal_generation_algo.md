---
name: Fix proposal generation algorithm
overview: Adjust the proposal generation algorithm to fix empty proposals caused by share scores being too sparse. The current system normalizes to top-3 candidates and uses unweighted votes, making it nearly impossible to reach the 0.80 threshold.
todos: []
---

# Fix Proposal Generation Algorithm

## Problem Analysis

The proposal generation is producing empty results because:

1. **Normalization issue**: `suggest_candidates()` always returns top 3 candidates and normalizes by their sum, diluting confidence scores to ~0.33-0.5 even for clear winners
2. **Unweighted voting**: Vote aggregation (line 760) treats all candidates equally (1 vote each), ignoring confidence scores
3. **Threshold mismatch**: `MIN_SHARE = 0.80` requires 80% consensus, which is nearly impossible with the current voting system
4. **Ambiguous filtering also broken**: The ambiguous detection (lines 794-810) uses the same `auto_suggest_distribution`, so it's also affected by the unweighted voting. With shares ~0.333, almost everything appears ambiguous (top_share < 0.75 check fails).

Current data shows shares clustering around 0.333 (1/3), well below the 0.80 threshold.

## Solution Options

### Option A: Weighted Voting + Winner-Focused Normalization (Recommended)

- Modify vote aggregation to weight votes by confidence scores instead of counting each candidate as 1 vote
- Improve normalization to encourage/amplify clear winners:
- If top score is significantly higher than second (e.g., 2x threshold), return only top candidate or give it much higher weight
- Use a normalization that amplifies winners (e.g., square scores before normalizing, or exponential weighting)
- Keep `MIN_SHARE` at 0.80 but make it achievable with weighted votes + better normalization
- More accurately reflects the heuristic's confidence and reduces dilution of clear winners

### Option B: Lower Threshold + Better Normalization

- Lower `MIN_SHARE` to ~0.50-0.60 (more realistic for 3-candidate normalization)
- Improve normalization to only return top candidate if confidence > threshold
- Simpler but less precise

### Option C: Hybrid Approach

- Use weighted voting for aggregation
- Adjust `MIN_SHARE` to 0.60-0.70 (balanced)
- Optionally filter ambiguous cases more aggressively

## Recommended Implementation (Option A)

### Changes to `paper_curation_phase1.py`:

1. **Modify vote aggregation** (lines 757-760):

- Change from counting votes to summing weighted confidence scores
- Each candidate contributes its confidence score to the selector's total
- This fixes both proposal generation AND ambiguous detection (since both use `auto_suggest_distribution`)

2. **Adjust MIN_SHARE threshold** (line 834):

- Keep at 0.80 for weighted votes, or lower to 0.60-0.70 if needed
- Test with actual data to find optimal threshold

3. **Improve normalization to encourage winners** (lines 468-485):

- **Winner detection**: If top score is 2x (or 1.5x) the second score, return only top candidate with confidence 1.0
- **Amplification**: Otherwise, use exponential weighting (e.g., square scores) before normalizing to amplify winners
- **Fallback**: If no clear winner, return top 3 but with better normalization that preserves winner advantage
- This reduces dilution: clear winners get high confidence (0.7-1.0) instead of being diluted to ~0.33-0.5

## Files to Modify

- [`paper_chunk/paper_curation_phase1.py`](paper_chunk/paper_curation_phase1.py):
- **Lines 468-485**: Improve normalization in `suggest_candidates()`:
- Add winner detection: if top_score >= 2.0 * second_score, return only top candidate with confidence 0.95
- Otherwise, square scores before normalizing to amplify winners
- Keep top 3 candidates but with amplified confidence for winners
- **Lines 757-760**: Change vote aggregation to weighted (sum confidence scores instead of counting votes)
- **Line 834**: Adjust `MIN_SHARE` threshold (test to find optimal, likely 0.60-0.80)

## interesting noste wrt full-softmax or winner detection for normalization?

Which is better: (1) winner + squaring vs (2) full softmax?

For your Phase 1 goal (auto-proposals that are safe), I’d choose:

✅ Winner detection + score squaring (with a margin threshold)

Because it increases separation and produces more high-confidence cases, which is exactly what proposals need.

Full softmax is “probabilistically nicer” but tends to stay moderate unless score gaps are huge — and your score gaps aren’t huge because your scoring is simple keyword hits + priors.

So: use winner+margin to get decisive picks, and keep top-3 list only for human review.