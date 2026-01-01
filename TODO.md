# TODO.md - Adaptive-P Documentation Fixes

## Priority 1: Blocking Issues (Must fix before posting)

- [ ] **Remove TODO marker in Abstract** — `<!-- TODO: Abstract length (~250 words...`
- [ ] **Remove stale TODO comment in "Target 0 WITH Min-P" sample** — Comment left behind after content was generated
- [ ] **Test or remove Python implementation** — Currently marked "has not been tested"

---

## Priority 2: Unsupported Claims (High risk of criticism)

- [ ] **Derive the 2.0 multiplier explicitly** — `calculated_target = 2.0 × configured_target − weighted_average` appears without mathematical justification
- [ ] **Demonstrate or remove cross-model consistency claim** — Claimed multiple times, all samples from one model (GLM-4.5-Air-Q4_K_M)
- [ ] **Add effective history window derivation** — Table gives "~10 tokens" for decay 0.9 without showing `1/(1-decay)`

---

## Priority 3: Methodological Gaps (Will be questioned)

- [ ] **Specify experimental methodology** — "25,000+ tokens across varied prompts" needs: which models, which prompts, how many runs, prompt diversity
- [ ] **Add samples from different model** — All samples currently from GLM-4.5-Air-Q4_K_M
- [ ] **Justify transformation function choice** — Why `dist² / (1 + dist)` vs Gaussian or other forms?
- [ ] **Justify magic constants** — PEAK=5.0, SHARPNESS=10.0, WIDTH=0.2 are "empirically tuned" without detail
- [ ] **Add ablation studies or acknowledge absence** — What if adaptive component removed? Different transformation?

---

## Priority 4: Presentation/Tone Issues (Medium risk)

- [ ] **Specify XTC implementation being criticized** — "Uniform redistribution" may not apply to all variants
- [ ] **Cite RLHF → sharper distributions claim** — Currently asserted without reference
- [ ] **Address Section 6.6 speculation** — Stability benefit is "not fully understood"; consider moving to Future Work

---

## Priority 5: Nice-to-Have Improvements (Lower risk)

- [ ] **Add standard NLG metrics** — No perplexity, MAUVE, or human evaluation
- [ ] **Expand references** — Only 6, no engagement with broader sampling literature
- [ ] **Acknowledge quick brown fox example is simplified** — Most high-confidence chains are subtler
- [ ] **Statistical significance testing** — No controlled comparisons with baselines

---

## Working Notes

### Cross-Model Consistency
Options:
1. Run samples on Llama, Mistral, etc. with same prompts/settings
2. Remove/soften all cross-model claims (appears in: Abstract, 1.3, 5.1)
3. Reframe as "observed in testing" without strong generalization

### 2.0 Multiplier Derivation
The formula comes from: wanting `E[selected_prob] = target`
If historical average is `avg`, and we want the new selection to pull toward target:
- New selection ≈ calculated_target (on average)
- New average ≈ (calculated_target + avg) / 2 (simplified)
- Want new average = target
- So: (calculated_target + avg) / 2 = target
- Therefore: calculated_target = 2*target - avg

(This is the intuition; actual derivation involves the weighted EMA)