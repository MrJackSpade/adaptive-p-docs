# 7. Empirical Validation

This section presents empirical evidence that Adaptive-P achieves its design goals: successfully targeting specified probability ranges while maintaining output quality.

## 7.0 A Note on Evaluation Metrics

**Adaptive-P is designed for creative writing, not factual accuracy or predictability.** This scope constrains which evaluation metrics are meaningful.

### Why Perplexity Is Not Reported

Perplexity measures how well a model predicts the tokens in a reference text—lower perplexity indicates the model found the text more predictable. For a sampler explicitly designed to select *less* predictable tokens, perplexity degradation is the intended behavior, not a flaw to be measured.

Consider: at target 0.3, Adaptive-P preferentially selects tokens the model assigns ~30% probability. By definition, these selections will have higher surprisal (and thus higher perplexity) than greedy or near-greedy decoding. Reporting this number would be equivalent to criticizing a random number generator for not producing sequential integers.

The same logic applies to other likelihood-based metrics (BLEU against deterministic references, likelihood under the base model, etc.). These metrics assume the goal is reproducing predictable text. Adaptive-P assumes the opposite.

### What Would Constitute Valid Evaluation

Meaningful evaluation of creative sampling methods requires:

- **Human preference studies:** Does output read as more engaging, varied, or creative?
- **Diversity metrics:** Does the sampler produce more lexically or semantically varied output across multiple generations?
- **Task-specific assessment:** For roleplay, does the output avoid repetitive phrasing? For fiction, does it produce unexpected but coherent plot developments?

This paper does not include formal human evaluation studies. The empirical validation focuses on verifying that Adaptive-P achieves its mechanical goal (targeting specified probability ranges) rather than proving that this goal produces subjectively better creative writing. The latter claim—that probability targeting improves creative output—is supported by community testing and qualitative assessment but not by controlled studies.

We invite researchers with human evaluation infrastructure to conduct such studies.

## 7.1 Selection Distribution Analysis

The primary claim of Adaptive-P is that it targets specific probability ranges. We validate this by analyzing selection patterns over large generation runs.

**Methodology:**

Generate 25,000+ tokens across varied prompts with Adaptive-P at target 0.5, decay 0.9. For each token selection, record:
- Input probability (from original distribution)
- Whether this token was selected
- Post-transform probability (from Adaptive-P distribution)
- Calculated target at this step

Selection frequency should peak near the target probability. Tokens at input probability 0.5 should be selected more often than tokens at 0.2 or 0.8.

The following scatter plots show selection behavior at different target values. Orange dots are available candidates; green dots are selected tokens; grey line is the calculated (clamped) target.

<table>
<tr>
<td width="50%"><img src="../charts/target_0.3.png" width="100%"><br><em>Target 0.3: Selections cluster around 0.3–0.4, avoiding high-probability tokens.</em></td>
<td width="50%"><img src="../charts/target_0.5.png" width="100%"><br><em>Target 0.5: Selections favor mid-range (0.5–0.7).</em></td>
</tr>
<tr>
<td width="50%"><img src="../charts/target_0.7.png" width="100%"><br><em>Target 0.7: Selections shift toward 0.7–1.0.</em></td>
<td width="50%"><img src="../charts/target_1.0.png" width="100%"><br><em>Target 1.0: Nearly greedy—almost all at p=1.0.</em></td>
</tr>
</table>

**Interpreting the scatter:**

The scatter plots reveal the real distribution patterns discussed in Section 3.2:
- Dense horizontal line at p=1.0: forced choices (no alternative available)
- Dense cluster at low probabilities: tail tokens that exist but are rarely selected
- Green (selected) points tracking the grey target line when choices exist

## 7.2 Target Achievement Over Time

Beyond instantaneous selection patterns, we verify that the *average* selected probability converges to the configured target.

**Methodology:**

Track rolling average of selected token probabilities over generation. Compare to configured target.

After initial warmup (if any), the rolling average should stabilize near the configured target. Deviations should be symmetric—sometimes above target, sometimes below.

The following charts show calculated target over time at decay 0.5, 0.9, and 0.99 (all at target 0.5):

<table>
<tr>
<td width="50%"><img src="../charts/target_d0.5.png" width="100%"><br><em>Decay 0.5: Large oscillations (fishtailing).</em></td>
<td width="50%"><img src="../charts/target_d0.9.png" width="100%"><br><em>Decay 0.9: Balanced responsiveness.</em></td>
</tr>
<tr>
<td colspan="2" align="center" width="50%"><img src="../charts/target_d0.99.png" width="50%"><br><em>Decay 0.99: Very tight oscillations—nearly flat.</em></td>
</tr>
</table>

## 7.3 Contrast with Baseline Methods

To demonstrate Adaptive-P's unique value, we compare selection patterns against baseline sampling methods on identical input distributions.

### 7.3.1 Adaptive-P vs. Temperature

**Setup:** Same generation run, comparing target 0.5 vs. temperature 1.0 (baseline) and temperature values tuned to match Adaptive-P's entropy.

**Observation:**

Temperature produces a flat selection rate curve—tokens are selected roughly in proportion to their input probability, regardless of where that probability falls. There's no preference for mid-range tokens.

Adaptive-P produces a curved selection rate that peaks near target. Mid-range tokens are selected more often than their raw probability would suggest.

The temperature vs. Adaptive-P comparison charts in Section 2.1 demonstrate this contrast visually: temperature uniformly scales while Adaptive-P creates a distinct peak.

## 7.4 Adaptation Dynamics

We verify that the adaptive mechanism functions correctly—compensating for selection variance to maintain target average.

**Methodology:**

Track calculated target over generation. Observe response to selection patterns.

**Expected behavior:**
- After high-probability selection: calculated target drops (compensate by targeting lower)
- After low-probability selection: calculated target rises (compensate by targeting higher)
- Over time: calculated target oscillates around configured target

The decay comparison charts in Section 6.2 show this adaptation in action—decay 0.5 produces large oscillations (fishtailing), while decay 0.99 shows nearly flat targeting.

## 7.5 Initialization Validation

We verify that correct initialization prevents the warmup artifacts shown with naive initialization.

| Initialization | First 50 Tokens | Steady State |
|----------------|-----------------|--------------|
| Naive (0, 0) | Calculated target starts at configured, spikes low on first high selection, slowly recovers | Normal |
| Correct (formula) | Calculated target stable from start | Normal |

<table>
<tr>
<td width="50%"><strong>Bad initialization (naive):</strong><br><img src="../charts/bad_init.png" width="100%"><br><em>Naive init: Target drops to 0, takes ~100 tokens to recover.</em></td>
<td width="50%"><strong>Correct initialization:</strong><br><img src="../charts/target_d0.9.png" width="100%"><br><em>Correct init: Target stable from first token.</em></td>
</tr>
</table>