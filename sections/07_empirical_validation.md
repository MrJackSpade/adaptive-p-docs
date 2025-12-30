# 6. Empirical Validation

This section presents empirical evidence that Adaptive-P achieves its design goals: successfully targeting specified probability ranges while maintaining output quality.

## 6.1 Selection Distribution Analysis

The primary claim of Adaptive-P is that it targets specific probability ranges. We validate this by analyzing selection patterns over large generation runs.

**Methodology:**

Generate 25,000+ tokens across varied prompts with Adaptive-P at target 0.5, decay 0.9. For each token selection, record:
- Input probability (from original distribution)
- Whether this token was selected
- Post-transform probability (from Adaptive-P distribution)
- Calculated target at this step

**Expected result:**

Selection frequency should peak near the target probability. Tokens at input probability 0.5 should be selected more often than tokens at 0.2 or 0.8.

> **Graph [G-17]: Selection Scatter Plot**  
> *25k+ tokens: plot input probability (x-axis) vs. selection frequency or selection rate (y-axis). Show clear clustering around target 0.5.*

<!-- TODO: G-17 may already exist
  @claude: Image 3 (scatter) shows this pattern. May be usable directly.
  @loxifi: 
-->

**Interpreting the scatter:**

The scatter plot reveals the real distribution patterns discussed in Section 3.2:
- Dense horizontal line at p=1.0: forced choices (no alternative available)
- Dense cluster at low probabilities: tail tokens that exist but are rarely selected
- Green (selected) points tracking the grey target line when choices exist

## 6.2 Target Achievement Over Time

Beyond instantaneous selection patterns, we verify that the *average* selected probability converges to the configured target.

**Methodology:**

Track rolling average of selected token probabilities over generation. Compare to configured target.

**Expected result:**

After initial warmup (if any), the rolling average should stabilize near the configured target. Deviations should be symmetric—sometimes above target, sometimes below.

> **Graph [G-18]: Rolling Average Convergence**  
> *Plot rolling average of selected probabilities vs. token index. Overlay configured target as reference line. Show convergence.*

<!-- TODO: G-18 rolling average details
  @claude: Derive from scatter data. Document window size (e.g., 50-token).
  @loxifi: 
-->

## 6.3 Contrast with Baseline Methods

To demonstrate Adaptive-P's unique value, we compare selection patterns against baseline sampling methods on identical input distributions.

### 6.3.1 Adaptive-P vs. Temperature

**Setup:** Same generation run, comparing target 0.5 vs. temperature 1.0 (baseline) and temperature values tuned to match Adaptive-P's entropy.

**Observation:**

Temperature produces a flat selection rate curve—tokens are selected roughly in proportion to their input probability, regardless of where that probability falls. There's no preference for mid-range tokens.

Adaptive-P produces a curved selection rate that peaks near target. Mid-range tokens are selected more often than their raw probability would suggest.

> **Graph [G-19]: Adaptive-P vs. Temperature Distribution**  
> *Selection rate curves for both methods on same input data. Show that temperature is flat while Adaptive-P peaks.*



## 6.4 Adaptation Dynamics

We verify that the adaptive mechanism functions correctly—compensating for selection variance to maintain target average.

**Methodology:**

Track calculated target over generation. Observe response to selection patterns.

**Expected behavior:**
- After high-probability selection: calculated target drops (compensate by targeting lower)
- After low-probability selection: calculated target rises (compensate by targeting higher)
- Over time: calculated target oscillates around configured target

> **Graph [G-21]: Target Adaptation Response**  
> *Calculated target (line) overlaid with selected probability (scatter). Show negative correlation: high selection → target drops.*

<!-- TODO: G-21 may already exist
  @claude: Image 2 shows this pattern. Green bounces, orange stable.
  @loxifi: 
-->

## 6.5 Initialization Validation

We verify that correct initialization prevents the warmup artifacts shown with naive initialization.

**Comparison:**

| Initialization | First 50 Tokens | Steady State |
|----------------|-----------------|--------------|
| Naive (0, 0) | Calculated target starts at configured, spikes low on first high selection, slowly recovers | Normal |
| Correct (formula) | Calculated target stable from start | Normal |

> **Graph [G-22]: Initialization Comparison**  
> *Two panels: bad initialization (calculated target recovery curve) vs. correct initialization (flat from start).*

<!-- TODO: G-22 needs correct init version
  @claude: Image 0 shows bad case. Need correct init version for comparison.
  @loxifi: 
-->

## 6.6 Cross-Model Consistency

A key claim is that target 0.5 produces similar behavior across different models. We validate this with comparative analysis.

**Models tested:**
- Llama family (7B, 13B)
- Mistral (7B)  
- GLM-4 variant

**Metric:** Selection rate distribution shape at target 0.5

**Expected result:**

All models show peaked selection rate near 0.5. The peak height may vary (models with flatter distributions allow more mid-range selection), but the *location* of the peak should be consistent.

> **Graph [G-23]: Cross-Model Selection Patterns**  
> *Overlay selection rate curves from different models at same target. Show peak locations align.*

<!-- TODO: G-23 cross-model data
  @claude: Requires multi-model runs. Lower priority.
  @loxifi: 
-->
