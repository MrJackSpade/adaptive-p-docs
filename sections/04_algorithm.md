# 3. The Adaptive-P Algorithm

This section presents the complete algorithm, starting with conceptual foundations and building to implementation details.

## 3.1 Core Concept: Probability Targeting

The fundamental question Adaptive-P answers is not "which tokens are most likely?" but rather "which tokens occupy a specific probability range?"

Consider a token distribution where:
- Token A has probability 0.85
- Token B has probability 0.10  
- Token C has probability 0.05

Standard sampling will select Token A roughly 85% of the time. But what if we want to encourage selection of tokens in the 0.3–0.5 probability range—tokens the model considers "plausible but not dominant"?

Adaptive-P addresses this by applying a bell-curve transformation centered on the target probability. Tokens close to target receive high logits; distant tokens are suppressed. After softmax normalization, the resulting distribution favors tokens near the target.

> **Graph [G-7]: Illustrative Bell Curve**  
> *Show the logit transformation curve centered at targets 0.3, 0.5, and 0.7. Label clearly as "illustrative of transformation function only" since real distributions are sparse and don't form continuous curves.*

<!-- TODO: G-7 must be labeled as illustrative
  @claude: Real distributions are sparse/clustered, won't match smooth curve.
  @loxifi: 
-->

## 3.2 Real Distribution Behavior

In practice, candidate pools after min-p filtering show characteristic patterns rather than smooth distributions. Understanding these patterns is essential to understanding why Adaptive-P's specific transformation design matters.

### Pattern A: Forced Choice

**Example from samples:** "hills" → `token 33966: 1.000000`

When only one token survives min-p filtering, the sampler has no choice. Adaptive-P passes through—the single token is selected regardless of its probability relative to target.

**Implication:** Adaptive-P cannot create choices that don't exist. It operates on the candidates provided by earlier pipeline stages.

### Pattern B: Binary Split

**Example from samples:** "green" → `token 33966: 0.944` vs `token 6176: 0.056`

Two candidates with a large probability gap. Both are viable (passed min-p), but one strongly dominates.

**Effect:** If target is 0.5, neither token is close, but the lower-probability token is closer. The transformation dramatically shifts probability toward the underdog. In this case, the 0.056 token's post-transform probability rises to 0.545—becoming the majority choice.

This is the chain-breaking mechanism in action. A high-confidence token followed by another high-confidence prediction gets disrupted because the target-adjustment favors alternatives.

### Pattern C: Clustered Tail

**Example from samples:** "prestigious" → `token 794: 0.301` + 21 tokens ranging 0.015–0.105

One mid-range leader with a cluster of low-probability alternatives. This is where the transformation's tail behavior matters most.

**The clustering problem:** If transformation applies a floor (e.g., minimum logit of 0), all 21 tail tokens hit that floor after softmax. Each gets exp(0) = 1.0 relative score. Twenty-one tokens times 1.0 = 21.0 cumulative score. The leader at 0.301 might have logit 5.0, so exp(5.0) ≈ 148. Ratio: 148 / (148 + 21) ≈ 87% for leader, 13% for entire tail.

That 13% split across 21 tokens seems okay, but imagine 100 tail tokens: now it's 148 / (148 + 100) ≈ 60% leader, 40% tail—significant garbage probability.

**Adaptive-P's solution:** Unbounded negative logits. Each additional unit of distance from target produces another unit of negative logit, which translates to another order of magnitude less probability after softmax. The cluster never accumulates mass.

### Pattern D: Competitive Mid-Range

**Example from samples:** "serene/river" → `token 21748: 0.469` vs `token 14785: 0.427`

Two tokens close in probability, both reasonably near target. This is where the quadratic core of the transformation matters.

**Effect:** The quadratic shape provides fine differentiation between close candidates. A token at distance 0.02 from target gets a noticeably different logit than one at distance 0.08, allowing the sampler to express graduated preference rather than treating all near-target tokens identically.

> **Graph [G-8]: Four Pattern Examples**  
> *Four panels showing input probabilities → output probabilities for each pattern type. Use actual data from samples.log.*

## 3.3 Configured Target vs. Calculated Target

Adaptive-P maintains a distinction between what the user requests and what the algorithm targets on each step:

- **Configured Target:** User-specified desired average probability (e.g., 0.5)
- **Calculated Target:** Per-token adjusted value based on selection history

The calculated target adapts based on a weighted moving average of previously selected token probabilities:

```
weighted_average = weighted_sum / total_weight
calculated_target = 2.0 × configured_target − weighted_average
```

**Intuition:** If recent selections averaged 0.6 probability but the user wants 0.5, the calculated target drops to 0.4 to compensate. The algorithm "aims past" the configured target to pull the average back toward it.

**From the samples:** Calculated targets range from 0.484 to 0.503 despite a fixed configured target of 0.5. This ~2% variation shows the adaptation in action—compensating for natural selection variance to maintain the desired average.

> **Graph [G-9]: Calculated Target Drift**  
> *Time-series from samples.log showing calculated target fluctuation over 20+ tokens. Overlay the configured target as a flat reference line.*

<!-- TODO: G-9 needs clean version
  @claude: Image 0 shows bad init. Need normal operation version.
  @loxifi: 
-->

The calculated target is clamped to [0.0, 1.0] before use. Extreme historical selections can push the raw calculated value outside this range, but the clamping ensures the transformation remains well-defined.

## 3.4 The Logit Transformation

The heart of Adaptive-P is a transformation applied to each token's logit based on its probability's distance from the calculated target:

```cpp
float dist = std::abs((cur_p->data[i].p - adapted_target) * INV_WIDTH);
cur_p->data[i].logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0f + dist);
```

Breaking this down:

1. **Distance calculation:** `|probability - target| × INV_WIDTH`
   - Absolute distance from target, scaled by inverse distribution width
   - INV_WIDTH (1/0.2 = 5.0) amplifies distance for sharper discrimination

2. **Transformation:** `PEAK − SHARPNESS × dist² / (1 + dist)`
   - PEAK_LOGIT_VALUE (5.0): Maximum logit for tokens exactly at target
   - SHARPNESS (10.0): Controls curve steepness
   - The `dist² / (1 + dist)` term is the key: quadratic near zero, linear at distance

**Why this specific form?**

The `dist² / (1 + dist)` function has critical properties:

- **Near target (dist → 0):** Behaves like `dist²` (quadratic). Small differences in distance produce proportionally small differences in logit. This allows fine discrimination among close competitors.

- **Far from target (dist → ∞):** Behaves like `dist` (linear). Each additional unit of distance subtracts another unit from the logit. This is the "unbounded negative" property that prevents tail accumulation.

- **Transition region:** Smooth interpolation between behaviors. No discontinuities or kinks.

> **Graph [G-10]: Transformation Function Shape**  
> *Plot the function PEAK - SHARPNESS × dist² / (1 + dist) showing the quadratic core transitioning to linear tails. Annotate the key regions.*

**Why not a floor-based transformation?**

A transformation with a minimum logit floor (e.g., one that asymptotically approaches zero) would cause all distant tokens to converge toward the same value, enabling the cluster accumulation problem described next.

## 3.5 Why Unbounded Negative Logits Matter

This section addresses the clustering problem in detail, as it's the key insight that drove the algorithm design.

**The setup:** Consider the "." token selection from samples.log—26 candidates ranging from 0.01 to 0.11 probability, none close to target 0.5.

**With a logit floor (bounded transformation):**
- All 26 tokens are far from target
- All receive approximately the minimum logit (let's say 0.0)
- After softmax: each gets exp(0) = 1.0 relative weight
- Total weight: 26.0
- Each token probability: 1/26 ≈ 3.8%

**With unbounded negative (Adaptive-P):**
- Closest token (0.11 probability) gets logit ≈ 3.0
- Next closest gets logit ≈ 2.5
- Distant tokens get logits of -5, -10, -15...
- After softmax: exp(3.0) ≈ 20, exp(2.5) ≈ 12, exp(-10) ≈ 0.00005
- Distant tokens contribute essentially zero probability

The practical difference:
- Floor-based: 26 tokens share probability approximately evenly
- Adaptive-P: Top 2-3 tokens dominate; rest are effectively excluded

This is the "selective redistribution" property. Probability doesn't flow uniformly to all candidates—it concentrates on those closest to target.

> **Graph [G-11]: Cluster Pile-up Comparison**  
> *The "." token case (26 candidates). Show post-softmax probabilities for floor-based vs. unbounded transformation.*

## 3.6 Softmax Normalization

The transformation outputs raw logit values. Softmax converts these to probabilities:

```
probability[i] = exp(logit[i]) / Σ exp(logit[j])
```

**Key property:** Relative differences between logits matter, not absolute values. If we added a constant to all logits, the probabilities would be unchanged.

This means PEAK_LOGIT_VALUE (5.0) is somewhat arbitrary—what matters is the *difference* between peak and suppressed logits.

**Interaction with tail behavior:**

Softmax's exponential nature amplifies the unbounded negative property. A logit difference of 10 produces a probability ratio of exp(10) ≈ 22,000. Very distant tokens become negligible contributors even without explicit removal.

> **Graph [G-12]: Pre vs. Post Softmax**  
> *Show raw logit values and corresponding post-softmax probabilities for a real sample. Demonstrate how the exponential amplifies differences.*

## 3.7 History State and Initialization

Adaptive-P maintains two state variables across token selections:

- **weighted_sum:** Running sum of selected token probabilities, decayed
- **total_weight:** Running sum of weight values, decayed

Updated after each selection:
```cpp
weighted_sum = original_probs[selected_idx] + decay × weighted_sum
total_weight = 1.0 + decay × total_weight
```

**The initialization problem:**

If weighted_sum and total_weight start at 0, the first calculated target becomes:
```
calculated_target = 2.0 × target − (0 / 0)  // undefined!
```

The code handles the 0/0 case by using the configured target directly. But this creates a transient: early selections have no history to compensate, so the sampler behaves differently during warmup.

**Correct initialization:**

Initialize as if the configured target had already been achieved:
```cpp
weighted_sum = target / (1.0 - decay)
total_weight = 1.0 / (1.0 - decay)
```

For target=0.5, decay=0.9:
- weighted_sum = 0.5 / 0.1 = 5.0
- total_weight = 1.0 / 0.1 = 10.0
- weighted_average = 5.0 / 10.0 = 0.5 ✓

This primes the history as if infinitely many tokens at the target probability had been selected, providing stable behavior from the first token.

> **Graph [G-13]: Initialization Effect**  
> *Compare first 50 tokens with bad initialization (target starts at 0, slowly recovers) vs. correct initialization (stable from start). The target values graph you provided shows the bad case clearly.*

<!-- TODO: G-13 needs correct initialization version
  @claude: Image 0 shows bad case (recovery over 80 tokens). Need correct init version.
  @loxifi: 
-->
