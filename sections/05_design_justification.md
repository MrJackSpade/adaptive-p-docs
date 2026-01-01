# 4. Design Rationale

This section explains why Adaptive-P uses its specific formulas and constants. Understanding these choices helps users predict behavior in edge cases and enables implementers to make informed modifications.

## 4.1 The Adaptive Target Formula

The core adaptive mechanism uses:

```
calculated_target = 2.0 × configured_target − weighted_average
```

**Why the 2.0 multiplier?**

The goal is for the *average* selected probability to converge to the configured target over time. If recent selections have averaged above target, we need to aim below target to compensate—and vice versa.

Consider the simplest case: if the next selection lands exactly at `calculated_target`, and we want the new average to equal `configured_target`:

```
(calculated_target + weighted_average) / 2 = configured_target
```

Solving for `calculated_target`:

```
calculated_target = 2 × configured_target − weighted_average
```

The actual implementation uses an exponentially-weighted moving average rather than a simple average, but the same principle applies. The 2.0 multiplier creates a control loop where overshooting in one direction produces proportional correction in the other.

**Boundary behavior:**

The calculated target is clamped to [0.0, 1.0] after computation. Without clamping, extreme historical averages could push the calculated target outside the valid probability range. For example, if `configured_target = 0.3` and `weighted_average = 0.8`, the raw calculated target would be `2(0.3) - 0.8 = -0.2`, which is clamped to 0.0.

## 4.2 The Transformation Function

The transformation converts probability-to-target distance into logits:

```
logit = PEAK − SHARPNESS × dist² / (1 + dist)
```

This specific form was selected after evaluating alternatives that failed in characteristic ways.

### Alternative 1: Lorentzian (Power Law)

```
logit = peak / (1 + dist²)
```

**Failure mode: Artificial floor**

The Lorentzian has a minimum value approaching zero but never reaching it. With typical constants, the minimum achievable logit is approximately 0.4, regardless of how far a token lies from target.

After softmax over 100 candidates, this floor means each garbage token retains ~1% probability. Collectively, distant tokens accumulate significant selection probability despite being semantically inappropriate. This is the same "fat tail" problem that affects XTC.

### Alternative 2: Gaussian

```
logit = peak × exp(−dist²)
```

**Failure mode: Cliff effect**

The Gaussian suppresses distant tokens effectively—too effectively. It creates a "cliff" where all tokens beyond a certain distance receive near-identical (negligible) logits.

When the target is 0.5 but only tokens at 0.1 and 0.2 probability are available, both are far from target. The Gaussian assigns them nearly equal selection probability despite 0.2 being objectively closer. The sampler loses the ability to express graduated preference among off-target candidates.

### Final Design: Adaptive Unbounded Quadratic

```
logit = PEAK − SHARPNESS × dist² / (1 + dist)
```

This function interpolates between two regimes:

**Near target (dist → 0):** The denominator approaches 1, yielding `PEAK − SHARPNESS × dist²`. Quadratic behavior preserves fine-grained differentiation among tokens close to target, similar to Gaussian.

**Far from target (dist → ∞):** The expression simplifies to approximately `PEAK − SHARPNESS × dist`. Linear decay ensures:
- Unbounded negative logits (no artificial floor)
- Proper suppression after softmax (each additional unit of distance costs another order of magnitude in probability)
- Maintained relative ordering (a token at distance 2 remains more probable than one at distance 3, even when both are far from target)

**Smooth transition:** The `(1 + dist)` denominator provides continuous interpolation without discontinuities or kinks.

## 4.3 Constant Selection

The three internal constants were empirically tuned across multiple models. This section documents the rationale and known limitations.

### PEAK_LOGIT_VALUE = 5.0

The peak value determines the maximum logit for tokens exactly at target. After softmax, this establishes probability ratios between on-target and off-target tokens.

At `PEAK = 5.0`, a token at target receives `exp(5.0) ≈ 148×` the probability weight of a token at logit 0. This provides strong differentiation without numerical instability.

- Lower values (e.g., 3.0) reduce differentiation, making the sampler less effective at targeting
- Higher values (e.g., 10.0) risk over-concentration on single tokens and potential overflow issues

### SHARPNESS = 10.0

Sharpness controls how quickly logits decay as tokens deviate from target. Higher values produce more aggressive suppression of off-target tokens.

The value 10.0 was selected to balance two failure modes:
- **Too low (e.g., 4.0):** Insufficient suppression of garbage tokens when min-p filtering is relaxed
- **Too high (e.g., 20.0):** Over-sensitivity to small probability differences, reducing effective candidate diversity

Sharpness interacts with the decay parameter: higher sharpness amplifies the effect of target adjustments, making low-decay configurations more volatile.

These values were tuned visually against sample candidate pool distributions—calibrated to produce reasonable drift among top candidates while preventing probability pile-up in the tail. They perform well across tested models but could likely be improved through systematic optimization.

### DISTRIBUTION_WIDTH = 0.2 (INV_WIDTH = 5.0)

Width normalizes the distance metric, defining what "near" and "far" mean in probability space.

With `WIDTH = 0.2`:
- A token 0.2 probability units from target has `dist = 1.0` (one "standard width")
- A token 0.4 probability units from target has `dist = 2.0`

This scaling produces intuitive relationships between target values and selection behavior. A target of 0.5 creates a preference band roughly spanning 0.3–0.7, with tokens outside this range progressively suppressed.

### Limitations of Constant Tuning

These constants were determined through iterative testing rather than systematic optimization:

- No grid search or formal hyperparameter tuning was conducted
- Testing concentrated on specific model families (GLM-4.x, Mistral, Cydonia)
- Interaction effects between constants were not formally characterized

The values work well across tested configurations but may benefit from adjustment for specific model architectures or use cases. The constants are defined as preprocessor macros to facilitate experimentation.

## 4.4 History Initialization

The weighted moving average requires careful initialization to avoid transient artifacts.

**The problem with naive initialization:**

Starting with `weighted_sum = 0` and `total_weight = 0` creates a degenerate first step. The code handles the 0/0 case by using configured target directly, but subsequent steps exhibit a characteristic "crash and recovery" pattern:

1. First selection (say, probability 0.8) enters an empty history
2. Weighted average immediately becomes 0.8
3. Calculated target swings to `2(0.5) - 0.8 = 0.2`
4. System spends many tokens recovering to equilibrium

**The correct initialization:**

Initialize as if the target had already been achieved at equilibrium:

```
weighted_sum = target / (1 − decay)
total_weight = 1 / (1 − decay)
```

**Derivation:** At equilibrium, the weighted average equals the target. For an exponentially-weighted sum with decay factor `d`, the sum of weights is the geometric series `1 + d + d² + ... = 1/(1-d)`. If every term in the weighted sum equals `target`, then `weighted_sum = target × 1/(1-d) = target/(1-d)`.

This initialization makes the first token behave identically to the hundredth token—no warmup period, no transient artifacts.

## 4.5 What Was Not Formally Tested

Transparency about evaluation limitations:

**No standard NLG metrics:** Evaluations used qualitative assessment ("does this read well?") rather than perplexity, MAUVE scores, or human preference ratings.

**No factorial design:** Interactions between target × decay × min-p × model architecture were not systematically characterized.

**Limited model coverage:** Limited model coverage: Most testing and all documentation samples used GLM-4.x. Qualitative testing on Mistral and Cydonia showed similar target-hitting behavior, but systematic cross-architecture comparison was not conducted. The "cross-model consistency" claim is based on observed selection patterns, not rigorous benchmarking.

**No ablation of adaptive component:** Direct quantitative comparison between static targeting (decay = 0) and adaptive targeting was not formally documented, though qualitative testing indicated static targeting produces oscillatory "fishtailing" behavior.

Community testing provided extensive qualitative feedback that informed the design, but this paper does not claim rigorous empirical validation by academic standards.