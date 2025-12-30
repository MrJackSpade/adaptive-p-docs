# Adaptive-P: Probability-Targeting Sampling for Controllable Text Generation

## Paper Outline (Revised v2)

---

## 1. Abstract
Brief overview: Adaptive-P targets a specific probability range rather than truncating or uniformly scaling. This enables controllable creativity while maintaining coherence, breaking the high-confidence chains that produce slop and repetition.

---

## 2. Introduction

### 2.1 The Problem: High-Confidence Token Chains
LLMs produce repetitive, generic output because certain sequences form self-reinforcing patterns—once committed to a common phrase, each subsequent token has an obvious high-probability continuation.

### 2.2 Why Existing Approaches Fall Short
Current samplers either truncate (removing options) or uniformly scale (affecting all tokens equally). Neither allows *targeting* a specific probability band.

### 2.3 The Adaptive-P Solution
Direct probability targeting with adaptive adjustment over time.

> **Graph [G-1]: Overview Comparison**
> Real token distributions (from samples.log) showing input vs. output for Temperature, Min-P, XTC, and Adaptive-P.

---

## 3. Related Work and Comparative Analysis

### 3.1 Temperature Sampling
- Uniform logit scaling
- **Limitation:** Affects all tokens equally; cannot prefer mid-range over extremes
- No awareness of where tokens sit in the confidence landscape

> **Graph [G-2]: Temperature Effect on Real Distribution**
> Take a real multi-token sample (e.g., 18 tokens from "serene" case) and show pre/post temperature at 0.5, 1.0, 1.5.

### 3.2 Top-K and Top-P (Nucleus Sampling)
- Truncation-based: keep top tokens until threshold reached
- **Limitation:** Binary include/exclude; no preference within the kept set
- Answers "which tokens are most likely?" not "which tokens occupy a target range?"

### 3.3 Min-P
- Removes tokens below a probability threshold relative to the top token
- **Strength:** Effective garbage removal
- **Limitation:** Does not encourage creativity or mid-range selection
- **Relationship to Adaptive-P:** Serves as a complementary guardrail. The samples.log shows tokens already filtered to p>0.01; min-p performs this cleanup before Adaptive-P operates.

> **Graph [G-3]: Min-P as Guardrail**
> Before/after min-p on a full vocabulary distribution, then handoff to Adaptive-P.

### 3.4 XTC (eXclude Top Choices)
- Randomly removes high-probability tokens to force alternatives
- **Critical Flaw:** Standard top truncation evenly redistributes probability across ALL remaining tokens—including garbage tokens. This causes fat tail accumulation and increases incoherent generation.
- Heavy RNG dependence leads to inconsistent results

> **Graph [G-4]: XTC Redistribution Failure**
> Demonstrate uniform redistribution spreading probability to low-quality tail tokens. Compare to Adaptive-P selective redistribution.

### 3.5 Mirostat
- Original inspiration for Adaptive-P
- Targets perplexity via dynamic Top-K adjustment
- **Why it fails with modern models:** Newer models have sharper probability distributions. Even Top-K of 100,000 often selects the same 2-3 tokens. The samples show this reality—most selections have 2-5 viable candidates after min-p.

### 3.6 Why Adaptive-P is Different
Selective redistribution: logits are transformed as a function of distance from the target probability.

Key insight from the samples: real distributions rarely resemble a smooth curve. They typically show:
- A single 100% token (no choice, no effect)
- One high token + sparse low cluster
- 2-3 mid-range candidates + low-end cluster with gap

The transformation must handle these sparse, clustered, gapped distributions—not just continuous curves.

> **Graph [G-5]: Selective Redistribution on Real Distribution**
> Use "prestigious" sample (22 tokens: one at 0.30, rest at 0.02-0.10). Show how exponential dropoff prevents clustered low tokens from accumulating probability.

---

## 4. The Adaptive-P Algorithm

### 4.1 Core Concept: Probability Targeting
Instead of "which tokens are most likely?", Adaptive-P asks "which tokens occupy this specific probability range?"

> **Graph [G-6]: Illustrative Bell Curve (with caveat)**
> The logit transformation curve centered at various targets. **Label clearly as "illustrative of transformation function only"** since real distributions are sparse.

### 4.2 Real Distribution Behavior
In practice, candidate pools after min-p show:

**Pattern A: Forced choice** — Single token at 100%
- Example: "hills" → `token 33966: 1.000000`
- Effect: Sampler has no choice; passthrough

**Pattern B: Binary split** — Two candidates with large gap
- Example: "green" → `0.944` vs `0.056`
- Effect: Transformation dramatically shifts probability toward target-closer token

**Pattern C: Clustered tail** — One mid-high token + cluster of low tokens
- Example: "prestigious" → `0.30` + 21 tokens at 0.02-0.10
- Effect: Exponential dropoff suppresses entire cluster; prevents pile-up

**Pattern D: Competitive mid-range** — 2-3 tokens near target
- Example: "serene/river" → `0.47` vs `0.43`
- Effect: Fine differentiation between close candidates

> **Graph [G-7]: Four Pattern Examples**
> Four panels showing input/output probabilities for each pattern type, using real samples.

### 4.3 The Configured Target vs. Calculated Target
- **Configured Target:** User-specified desired average (e.g., 0.5)
- **Calculated Target:** Per-token adjusted value based on selection history

Formula:
```
adapted_target = 2.0 × target − (weighted_sum / total_weight)
```

Sample evidence: Calculated targets in samples.log range from 0.484 to 0.503 despite fixed configured target of 0.5, showing adaptation in action.

> **Graph [G-8]: Calculated Target Drift**
> Time-series from samples.log showing calculated target fluctuation over 20 tokens.

### 4.4 The Logit Transformation
```cpp
float dist = std::abs((cur_p->data[i].p - adapted_target) * INV_WIDTH);
cur_p->data[i].logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0f + dist);
```

Key properties:
- **Quadratic near target:** Fine differentiation for close candidates
- **Linear in tails:** Gentler falloff prevents over-suppression of moderately-distant tokens
- **Unbounded negative:** Essential for handling clustered distributions—prevents cluster pile-up where many distant tokens accumulate probability

### 4.5 Why Unbounded Negative Logits Matter (The Clustering Problem)
With logit floors (like Gaussian's asymptote to 0), clustered low-probability tokens accumulate:

Example from samples: After "." there are 26 tokens ranging 0.01-0.11.
- With a floor: Each gets exp(floor), cumulative mass dominates
- With unbounded negative: Each additional distance unit = another order of magnitude suppression

> **Graph [G-9]: Cluster Pile-up Comparison**
> Same 26-token input. Show post-softmax probabilities for: floor-based (Gaussian) vs. unbounded (Adaptive-P).
> The samples.log data shows this directly—compare GAUSSIAN vs ADAPTIVE-P columns for "." token selection.

### 4.6 Softmax Normalization
The final probability distribution emerges after softmax. Relative differences between logits matter, not absolute values.

> **Graph [G-10]: Pre vs. Post Softmax**
> Raw logit values vs. post-softmax probabilities for a real sample, showing normalization effect.

### 4.7 History State and Initial Values
Weighted moving average requires proper initialization.

Initialization:
```cpp
weighted_sum = target / (1.0f - decay)
total_weight = 1.0f / (1.0f - decay)
```

This primes the system as if the target had already been achieved.

> **Graph [G-11]: Initialization Effect**
> First 10 tokens with bad initialization (calculated target spikes low) vs. correct initialization (stable).

---

## 5. Parameters

### 5.1 Target
- **Range:** 0.0 to 1.0 (negative = sampler disabled)
- **Effect:** Higher = more conservative, Lower = more creative
- **Recommended:** 0.4–0.6 for most use cases
- **Key property:** Works consistently across different models—the same target produces similar behavior regardless of architecture

> **Graph [G-12]: Target Effect on Selection Distribution**
> Aggregate selection histograms at target 0.3, 0.5, 0.7, 0.9 from large generation runs.

### 5.2 Decay
- **Range:** 0.0 to 0.99
- **Default:** 0.9
- **Effect:** Controls adaptation speed. Lower decay = more responsive to recent selections.

> **Graph [G-13]: Decay Effect**
> Normalized selection frequency at decay 0.5, 0.7, 0.9.

### 5.3 Internal Constants (Fixed)
| Constant | Value | Purpose |
|----------|-------|---------|
| PEAK_LOGIT_VALUE | 5.0 | Maximum logit for on-target tokens |
| SHARPNESS | 10.0 | Controls curve steepness |
| DISTRIBUTION_WIDTH | 0.2 | Probability distance scaling |

---

## 6. Integration and Sampler Chain

### 6.1 Chain Positioning
Adaptive-P **must be last** in the sampler chain.

Recommended minimal chain:
```
min_p → adaptive_p
```

### 6.2 Why Min-P Complements Adaptive-P
Min-p removes true garbage tokens. Adaptive-P then operates on quality candidates.

The samples.log shows tokens already filtered to p>0.01—that's min-p at work.

> **Graph [G-14]: Combined Pipeline**
> Full vocabulary → min-p filter → Adaptive-P transform → final selection.

### 6.3 What Becomes Unnecessary
- **DRY/Repetition Penalty:** Adaptive-P breaks repetition by design
- **XTC:** Redundant goals; Adaptive-P is more consistent

---

## 7. Empirical Validation

### 7.1 Selection Distribution Analysis
Large-scale aggregation proving target achievement.

> **Graph [G-15]: Aggregate Selection Scatter**
> Input probability vs. selection frequency over 25k+ tokens, showing clustering around target.

### 7.2 Adaptive-P vs. Gaussian (Side-by-Side)
The samples.log includes both Adaptive-P and Gaussian outputs for direct comparison.

Key observations:
- On sparse distributions (2-3 tokens): Similar behavior
- On clustered distributions (20+ low tokens): Gaussian preserves more tail probability

> **Graph [G-16]: Adaptive-P vs. Gaussian on Clustered Input**
> The "." token case (26 candidates). Show how Gaussian's tail handling differs.

### 7.3 Per-Token Comparison Examples
Walk through specific real samples showing transformation logic.

> **Graph [G-17]: Annotated Token Walkthrough**
> 3-4 selected real examples from samples.log with input→output annotations explaining the transformation.

---

## 8. Model Considerations

### 8.1 Models with Diverse Distributions
Models trained on varied data benefit most. Examples: Llama, Mistral, GLM.

### 8.2 Models with Sharp Distributions
Heavily instruction-tuned models (Gemma, Phi) may lack mid-range tokens. Consider higher targets (0.6+).

---

## 9. Implementation Reference

### 9.1 Core Algorithm
```cpp
static void llama_sampler_adaptive_p_apply(
    struct llama_sampler * smpl, 
    llama_token_data_array * cur_p) 
{
    auto * ctx = (llama_sampler_adaptive_p *) smpl->ctx;

    if (ctx->target < 0.0f) {
        llama_sampler_softmax_impl(cur_p, false);
        cur_p->selected = llama_sample_dist(cur_p, ctx->rng);
        return;
    }

    llama_sampler_softmax_impl(cur_p, false);
    ctx->original_probs.resize(cur_p->size);
    for (size_t i = 0; i < cur_p->size; ++i) {
        ctx->original_probs[i] = cur_p->data[i].p;
    }

    auto target = std::clamp(ctx->target, 0.0f, 1.0f);
    float adapted_target_raw = ctx->total_weight == 0.0f 
        ? target 
        : 2.0f * target - (ctx->weighted_sum / ctx->total_weight);
    float adapted_target = std::clamp(adapted_target_raw, 0.0f, 1.0f);

    for (size_t i = 0; i < cur_p->size; ++i) {
        float dist = std::abs(
            (cur_p->data[i].p - adapted_target) * INV_WIDTH);
        cur_p->data[i].logit = PEAK_LOGIT_VALUE 
            - SHARPNESS * dist * dist / (1.0f + dist);
    }

    llama_sampler_softmax_impl(cur_p, false);
    const int idx = llama_sample_dist(cur_p, ctx->rng);
    cur_p->selected = idx;

    ctx->weighted_sum = ctx->original_probs[idx] 
        + ctx->decay * ctx->weighted_sum;
    ctx->total_weight = 1.0f + ctx->decay * ctx->total_weight;
}
```

### 9.2 Correct Initialization
```cpp
weighted_sum = target / (1.0f - clamped_decay)
total_weight = 1.0f / (1.0f - clamped_decay)
```

---

## 10. Conclusion

Adaptive-P provides a principled approach to controlling text generation creativity through direct probability targeting. Unlike temperature or truncation samplers, it enables nuanced control over which probability range the model prefers while handling the sparse, clustered distributions that real models produce.

---

## Figure Reference List

| ID | Description | Type | Section |
|----|-------------|------|---------|
| G-1 | Overview comparison on real tokens | Real data | 2 |
| G-2 | Temperature effect on real distribution | Real data | 3.1 |
| G-3 | Min-P truncation before/after | Illustrative | 3.3 |
| G-4 | XTC uniform redistribution failure | Illustrative | 3.4 |
| G-5 | Selective redistribution on 22-token real sample | Real data | 3.6 |
| G-6 | Bell curve transformation function | Illustrative (labeled) | 4.1 |
| G-7 | Four distribution pattern examples | Real data | 4.2 |
| G-8 | Calculated target drift over time | Real data | 4.3 |
| G-9 | Cluster pile-up: floor vs. unbounded | Real data | 4.5 |
| G-10 | Pre vs. post softmax | Real data | 4.6 |
| G-11 | Initialization effect comparison | Real data | 4.7 |
| G-12 | Target effect on selection distribution | Real data | 5.1 |
| G-13 | Decay effect comparison | Real data | 5.2 |
| G-14 | Combined min-p + Adaptive-P pipeline | Illustrative | 6.2 |
| G-15 | Aggregate selection scatter (25k tokens) | Real data | 7.1 |
| G-16 | Adaptive-P vs. Gaussian on clustered input | Real data | 7.2 |
| G-17 | Annotated token walkthrough examples | Real data | 7.3 |
