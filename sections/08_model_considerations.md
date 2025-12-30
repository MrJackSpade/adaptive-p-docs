# 7. Model Considerations

Adaptive-P's effectiveness varies with model characteristics. This section provides guidance on which models benefit most and how to adjust parameters for different model types.

## 7.1 Models with Diverse Distributions

**Characteristics:**
- Probability mass spread across many candidates
- Multiple mid-range options available for most tokens
- Trained on varied, natural data

**Examples:** Llama, Mistral, Qwen, GLM

**Adaptive-P behavior:**

These models respond well to Adaptive-P with default parameters. The mid-range tokens exist and can be selected when targeted. Users report noticeable differences from baseline sampling—more varied vocabulary, less formulaic phrasing.

**Recommended settings:**
- Target: 0.4–0.6 (mid-range benefits from diverse options)
- Decay: 0.9 (default)
- Min-P: 0.05 (standard guardrail)

## 7.2 Models with Sharp Distributions

**Characteristics:**
- Most probability mass concentrated in top 1-3 tokens
- Few mid-range candidates available
- Often heavily RLHF'd or trained on synthetic data

**Examples:** Gemma, Phi, heavily instruction-tuned variants

**Adaptive-P behavior:**

When distributions are peaked, there may be no tokens near target 0.5. The sampler compensates by selecting the closest available—often still the dominant token.

Users report less dramatic effects compared to temperature adjustment. This is expected: Adaptive-P can't create diversity that doesn't exist in the distribution. It operates on available candidates.

**Recommended settings:**
- Target: 0.6–0.8 (work with available high-probability range)
- Decay: 0.7–0.85 (lower than default—see note below)
- Min-P: 0.1 (aggressive cleanup since distribution is sharp anyway)

> [!NOTE]
> **Lower decay for peaked distributions:** With limited candidates, the sampler often has no choice but to select whatever's available. High decay (0.9+) causes the history to "remember" these forced high-probability selections, leading to fishtailing—the calculated target swings wildly trying to compensate for choices that weren't really choices. Lower decay (0.7–0.85) lets the sampler "forget" forced selections more quickly.

**Alternative approach:**

For peaked-distribution models, consider:
1. Light temperature increase (1.1–1.2) to spread probability slightly
2. Then Adaptive-P targeting within the spread range

This pipeline creates the diversity that Adaptive-P then targets.

## 7.3 Instruct vs. Base Models

**Instruct models:**

Generally have sharper distributions due to RLHF/DPO training. They've learned to commit to specific continuations. Adaptive-P may need higher target or temperature preprocessing.

**Base models:**

Often have more spread distributions. Adaptive-P with default parameters typically works well. Users report particularly strong results with base models for creative writing.

## 7.4 Quantization Effects

**Observation:**

Heavily quantized models (Q4, Q3) sometimes show numerical artifacts in probability distributions. Very small probabilities may be represented identically, creating artificial clustering.

**Recommendation:**

Use slightly higher min-p threshold (0.07–0.1) to remove tokens whose probabilities may be unreliable due to quantization artifacts. Adaptive-P operates normally above this threshold.

## 7.5 Model-Specific Tuning

If default parameters don't produce satisfactory results:

**Step 1: Diagnose**

Generate with logging enabled. Check:
- How many candidates survive min-p? (Pattern diversity)
- What probability range are candidates in? (Available targeting space)
- Does calculated target converge to configured? (Adaptation health)

**Step 2: Adjust decay**

For peaked distributions, lower decay (0.7–0.85) helps prevent fishtailing—where the calculated target swings wildly between extremes after forced high-probability selections.

> [!CAUTION]
> **Don't chase the cluster with target.** If candidates cluster at 0.8, setting target to 0.8 defeats Adaptive-P's purpose—you're just accepting what the model already wants to output. Instead, keep target at your desired creativity level and let the sampler work with available options.

**Step 3: Consider preprocessing (as last resort)**

If distribution is too peaked for any target to work well, light temperature (1.1) before Adaptive-P can spread the distribution. Use sparingly—this is a workaround, not the intended workflow.

**Step 4: Adjust decay**

If outputs seem too variable, raise decay (0.95).
If outputs seem stuck or fishtailing, lower decay (0.7–0.8).

Most users find default (0.9) works well; adjust if you notice oscillating behavior.
