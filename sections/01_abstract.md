# Abstract

We present Adaptive-P, an alternative sampling method for autoregressive language models that targets a specific probability range rather than truncating or uniformly scaling the token distribution. Unlike temperature sampling (which affects all tokens equally) or truncation methods like Top-P and Min-P (which make binary include/exclude decisions), Adaptive-P applies a continuous transformation that preferentially selects tokens near a user-specified target probability.

The method maintains a weighted moving average of previously selected token probabilities and dynamically adjusts its targeting to achieve this average over time. When recent selections skew toward high-probability tokens, the sampler compensates by targeting lower probabilities on subsequent steps, and vice versa. This adaptive behavior breaks the high-confidence token chains that produce repetitive, generic output—a phenomenon practitioners often call "slop".

This document presents:
- A probability-targeting paradigm as an alternative to standard temperature scaling
- An unbounded quadratic transformation that handles sparse, clustered real-world token distributions without probability pile-up
- Adaptive adjustment that prevents both monotony (always selecting high-probability tokens) and chaos (random low-quality selections)
- Clean integration with existing truncation samplers like Min-P, which complement Adaptive-P when used as a guardrail


Empirical evaluation across multiple models demonstrates that Adaptive-P successfully targets user-specified probability ranges while maintaining output coherence. In preliminary testing, the same target parameter produced similar selection patterns across tested models (GLM-4.x, Mistral, Cydonia), making it a practical tool for controlling the creativity-coherence tradeoff in text generation. Adaptive-P is intended for creative applications where output diversity is valued over predictability; standard likelihood-based metrics are not the intended goal of this sampler.
