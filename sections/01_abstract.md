# Abstract

We present Adaptive-P, a novel sampling method for autoregressive language models that targets a specific probability range rather than truncating or uniformly scaling the token distribution. Unlike temperature sampling (which affects all tokens equally) or truncation methods like top-p and min-p (which make binary include/exclude decisions), Adaptive-P applies a continuous transformation that preferentially selects tokens near a user-specified target probability.

The method maintains a weighted moving average of previously selected token probabilities and dynamically adjusts its targeting to achieve this average over time. When recent selections skew toward high-probability tokens, the sampler compensates by targeting lower probabilities on subsequent steps, and vice versa. This adaptive behavior breaks the high-confidence token chains that produce repetitive, generic output—a phenomenon often termed "slop" in the LLM community.

Key contributions include:
- A probability-targeting paradigm as an alternative to truncation-based sampling
- An unbounded quadratic transformation that handles sparse, clustered real-world token distributions without probability pile-up
- Adaptive adjustment that prevents both monotony (always selecting high-probability tokens) and chaos (random low-quality selections)
- Clean integration with existing sampler chains, complementing min-p as a guardrail

Empirical evaluation across multiple models demonstrates that Adaptive-P successfully targets user-specified probability ranges while maintaining output coherence. The same target parameter produces consistent behavior across different model architectures, making it a practical tool for controlling the creativity-coherence tradeoff in text generation.

<!-- TODO: Abstract length (~250 words, arxiv typically wants 150-250)
  @claude: May need trimming after full paper is drafted.
  @loxifi: 
-->
