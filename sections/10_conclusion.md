# 9. Conclusion

## 9.1 Summary

We have presented Adaptive-P, a novel sampling method for autoregressive language models that introduces probability targeting as an alternative to traditional truncation and scaling approaches.

**The problem addressed:**

High-confidence token chains produce repetitive, generic output. Once a model commits to a common phrase pattern, each subsequent token reinforces the chain. Standard sampling methods—temperature and truncation—don't provide a mechanism to prefer mid-range probabilities over the dominant choice.

**The solution:**

Adaptive-P applies a distance-based transformation that preferentially selects tokens near a user-specified target probability. The transformation uses:
- Quadratic core for fine differentiation among close candidates
- Linear tails for unbounded suppression of distant tokens
- Adaptive adjustment to maintain target average over time

**Key properties:**
1. **Direct targeting:** Users specify the desired probability range, not an abstract "temperature" value
2. **Chain breaking:** Consecutive high-confidence selections trigger adaptive compensation, making alternatives more attractive
3. **No fat tails:** Unlike XTC-style redistribution, probability concentrates on near-target tokens rather than spreading to garbage
4. **Cross-model consistency:** The same target produces similar selection patterns across different architectures

## 9.2 Contributions

This work makes the following contributions:

1. **Paradigm:** Probability targeting as a principled alternative to truncation-based sampling. Rather than asking "which tokens should we keep?" we ask "which probability range should we prefer?"

2. **Algorithm:** The specific transformation design—unbounded quadratic with adaptive history—that makes probability targeting work on real token distributions with their sparse, clustered characteristics.

3. **Integration:** Clear guidance on sampler chain positioning and interaction with existing methods, particularly the complementary relationship with min-p.

4. **Validation:** Empirical demonstration that Adaptive-P achieves its targeting goal across varied models and prompts.

## 9.3 Limitations

**Adaptive-P cannot create diversity that doesn't exist.** If a model's distribution has only one viable token (p ≈ 1.0), or all candidates are far from target, the sampler cannot manufacture mid-range options. It operates on available candidates.

**Sharp distributions limit effectiveness.** Heavily instruction-tuned models with peaked distributions show less dramatic effects from Adaptive-P compared to base models with spread distributions.

**Parameter tuning still required.** While "target" is more intuitive than "temperature," users must still learn what probability range produces their desired output quality. The default (0.5) works well for many cases but isn't universal.

## 9.4 Future Work

**Adaptive constants:** The current SHARPNESS value (10.0) was empirically tuned. Automatic tuning based on input distribution characteristics could improve robustness.

**Multi-token lookahead:** Current implementation considers only the immediate selection. Considering how the selection affects future distributions could enable longer-term coherence optimization.

**Integrated quality metrics:** Coupling Adaptive-P with perplexity or coherence scoring could allow dynamic target adjustment based on output quality rather than purely probability-based feedback.

**Broader framework support:** Current implementation targets llama.cpp. Ports to vLLM, Hugging Face Transformers, and other popular frameworks would increase accessibility.

## 9.5 Availability

Adaptive-P is implemented in llama.cpp and available via PR to the main repository. Source code and documentation are provided under permissive license.

---

<!--
AUTHOR NOTE: Final sections to consider adding:
- Acknowledgments (loxifi, geechan, mfunlimited as key contributors)
- References (Mirostat paper, original top-p paper, etc.)
- Appendix with additional experimental results

Also consider whether the paper needs a "Related Work" references section 
with proper academic citations for temperature, top-k, top-p, Mirostat, etc.
-->
