# 9. Conclusion

## 9.1 Summary

We have presented Adaptive-P, an alternative sampling method for autoregressive language models that introduces probability targeting as an alternative to traditional truncation and scaling approaches.

**The problem addressed:**

High-confidence token chains produce repetitive, generic output. Once a model commits to a common phrase pattern, each subsequent token reinforces the chain. Standard sampling methods—temperature and truncation—don't provide a mechanism to prefer mid-range probabilities over the dominant choice.

**The solution:**

Adaptive-P applies a distance-based transformation that preferentially selects tokens near a user-specified target probability. The transformation uses:
- Quadratic core for fine differentiation among close candidates
- Linear tails for unbounded suppression of distant tokens
- Adaptive adjustment to maintain target average over time

**Key properties:**
1. **Direct targeting:** Users specify their desired probability range, not an abstract "temperature" value
2. **Chain breaking:** Consecutive high-confidence selections trigger adaptive compensation, making alternatives more attractive
3. **No fat tails:** Unlike XTC-style redistribution, probability concentrates on near-target tokens rather than spreading to undesirable options


## 9.2 Contributions

This work offers the following:

1. **Paradigm:** Probability targeting as a principled alternative to truncation-based sampling. Rather than asking "which tokens should we keep?" we ask "which probability range should we prefer?"

2. **Algorithm:** The specific transformation design—unbounded quadratic with adaptive history—that makes probability targeting work on real token distributions with their sparse, clustered characteristics.

3. **Integration:** Clear guidance on sampler chain positioning and interaction with existing methods, particularly the complementary relationship with Min-P.

4. **Validation:** Empirical demonstration that Adaptive-P achieves its targeting goal across varied models and prompts.

## 9.3 Limitations

**Adaptive-P cannot create diversity that doesn't exist.** If a model's distribution has only one viable token (p ≈ 1.0), or all candidates are far from target, the sampler cannot manufacture mid-range options. It operates on available candidates.

**Sharp distributions limit effectiveness.** Heavily instruction-tuned models with peaked distributions show less dramatic effects from Adaptive-P compared to base models with spread distributions.

**Parameter tuning is still required.** While "target" is more intuitive than "temperature," users must still learn what probability range produces their desired output quality. The default (0.5) works well for many cases but isn't universal.

## 9.4 Future Work

**Adaptive constants:** The current SHARPNESS value (10.0) was empirically tuned. Automatic tuning based on input distribution characteristics could improve robustness.

**Multi-token lookahead:** Current implementation considers only the immediate selection. Considering how the selection affects future distributions could enable longer-term coherence optimization.

**Integrated quality metrics:** Coupling Adaptive-P with perplexity or coherence scoring could allow dynamic target adjustment based on output quality rather than purely probability-based feedback.

**Broader framework support:** Current implementation targets llama.cpp. Ports to vLLM, Hugging Face Transformers, and other popular frameworks would increase accessibility.

## 9.5 Availability

Adaptive-P is implemented in llama.cpp and available via PR to the main repository. Source code and documentation are provided under permissive license.

---

## Acknowledgments

This work wouldn't exist without the following contributors from the BeaverAI community for their invaluable assistance:

- **[mfunlimited](https://github.com/ddh0)** — Created and maintained the llama.cpp mainline PR (#17927), ported the C# implementation to C++, iterated through multiple algorithm versions, and coordinated with upstream maintainers
- **[concedo](https://github.com/LostRuins)** — Identified the long tail issue with the original Lorentzian formula, collaborated on deriving the correct initialization formula (`target / (1 - decay)`), validated mathematical correctness, and implemented the sampler in KoboldCpp
- **[AesSedai](https://github.com/AesSedai)** — Created and maintained the SillyTavern fork with sampler support, hosted test APIs for community testing, and created Docker images for RunPod deployment
- **[Geechan](https://github.com/Geechan)** — Community coordination, documentation planning, opened the ik_llama feature request, and organized testing efforts across models

Special thanks to the broader llama.cpp community for their continued development of accessible LLM inference tooling.

> [!NOTE]
> **AI Disclosure:** This documentation was written with the assistance of AI language models. However, the Adaptive-P algorithm, its mathematical formulation, empirical validation, and all technical contributions presented in this paper are the original work of the author and human collaborators listed above. AI was used solely as a writing aid for documentation purposes.

---

## References

[1] **Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y.** (2020). *The Curious Case of Neural Text Degeneration.* International Conference on Learning Representations (ICLR). arXiv:1904.09751.
> Introduced **nucleus sampling (top-p)**, showing that maximization-based decoding leads to degenerate text and proposing dynamic truncation of the probability distribution.

[2] **Fan, A., Lewis, M., & Dauphin, Y.** (2018). *Hierarchical Neural Story Generation.* Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL).
> Introduced **top-k sampling** for neural text generation, demonstrating improved diversity over beam search for creative writing tasks.

[3] **Basu, S., Ramachandran, G. S., Keskar, N. S., & Varshney, L. R.** (2021). *Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity.* International Conference on Learning Representations (ICLR). arXiv:2007.14966.
> Introduced **Mirostat**, which dynamically adjusts top-k to maintain target perplexity. This work was the original inspiration for Adaptive-P's probability targeting approach.

[4] **Nguyen, M. N., Baker, A., Neo, C., Roush, A., Kirsch, A., & Shwartz-Ziv, R.** (2025). *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs.* International Conference on Learning Representations (ICLR). arXiv:2407.01082.
> Introduced **min-p sampling**, which dynamically sets a probability threshold relative to the top token. Min-p serves as a complementary guardrail to Adaptive-P.

[5] **Hinton, G., Vinyals, O., & Dean, J.** (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
> Introduced softmax **temperature scaling** for knowledge distillation, establishing the foundational mechanism used by temperature sampling in language models.

[6] **Ackley, D. H., Hinton, G. E., & Sejnowski, T. J.** (1985). *A Learning Algorithm for Boltzmann Machines.* Cognitive Science, 9(1), 147-169.
> Early work on **stochastic sampling** with temperature in neural networks, establishing the temperature parameter's role in controlling exploration-exploitation tradeoffs.


