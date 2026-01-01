# Adaptive-P Sampler Documentation

A probability-targeting sampler for autoregressive language models that breaks high-confidence token chains while maintaining coherent output.

> [!NOTE]
> Implementation available via [llama.cpp PR #17927](https://github.com/ggml-org/llama.cpp/pull/17927)

---

## Documentation Sections

### [Abstract](sections/01_abstract.md)
Summary of Adaptive-P's probability-targeting approach, key contributions, and empirical findings.

### [1. Introduction](sections/02_introduction.md)
The high-confidence token chain problem, why temperature and truncation don't solve it, and how Adaptive-P's targeting approach differs.

### [2. Related Work](sections/03_related_work.md)
Comparison with Temperature, Top-K, Top-P, Min-P, XTC, and Mirostat. Why uniform redistribution fails and how selective redistribution works.

### [3. The Algorithm](sections/04_algorithm.md)
Core probability targeting, real distribution patterns (forced choice, binary split, clustered tail), configured vs. calculated target, the logit transformation function, and why unbounded negative logits matter.

### [5. Parameters](sections/05_parameters.md)
`target` (0.0–1.0), `decay` (0.0–0.99), internal constants. Includes elasticity/stubbornness/fishtailing behavior and why SHARPNESS isn't user-configurable.

### [6. Integration](sections/06_integration.md)
Chain positioning (must be last), Min-P complementarity, temperature interaction, samplers made unnecessary. Includes llama.cpp usage examples.

### [7. Empirical Validation](sections/07_empirical_validation.md)
Selection distribution analysis, target achievement, comparisons with temperature, adaptation dynamics, initialization validation, cross-model consistency.

### [8. Implementation Reference](sections/08_implementation.md)
Complete annotated C++ implementation, Python pseudocode, data structures, and porting notes.

### [9. Conclusion](sections/09_conclusion.md)
Summary of contributions, limitations, and future work directions.

---

## Quick Start

```bash
./llama-cli -m model.gguf \
    --min-p 0.05 \
    --adaptive-target 0.5 \
    --adaptive-decay 0.9 \
    -p "Once upon a time"
```

**Key parameters:**
- `target`: Desired average selection probability (0.5 = mid-range tokens)
- `decay`: History responsiveness (0.9 default; lower for peaked models)
