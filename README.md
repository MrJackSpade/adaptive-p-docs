# Adaptive-P Sampler Documentation

A sampler for autoregressive language models that selects tokens near a configurable target probability over time.

> [!NOTE]
> Implementation available in [llama.cpp#17927](https://github.com/ggml-org/llama.cpp/pull/17927)

---

## Documentation sections

### [Abstract](sections/01_abstract.md)
Summary of Adaptive-P's probability-targeting approach, key contributions, and empirical findings.

### [1. Introduction](sections/02_introduction.md)
The high-confidence token chain problem, why temperature and truncation don't solve it, and how Adaptive-P's targeting approach differs.

### [2. Related Work](sections/03_related_work.md)
Comparison with Temperature, Top-K, Top-P, Min-P, XTC, and Mirostat. Why renormalization fails and how selective redistribution works.

### [3. The Algorithm](sections/04_algorithm.md)
Core probability targeting, real distribution patterns (forced choice, binary split, clustered tail), configured vs. calculated target, the logit transformation function, and why unbounded negative logits matter.

### [4. Design Justification](sections/05_design_justification.md)
Why the logit transformation function was selected, empirical tuning of constants, and design tradeoffs.

### [5. Parameters](sections/06_parameters.md)
`target` (0.0–1.0), `decay` (0.0–0.99), internal constants. Includes elasticity/stubbornness/fishtailing behavior and why SHARPNESS isn't user-configurable.

### [6. Integration](sections/07_integration.md)
Chain positioning (must be last), Min-P complementarity, temperature interaction, samplers made unnecessary. Includes llama.cpp usage examples.

### [7. Empirical Validation](sections/08_empirical_validation.md)
Selection distribution analysis, target achievement, comparisons with temperature, adaptation dynamics, initialization validation, cross-model consistency.

### [8. Reference Implementation](sections/09_implementation.md)
Annotated C++ implementation, Python pseudocode, data structures, and porting notes.

### [9. Conclusion](sections/10_conclusion.md)
Summary of contributions, limitations, and future work directions.

---

## Quick Start

```bash
./llama-cli -m model.gguf \
    --samplers "min-p;adaptive-p" \
    --min-p 0.05 \
    --adaptive-target 0.5 \
    --adaptive-decay 0.9 \
    -p "Once upon a time"
```

This sampler exposes two parameters:

| Parameter name | Description                                                                                                | CLI argument          | Valid range | Default value | Notes                                                                                                                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------- | --------------------- | ----------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target`       | Select tokens near this probability                                                                        | `--adaptive-target N` | 0.0 - 1.0   | -1.0          | When set to -1.0, the adaptive probability transform is **disabled**, and instead it just samples normally. Note that since the default value is -1.0, the sampler is disabled by default. This is intentional. |
| `decay`        | Decay value for exponential moving average - lower values are more reactive, higher values are more stable | `--adaptive-decay N`  | 0.0 - 0.99  | 0.90          | Clamped to <=0.99 at init to avoid unbounded accumulation                                                                                                                                                       |
