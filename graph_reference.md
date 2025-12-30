# Graph Reference Summary

This document lists all graphs referenced in the paper sections with descriptions of what each should contain.

## Section 2: Introduction
| ID | Description | Data Source |
|----|-------------|-------------|
| G-1 | Overview comparison: input vs output distributions for Temperature, Min-P, XTC, and Adaptive-P on same real tokens | Generate from samples.log or similar |

## Section 3: Related Work
| ID | Description | Data Source |
|----|-------------|-------------|
| G-2 | Temperature effect on real distribution (T=0.5, 1.0, 1.5) | Generate from 18-token "serene" case |
| G-3 | Min-P truncation before/after | May need pre-min-p data |
| G-4 | XTC uniform redistribution failure showing fat tail | Simulated or theoretical |
| G-5 | Selective redistribution on "prestigious" 22-token sample | samples.log |
| G-6 | Selective vs uniform redistribution side-by-side | Combined from above |

## Section 4: Algorithm
| ID | Description | Data Source |
|----|-------------|-------------|
| G-7 | Illustrative bell curve at targets 0.3, 0.5, 0.7 (LABELED AS ILLUSTRATIVE) | Mathematical function plot |
| G-8 | Four pattern examples (forced, binary, clustered, competitive) | samples.log real data |
| G-9 | Calculated target drift over time | samples.log or new generation |
| G-10 | Transformation function shape (quadratic→linear) | Mathematical function plot |
| G-11 | Cluster pile-up: Gaussian vs Adaptive-P on "." token | samples.log (has both outputs) |
| G-12 | Pre vs post softmax comparison | samples.log |
| G-13 | Initialization effect (bad vs correct) | Image 0 + new correct version |

## Section 5: Parameters
| ID | Description | Data Source |
|----|-------------|-------------|
| G-14 | Target effect: selection histograms at 0.3, 0.5, 0.7, 0.9 | New generation runs |
| G-15 | Decay effect: curves at 0.5, 0.7, 0.8, 0.9 | Image 1 (already have this!) |

## Section 6: Integration
| ID | Description | Data Source |
|----|-------------|-------------|
| G-16 | Combined pipeline: vocabulary → min-p → Adaptive-P | Illustrative diagram |

## Section 7: Empirical Validation
| ID | Description | Data Source |
|----|-------------|-------------|
| G-17 | Selection scatter plot (25k+ tokens) | Large generation run |
| G-18 | Rolling average convergence | Derived from scatter data |
| G-19 | Adaptive-P vs Temperature distribution | Comparative generation |
| G-20 | Adaptive-P vs Gaussian on "." token | samples.log |
| G-21 | Target adaptation response | Image 2 (already have this!) |
| G-22 | Initialization comparison | Image 0 + correct version |
| G-23 | Cross-model selection patterns | Multi-model generation runs |

---

## Already Available (from your images)

**Image 0:** Bad initialization recovery (decay=0.99) - usable for G-13, G-22
**Image 1:** Selection rate vs probability at multiple decays - usable for G-15
**Image 2:** Target vs selected probability over time - usable for G-21
**Image 3:** Candidate/selected scatter with target line - usable for G-17

## Need to Generate

**Priority 1 (core proof):**
- G-8: Four pattern examples
- G-11: Cluster pile-up comparison
- G-14: Target effect comparison

**Priority 2 (supporting):**
- G-7: Illustrative transformation curve
- G-10: Transformation function math plot
- G-13 correct version: Good initialization comparison

**Priority 3 (nice to have):**
- G-2, G-3, G-4: Other sampler comparisons
- G-23: Cross-model analysis
