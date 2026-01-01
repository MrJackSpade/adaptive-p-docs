# Adaptive-P Documentation TODO

## 03_related_work.md
- [ ] **Generate Min-P comparison samples** — Generate target 0 text with and without min-p to demonstrate guardrail effect (stub files created at `samples/target_0_no_minp.md` and `samples/target_0_with_minp.md`)

---

## 04_algorithm.md
- [ ] **Create Graph [G-7]: Illustrative Bell Curve** — Logit transformation curve centered at targets 0.3, 0.5, and 0.7 (label as "illustrative of transformation function only" since real distributions are sparse)
- [ ] **Create Graph [G-8]: Four Pattern Examples** — Four panels showing input→output probabilities for Forced Choice, Binary Split, Clustered Tail, and Competitive Mid-Range patterns (use actual samples.log data)
- [ ] **Create Graph [G-10]: Transformation Function Shape** — Plot PEAK - SHARPNESS × dist² / (1 + dist) showing quadratic core transitioning to linear tails
- [ ] **Create Graph [G-12]: Pre vs. Post Softmax** — Show raw logit values and corresponding post-softmax probabilities for a real sample

---

## 06_integration.md
- [ ] **Create Graph [G-16]: Combined Pipeline Effect** — Full vocabulary → min-p filter → Adaptive-P transform → final selection (show probability mass at each stage)

---

## 08_implementation.md
- [ ] **Test Python implementation** — Verify correctness before publishing

---

## 09_conclusion.md
- [x] **Add Acknowledgments** — mfunlimited, geechan, concedo, kurgan1138
- [ ] **Add References** — Mirostat paper, top-p paper, etc.
- [ ] **Add Appendix** (if needed)

---

## Summary

| Priority | Count | Items |
|----------|-------|-------|
| **Graphs** | 6 | G-3, G-7, G-8, G-10, G-12, G-16 |
| **Validation** | 1 | Python impl |
| **Text/Citations** | 1 | References |

**Note:** Sample files (`target_0.3_sample.md`, etc.) and most charts are already in place ✅
