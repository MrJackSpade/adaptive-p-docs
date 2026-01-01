# TODO.md - Adaptive-P Documentation Fixes

## Critical (Fix Before Publishing)

- [ ] Fix duplicate Section 4 numbering (Design Rationale and Parameters both labeled Section 4)
- [ ] Renumber all sections to match Table of Contents (should be 1-9)
- [ ] Complete "Target 0 WITH Min-P" sample (currently contains TODO placeholder)
- [ ] Fix compilation date "2026-01-01" (typo or future-dating)
- [ ] Reconcile "Target 0" header vs `--adaptive-target 0.1` in settings block for no-minp sample

## Empirical Validation

- [ ] Add perplexity measurements comparing Adaptive-P at different targets vs baseline
- [ ] Add MAUVE or other distribution-level metric comparison
- [ ] Consider human preference evaluation (even informal A/B)
- [ ] Add scatter plots for temperature sampling as baseline comparison
- [ ] Add scatter plots for top-p sampling as baseline comparison
- [ ] Document at least one failure case / bad output example
- [ ] Add ablation study: adaptive targeting vs static targeting (show the "fishtailing")
- [ ] Test and include samples from Llama-3 or Llama-3.1
- [ ] Test and include samples from Qwen
- [ ] Test and include samples from Phi
- [ ] Test and include samples from Gemma
- [ ] Diversify sample prompts beyond horror stories

## Technical Claims Requiring Evidence

- [ ] Add citation or experimental data for XTC "fat tail accumulation" claim
- [ ] Add citation or data for "modern models produce sharper distributions" (Mirostat section)
- [ ] Soften XTC critique language to "our observations suggest" if no hard data available
- [ ] Soften Mirostat critique language to "our observations suggest" if no hard data available
- [ ] Add experimental comparison: Adaptive-P vs XTC on same prompts
- [ ] Add experimental comparison: Adaptive-P vs Mirostat on same prompts
- [ ] Justify the 94%→50% transform behavior more thoroughly (why is this desirable?)
- [ ] Document sensitivity analysis for magic constants (PEAK, SHARPNESS, WIDTH)
- [ ] Fix 2× multiplier derivation to properly address exponential weighting (not just simple average)

## Missing Technical Content

- [ ] Add computational overhead / performance benchmarks (tokens/sec impact)
- [ ] Document batch inference implications
- [ ] Document interaction with speculative decoding
- [ ] Document interaction with beam search
- [ ] Explain mathematically why extreme temperature (T<0.5, T>1.5) causes problems
- [ ] Add failure mode documentation (what prompts/distributions break it?)
- [ ] Document edge case: what happens when all tokens are far from target?
- [ ] Document edge case: what happens with very long context?

## Structural & Presentation

- [ ] Add "Limitations" callout box early in document (Section 1 or 2), not just conclusions
- [ ] Move "destructive transformation" from note box to main text (it's fundamental)
- [ ] Make "must be last in chain" constraint more prominent
- [ ] Make Mirostat mutual exclusivity more prominent
- [ ] Trim abstract to 150-250 words per arxiv conventions (noted in existing TODO)
- [ ] Add XTC to References section (currently uncited)
- [ ] Expand References beyond 6 citations if positioning as academic paper

## Terminology & Tone

- [ ] Decide on audience: academic paper vs community documentation
- [ ] If academic: remove "slop" or define it formally
- [ ] If academic: reconsider "novel" claim in abstract (probability-distance weighting isn't new)
- [ ] If community: consider relaxing academic framing
- [ ] Review all claims for overclaiming (novel, consistent, etc.)

## Code & Implementation

- [ ] Test Python reference implementation (currently marked "not tested")
- [ ] Add tested Python implementation or remove the warning
- [ ] Document llama.cpp PR status more specifically (merge timeline if known)
- [ ] Add vLLM integration example
- [ ] Add Hugging Face LogitsProcessor example

## Samples

- [ ] Generate missing "Target 0 WITH Min-P" sample
- [ ] Add samples from non-GLM models
- [ ] Add at least one non-horror-story prompt
- [ ] Add a "bad output" sample showing failure mode
- [ ] Add sample showing fishtailing with low decay
- [ ] Add sample showing stubbornness with high decay

## Nice to Have

- [ ] Add interactive demo or notebook
- [ ] Create simplified "quick start" section for practitioners
- [ ] Add FAQ section addressing anticipated criticisms
- [ ] Add troubleshooting section
- [ ] Consider splitting into "paper" and "documentation" versions