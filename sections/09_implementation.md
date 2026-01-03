# 8. Reference Implementation

This section details the llama.cpp implementation from [llama.cpp#17927](https://github.com/ggml-org/llama.cpp/pull/17927).

## 8.1. API definition

The sampler is exposed through llama.cpp's public C/C++ API in `llama.h`. Like other samplers in the codebase, it returns an opaque `llama_sampler` pointer that integrates with the sampler chain infrastructure. The function takes just three parameters: the probability target to aim for, the EMA decay rate controlling adaptation speed, and an RNG seed for reproducibility.

```cpp
    ///
    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);
```

## 8.2. Sampler state definition

The sampler maintains persistent state across sampling steps. The `target` and `decay` parameters are stored as `const` since they never change after initialization. The EMA is tracked via two accumulators: `weighted_sum` holds the decay-weighted sum of selected token probabilities, while `total_weight` holds the sum of the decay weights themselves. Dividing these yields the current EMA value. The `original_probs` vector caches the pre-transformation probability distribution so we can update the EMA with the *original* probability of whichever token gets selected after transformation.

```cpp
//
// adaptive-p sampler state
//
// maintains an exponential moving average of the *ORIGINAL* probabilities
// of selected tokens, used to compute an adapted target at each sampling step.
//
struct llama_sampler_adaptive_p {
    //
    // actual sampler-specific external parameters
    //
    const float target; // target probability (0.0 - 1.0; negative = disabled)
    const float decay;  // EMA decay; history ~= 1/(1-decay) tokens (0.0 - 0.99)
    //
    // set by sampler chain / per instance
    //
    const uint32_t seed; // RNG seed
    std::mt19937   rng;  // RNG
    //
    // member variables for EMA (internal state tracking / scratch space)
    //
    float              weighted_sum;   // sum(p_i * decay^i)
    float              total_weight;   // sum(decay^i), converges to 1/(1-decay)
    std::vector<float> original_probs; // pre-transform probs, cached for EMA update
};
```

## 8.3. Sampler state initialization

Initialization clamps the decay to a maximum of 0.99 to prevent unbounded accumulation in the EMA. The EMA state is initialized to its *converged* value rather than zero — this means `weighted_sum / total_weight` equals exactly `target` from the very first sample. This avoids cold-start bias where early tokens would otherwise be sampled without any adaptive steering. The geometric series `1 + decay + decay² + ...` converges to `1/(1-decay)`, which is why `total_weight` is initialized to that value.

```cpp
struct llama_sampler * llama_sampler_init_adaptive_p(
    float    target,
    float    decay,
    uint32_t seed
) {
    auto seed_cur = get_rng_seed(seed);
    float clamped_decay = std::clamp(decay, 0.0f, 0.99f); // clamp once
    return llama_sampler_init(
        /* .iface = */ &llama_sampler_adaptive_p_i,
        /* .ctx   = */ new llama_sampler_adaptive_p {
            /* .target         = */ target,
            /* .decay          = */ clamped_decay,
            /* .seed           = */ seed_cur,
            /* .rng            = */ std::mt19937(seed_cur),
            /* .weighted_sum   = */ target / (1.0f - clamped_decay),
            /* .total_weight   = */ 1.0f / (1.0f - clamped_decay),
            /* .original_probs = */ {},
        }
    );
}
```

## 8.4. Adaptive probability transformation constants

These constants control the shape of the logit transformation function. `DISTRIBUTION_WIDTH` defines how wide the "favored" probability region is around the target — probabilities within ±0.3 of the target receive relatively high logits. `PEAK_LOGIT_VALUE` sets the maximum logit assigned to tokens exactly at the target probability. `SHARPNESS` controls how steeply logits fall off as distance from the target increases. These values were empirically tuned; see section 4 (Design Justification) for details on why they aren't user-configurable.

```cpp
static constexpr float DISTRIBUTION_WIDTH =  0.3f;
static constexpr float PEAK_LOGIT_VALUE   =  5.0f;
static constexpr float SHARPNESS          = 10.0f;
static constexpr float INV_WIDTH          =  1.0f / DISTRIBUTION_WIDTH;
```

## 8.5. Core algorithm (`llama_sampler_adaptive_p_apply`)

The apply function is called once per sampling step. When the target is negative, the sampler acts as a pass-through — it just samples from the existing distribution without any transformation. This allows adaptive-p to be disabled at runtime without removing it from the sampler chain.

For normal operation, the algorithm proceeds in four phases. First, it applies softmax to get probabilities and caches them before any transformation. Second, it computes an *adapted* target using the formula `2 * target - EMA`: if the running average is above target, the adapted target drops below target to compensate, creating a self-correcting feedback loop. Third, it transforms the cached probabilities into new logits using a function that peaks at the adapted target and falls off with distance — quadratic near the target for fine differentiation, transitioning to linear in the tails to ensure proper suppression after the subsequent softmax. Finally, it samples from the transformed distribution and updates the EMA using the *original* (pre-transformation) probability of the selected token.

```cpp
static void llama_sampler_adaptive_p_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_adaptive_p *) smpl->ctx;

    if (ctx->target < 0.0f) {
        // at negative target values, adaptive-p is no-op
        // we simply sample from the existing distribution
        llama_sampler_softmax_impl(cur_p, false);
        cur_p->selected = llama_sample_dist(cur_p, ctx->rng);
        return;
    }

    // softmax and store the original probabilities
    llama_sampler_softmax_impl(cur_p, false);
    ctx->original_probs.resize(cur_p->size);
    for (size_t i = 0; i < cur_p->size; ++i) {
        ctx->original_probs[i] = cur_p->data[i].p;
    }

    // compute the adapted target probability for the current sampling step
    auto target = std::clamp(ctx->target, 0.0f, 1.0f);
    float adapted_target = std::clamp(
        ctx->total_weight == 0.0f ? target : 2.0f * target - (ctx->weighted_sum / ctx->total_weight),
        0.0f, 1.0f
    );

    // adaptive probability transform
    //
    // quadratic near target for fine differentiation, transitioning to linear decay in the
    // tails. unbounded negative logits ensure proper suppression of far-from-target tokens
    // after the softmax.
    //
    for (size_t i = 0; i < cur_p->size; ++i) {
        float dist = std::abs((cur_p->data[i].p - adapted_target) * INV_WIDTH);
        cur_p->data[i].logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0f + dist);
    }

    // softmax and sample from the transformed distribution
    llama_sampler_softmax_impl(cur_p, false);
    const int idx   = llama_sample_dist(cur_p, ctx->rng);
    cur_p->selected = idx;

    // update EMA with the original probability of the selected token
    ctx->weighted_sum = ctx->original_probs[idx] + ctx->decay * ctx->weighted_sum;
    ctx->total_weight = 1.0f + ctx->decay * ctx->total_weight;
}
```

## 8.6. Sampler state reset

The reset function restores the sampler to its initial state. Since `target` and `decay` are const and never modified, and `original_probs` is completely overwritten on each apply call, only the EMA accumulators need resetting. They're restored to the same converged-at-target values used during initialization, ensuring consistent behavior when a sampler is reused across independent generation sessions.

```cpp
static void llama_sampler_adaptive_p_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_adaptive_p *) smpl->ctx;
    // ctx->target and ctx->decay never change after init, so it's safe to keep them as is.
    // original_probs is completely overwritten on every call to _apply.
    // so we only need to reset the EMA state.
    ctx->weighted_sum = ctx->target / (1.0f - ctx->decay);
    ctx->total_weight = 1.0f / (1.0f - ctx->decay);
}
```

## 8.7. Sampler state clone

Cloning creates an independent copy of the sampler with identical state. This is used when forking generation paths (e.g., for beam search or speculative decoding). The clone function creates a fresh sampler with the same parameters, then manually copies over the mutable state: the cached original probabilities, the EMA accumulators, and critically, the RNG state. Copying the RNG ensures the clone will produce the same sequence of random choices as the original would have from that point forward.

```cpp
static struct llama_sampler * llama_sampler_adaptive_p_clone(const struct llama_sampler * smpl) {
    const auto * ctx  = (const llama_sampler_adaptive_p *) smpl->ctx;
    auto * result     = llama_sampler_init_adaptive_p(ctx->target, ctx->decay, ctx->seed);
    auto * result_ctx = (llama_sampler_adaptive_p *) result->ctx;

    // copy everything (target, decay, and seed are already set)
    result_ctx->original_probs = ctx->original_probs;
    result_ctx->weighted_sum   = ctx->weighted_sum;
    result_ctx->total_weight   = ctx->total_weight;
    result_ctx->rng            = ctx->rng;

    return result;
}
```