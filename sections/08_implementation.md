# 7. Implementation Reference

This section provides the complete implementation for reference and porting to other frameworks.

## 7.1 Data Structures

```cpp
struct llama_sampler_adaptive_p {
    // Configuration (immutable after init)
    float    target;          // Configured target probability
    float    decay;           // History decay rate
    uint32_t seed;            // RNG seed
    
    // Runtime state
    std::mt19937 rng;         // Random number generator
    float weighted_sum;       // Running weighted sum of selected probabilities
    float total_weight;       // Running sum of weights
    
    // Temporary storage (per-step)
    std::vector<float> original_probs;  // Original probabilities before transform
};
```

## 7.2 Constants

```cpp
#define PEAK_LOGIT_VALUE    5.0f    // Maximum logit for on-target tokens
#define SHARPNESS           10.0f   // Transformation curve steepness
#define DISTRIBUTION_WIDTH  0.2f    // Probability distance scaling
#define INV_WIDTH           (1.0f / DISTRIBUTION_WIDTH)  // Precomputed inverse
```

## 7.3 Initialization

```cpp
struct llama_sampler * llama_sampler_init_adaptive_p(
    float target, 
    float decay, 
    uint32_t seed
) {
    auto seed_cur = get_rng_seed(seed);
    float clamped_decay = std::clamp(decay, 0.0f, 0.99f);
    
    return llama_sampler_init(
        &llama_sampler_adaptive_p_i,
        new llama_sampler_adaptive_p{
            /* .target         = */ target,
            /* .decay          = */ clamped_decay,
            /* .seed           = */ seed_cur,
            /* .rng            = */ std::mt19937(seed_cur),
            // Correct initialization: prime as if target already achieved
            /* .weighted_sum   = */ target / (1.0f - clamped_decay),
            /* .total_weight   = */ 1.0f / (1.0f - clamped_decay),
            /* .original_probs = */ {},
        });
}
```

**Critical:** The weighted_sum and total_weight initialization prevents warmup artifacts. Starting at (0, 0) would cause the calculated target to spike low on first high-probability selection.

## 7.4 Core Algorithm

```cpp
static void llama_sampler_adaptive_p_apply(
    struct llama_sampler * smpl, 
    llama_token_data_array * cur_p
) {
    auto * ctx = (llama_sampler_adaptive_p *) smpl->ctx;

    // ===== SECTION 1: Disabled check =====
    if (ctx->target < 0.0f) {
        // Negative target = disabled
        // Sample from unmodified distribution
        llama_sampler_softmax_impl(cur_p, false);
        cur_p->selected = llama_sample_dist(cur_p, ctx->rng);
        return;
    }

    // ===== SECTION 2: Store original probabilities =====
    // Apply softmax to get current probabilities
    llama_sampler_softmax_impl(cur_p, false);
    
    // Save for history update later
    ctx->original_probs.resize(cur_p->size);
    for (size_t i = 0; i < cur_p->size; ++i) {
        ctx->original_probs[i] = cur_p->data[i].p;
    }

    // ===== SECTION 3: Compute adapted target =====
    auto target = std::clamp(ctx->target, 0.0f, 1.0f);
    
    float adapted_target_raw = ctx->total_weight == 0.0f 
        ? target 
        : 2.0f * target - (ctx->weighted_sum / ctx->total_weight);
    
    float adapted_target = std::clamp(adapted_target_raw, 0.0f, 1.0f);

    // ===== SECTION 4: Apply transformation =====
    for (size_t i = 0; i < cur_p->size; ++i) {
        // Distance from target, scaled
        float dist = std::abs(
            (cur_p->data[i].p - adapted_target) * INV_WIDTH
        );
        
        // Unbounded quadratic transformation
        cur_p->data[i].logit = PEAK_LOGIT_VALUE 
            - SHARPNESS * dist * dist / (1.0f + dist);
    }

    // ===== SECTION 5: Sample from transformed distribution =====
    llama_sampler_softmax_impl(cur_p, false);
    const int idx = llama_sample_dist(cur_p, ctx->rng);
    cur_p->selected = idx;

    // ===== SECTION 6: Update history =====
    // Use ORIGINAL probability of selected token
    ctx->weighted_sum = ctx->original_probs[idx] 
        + ctx->decay * ctx->weighted_sum;
    ctx->total_weight = 1.0f 
        + ctx->decay * ctx->total_weight;
}
```

## 7.5 Supporting Functions

**Softmax implementation** (typical):

```cpp
static void llama_sampler_softmax_impl(
    llama_token_data_array * cur_p, 
    bool use_temperature
) {
    float max_logit = -INFINITY;
    for (size_t i = 0; i < cur_p->size; ++i) {
        max_logit = std::max(max_logit, cur_p->data[i].logit);
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p = expf(cur_p->data[i].logit - max_logit);
        sum += cur_p->data[i].p;
    }
    
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= sum;
    }
}
```

**Sampling function** (typical):

```cpp
static int llama_sample_dist(
    llama_token_data_array * cur_p, 
    std::mt19937 & rng
) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cumsum += cur_p->data[i].p;
        if (r <= cumsum) {
            return i;
        }
    }
    return cur_p->size - 1;  // Fallback
}
```

## 7.6 Reset and Clone

For proper sampler lifecycle management:

```cpp
static void llama_sampler_adaptive_p_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_adaptive_p *) smpl->ctx;
    
    // Reset to initial state (as if freshly initialized)
    float clamped_decay = std::clamp(ctx->decay, 0.0f, 0.99f);
    ctx->weighted_sum = ctx->target / (1.0f - clamped_decay);
    ctx->total_weight = 1.0f / (1.0f - clamped_decay);
    
    // Re-seed RNG
    ctx->rng = std::mt19937(ctx->seed);
}

static struct llama_sampler * llama_sampler_adaptive_p_clone(
    const struct llama_sampler * smpl
) {
    const auto * ctx = (const llama_sampler_adaptive_p *) smpl->ctx;
    return llama_sampler_init_adaptive_p(ctx->target, ctx->decay, ctx->seed);
}
```

## 7.7 Porting Notes

> [!CAUTION]
> **The Python implementation below is provided for reference only and has not been tested.** Use with caution and verify correctness before use in production.

**Python reference implementation:**

```python
import numpy as np

class AdaptiveP:
    def __init__(self, target=0.5, decay=0.9):
        self.target = target
        self.decay = np.clip(decay, 0.0, 0.99)
        self.weighted_sum = target / (1.0 - self.decay)
        self.total_weight = 1.0 / (1.0 - self.decay)
    
    def sample(self, logits):
        if self.target < 0:
            probs = softmax(logits)
            return np.random.choice(len(probs), p=probs)
        
        # Get original probabilities
        original_probs = softmax(logits)
        
        # Compute adapted target
        avg = self.weighted_sum / self.total_weight
        adapted_target = np.clip(2 * self.target - avg, 0, 1)
        
        # Transform
        PEAK = 5.0
        SHARPNESS = 10.0
        INV_WIDTH = 5.0
        
        dist = np.abs((original_probs - adapted_target) * INV_WIDTH)
        new_logits = PEAK - SHARPNESS * dist * dist / (1 + dist)
        
        # Sample
        new_probs = softmax(new_logits)
        idx = np.random.choice(len(new_probs), p=new_probs)
        
        # Update history
        self.weighted_sum = original_probs[idx] + self.decay * self.weighted_sum
        self.total_weight = 1.0 + self.decay * self.total_weight
        
        return idx

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

**Framework integration points:**

| Framework | Integration Point |
|-----------|------------------|
| llama.cpp | Sampler chain (`llama_sampler_chain_add`) |
| vLLM | Custom sampler class |
| Hugging Face | `LogitsProcessor` subclass |
| exllamav2 | Sampler settings |

