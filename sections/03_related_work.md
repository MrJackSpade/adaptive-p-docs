# 2. Related Work and Comparative Analysis

This section examines existing sampling methods, their limitations, and how Adaptive-P addresses these gaps. For each method, we analyze both theoretical properties and practical behavior on real token distributions.

## 2.1 Temperature Sampling

Temperature scaling is the most widely used sampling parameter. It divides all logits by a temperature value T before softmax:

```
logit_scaled = logit / T
```

**Properties:**
- T < 1.0: Sharpens the distribution, making high-probability tokens more dominant
- T = 1.0: No change from model's natural distribution  
- T > 1.0: Flattens the distribution, spreading probability mass more evenly

**Limitations:**

Temperature affects all tokens uniformly. It cannot express preferences like "I want mid-range tokens, not the top token or garbage tokens." When you increase temperature to encourage variety, you also increase the probability of low-quality tokens. When you decrease it for coherence, you reinforce the high-confidence chains.

The effect is also highly dependent on the input distribution's shape. Applying T=1.2 to a peaked distribution (one token at 0.8) produces a very different result than applying it to a flat distribution (many tokens around 0.1). This makes temperature difficult to tune consistently across different models and contexts.

> **Graph [G-2]: Temperature Effect on Real Distribution**  
> *Show a real multi-token distribution (e.g., the 18-token "serene" case from samples.log) with pre/post-softmax probabilities at T=0.5, T=1.0, T=1.5. Demonstrate that temperature shifts everything uniformly rather than targeting a range.*

<!-- TODO: G-2 data source
  @claude: Suggest "serene" example (18 tokens) with nice spread.
  @loxifi: 
-->

## 2.2 Top-K Sampling

Top-K sampling keeps only the K highest-probability tokens and renormalizes.

**Properties:**
- Simple threshold: exactly K tokens remain
- Removes low-probability mass entirely

**Limitations:**

The fixed K value doesn't adapt to distribution shape. For a peaked distribution where the top 3 tokens have 0.95 cumulative probability, K=50 is wasteful—you're including 47 tokens that will rarely be selected anyway. For a flat distribution where probability is spread across hundreds of reasonable options, K=50 may be too restrictive.

More fundamentally, Top-K answers "which tokens are most likely?" but provides no mechanism for preferring specific probability ranges within the kept set.

## 2.3 Top-P (Nucleus Sampling)

Top-P keeps the smallest set of tokens whose cumulative probability exceeds threshold P.

**Properties:**
- Adaptive threshold: includes more tokens when distribution is flat
- Commonly used default: P = 0.9 or 0.95

**Limitations:**

Like Top-K, Top-P is a truncation method. It makes binary include/exclude decisions without expressing preferences within the kept set. A token at 0.4 probability and a token at 0.05 probability are treated identically—both are "in" if they fall within the nucleus.

Top-P also doesn't specifically encourage mid-range selection. If the distribution has a clear winner, that winner will dominate even with Top-P=0.9.

## 2.4 Min-P

Min-P removes tokens whose probability falls below a threshold relative to the most probable token:

```
threshold = max_probability * min_p
```

**Properties:**
- Scales with distribution shape: peaked distributions have higher absolute thresholds
- Effective garbage removal: tokens far below the leader are excluded
- Commonly used values: min_p = 0.05 to 0.1

**Strengths:**

Min-P is excellent at what it does: removing tokens that should never be selected regardless of other sampling decisions. It prevents the pathological case where temperature increase allows garbage tokens to accumulate probability.

**Relationship to Adaptive-P:**

Min-P and Adaptive-P are complementary. Min-P serves as a guardrail that cleans the candidate pool. Adaptive-P then operates on the remaining quality candidates, applying preference among them.

The samples in this paper show tokens already filtered to p > 0.01—that's min-p at work. Adaptive-P assumes this cleanup has happened and focuses on the selection decision among viable candidates.

> **Graph [G-3]: Min-P as Guardrail**  
> *Show a full vocabulary distribution (with garbage tail) before min-p, then the cleaned distribution after min-p, then the Adaptive-P transform result. Demonstrate the pipeline.*

<!-- TODO: Pre-min-p data needed for G-3
  @claude: Current samples.log already has min-p applied. Need pre-filtered data.
  @loxifi: 
-->

## 2.5 XTC (eXclude Top Choices)

XTC randomly removes high-probability tokens to force consideration of alternatives.

**Mechanism:**
1. Identify tokens above a probability threshold
2. Randomly exclude some portion of these top tokens
3. Renormalize and sample from remaining candidates

**The Critical Flaw: Uniform Redistribution**

When XTC removes top tokens, the probability mass must go somewhere. Standard implementations redistribute uniformly across all remaining tokens—including garbage tokens at the far tail.

This causes **fat tail accumulation**: removing a 0.6 probability token and redistributing to 1000 remaining tokens gives each an extra 0.0006. But those 1000 tail tokens, collectively, now have significant mass. The probability of selecting a low-quality token increases substantially.

The problem compounds because XTC users typically increase the exclusion probability to get "more creative" output. But more exclusion means more probability mass dumped into the garbage tail.

**Practical Issues:**

Users report that XTC produces unpredictable results. Some generations are excellent—the forced alternative selection leads to interesting choices. Others are incoherent—garbage tokens were selected due to accumulated tail probability.

The RNG dependence also means that the same prompt with the same settings can produce wildly different quality outputs. Users describe "never knowing what you were going to get."

> **Graph [G-4]: XTC Redistribution Failure**  
> *Demonstrate uniform redistribution spreading probability to low-quality tail tokens. Show the fat tail problem. Compare to Adaptive-P's selective redistribution which doesn't have this issue.*

<!-- TODO: G-4 data (XTC fat tail)
  @claude: May need simulated data or theoretical calculation.
  @loxifi: 
-->

## 2.6 Mirostat

Mirostat was the original inspiration for Adaptive-P. It targets a specific perplexity level by dynamically adjusting Top-K:

**Mechanism:**
1. Compute perplexity of current selection
2. If perplexity is below target, increase K (allow more options)
3. If perplexity is above target, decrease K (restrict options)

**Why It Fails with Modern Models:**

Mirostat was designed when model probability distributions were broader. A Top-K adjustment from 50 to 100 meaningfully changed the candidate pool.

Modern models, especially those trained with RLHF, produce much sharper distributions. The top 2-3 tokens often hold 90%+ of the probability mass. Adjusting Top-K from 50 to 100,000 often selects the same tokens because nothing else has meaningful probability.

The samples in this paper illustrate this reality: most token selections have only 2-5 viable candidates after min-p filtering. Top-K adjustment cannot create variety that doesn't exist in the distribution.

Adaptive-P addresses this by operating on probabilities directly, not on rank. It can boost a 0.1 probability token's chances relative to a 0.8 probability token, which Top-K adjustment cannot accomplish.

<!-- TODO: Add Mirostat discussion quote from chat logs
  @claude: Key discussion around showcase log lines 200-400.
  @loxifi: 
-->

## 2.7 Why Adaptive-P is Different: Selective Redistribution

The key differentiator is **selective redistribution**. When Adaptive-P transforms logits, it applies changes as a function of distance from target probability:

- Tokens near target: receive high logits, compete for selection
- Tokens moderately far: receive lower logits, still contribute
- Tokens very far: receive unboundedly negative logits, effectively excluded

This is fundamentally different from:
- Temperature (uniform scaling—all tokens affected equally)
- Truncation (binary—tokens are in or out)
- XTC (mass redistribution—excluded probability goes everywhere)

**Handling Real Distributions:**

Real token distributions after min-p rarely resemble smooth curves. They typically show:
- A single 100% token (forced choice—no alternatives exist)
- One high token + sparse low cluster (one clear option, some distant alternatives)
- 2-3 mid-range candidates + low cluster with gap (competitive choice with tail)
- Many low-probability tokens with no clear leader (creative opportunity)

Adaptive-P's transformation handles each case appropriately. The unbounded negative logits prevent clustered low tokens from accumulating probability (the XTC failure mode). The quadratic core provides fine differentiation among close competitors.

> **Graph [G-5]: Selective Redistribution on Real Distribution**  
> *Use the "prestigious" sample (22 tokens: one at 0.30, rest at 0.02-0.10). Show how the exponential dropoff prevents clustered low tokens from accumulating probability. Contrast with what uniform redistribution would produce.*

> **Graph [G-6]: Selective vs. Uniform Redistribution Comparison**  
> *Side-by-side: XTC-style uniform redistribution showing fat tail vs. Adaptive-P selective redistribution showing focused distribution around target.*

<!-- TODO: Combine G-5 and G-6?
  @claude: Could be single visualization with two panels.
  @loxifi: 
-->
