# 1. Introduction

## 1.1 The Problem: High-Confidence Token Chains

Modern large language models produce remarkably fluent text, but this fluency often comes at the cost of creativity and variety. The underlying issue is what we term "high-confidence token chains"—sequences where each token's selection reinforces the next high-probability choice, creating a self-perpetuating cycle of predictable output.

Consider a model continuing the phrase "The quick brown fox..." The most probable next token is almost certainly "jumps." Once "jumps" is selected, "over" becomes extremely likely. Then "the," then "lazy," then "dog." The model has locked onto a memorized sequence, and nothing in standard sampling breaks this chain.

This behavior manifests in practical terms as:
- **Repetitive phrasing**: The same sentence structures and word choices appearing across generations
- **Generic descriptions**: "The room was dimly lit" instead of more specific, evocative language  
- **Predictable narratives**: Story beats following the most common patterns from training data
- **Slop**: The community term for output that feels AI-generated—technically correct but lacking authentic voice

The root cause isn't the model's knowledge—modern LLMs encode vast linguistic diversity. The problem is that standard sampling methods don't provide a mechanism to *access* that diversity in a controlled way.

## 1.2 Why Existing Approaches Fall Short

Current sampling methods fall into two broad categories, neither of which solves the high-confidence chain problem:

**Scaling approaches** (temperature) multiply all logits by a constant factor. This uniformly spreads probability mass but provides no way to *target* a specific probability range. High temperature makes everything more random, including garbage tokens. Low temperature makes everything more deterministic, reinforcing the chain problem.

**Truncation approaches** (top-k, top-p, min-p) remove tokens below some threshold and renormalize. This is a binary decision—tokens are either in the candidate pool or excluded entirely. Within the remaining pool, there's no preference mechanism. If the top token has 0.6 probability after truncation, it will be selected roughly 60% of the time regardless of whether the user wants more varied output.

Neither approach asks the question we actually want to answer: "Can we preferentially select tokens at a specific probability level?"

**Example:** Consider three tokens with probabilities 0.7, 0.2, and 0.1:
- **Temperature 1.5:** Flattens to roughly 0.5, 0.3, 0.2. The top token still dominates; low tokens get a boost but remain minority choices.
- **Top-P 0.9:** Keeps all three (cumulative 1.0 > 0.9). No preference within the set—the 0.7 token is still selected 70% of the time.
- **Min-P 0.15:** Keeps the 0.7 and 0.2 tokens. Again, no preference—just truncation.

None of these methods can say "prefer the 0.2 token over the 0.7 token."

## 1.3 The Adaptive-P Solution

Adaptive-P takes a fundamentally different approach: rather than scaling or truncating, it *reshapes* the probability distribution to favor tokens near a target probability.

The core insight is that probability targeting can be framed as a distance metric. For each token, we compute how far its probability lies from the target, then transform its logit based on that distance. Tokens near the target receive high logits; distant tokens are suppressed.

This creates several desirable properties:

1. **Controlled creativity**: A target of 0.5 means the sampler prefers tokens that the model considers "plausible but not dominant." A target of 0.3 encourages more surprising choices. A target of 0.7 stays closer to the model's top predictions.

2. **Adaptive adjustment**: The sampler tracks which probabilities have been selected recently. If recent selections skewed high, it compensates by targeting lower on the next step. This prevents both stuck-in-a-rut determinism and chaotic randomness.

3. **Chain breaking**: High-confidence chains are disrupted because the sampler actively resists selecting 0.9+ probability tokens repeatedly. The first high-confidence token in a potential chain shifts the target downward, making alternatives more attractive for subsequent tokens.

4. **Consistent behavior**: Unlike temperature (where the effect depends heavily on the input distribution's shape), the same target parameter produces similar selection patterns across different models and contexts.