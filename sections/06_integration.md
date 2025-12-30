# 5. Integration and Sampler Chain

This section provides practical guidance for integrating Adaptive-P into existing LLM inference pipelines.

## 5.1 Chain Positioning

**Critical requirement:** Adaptive-P must be the **last** sampler in the chain.

Adaptive-P performs the final token selection. It:
1. Reads the current probability distribution
2. Transforms logits based on distance from target
3. Applies softmax
4. Samples and selects a token
5. Updates history state

If other samplers follow Adaptive-P, they would either:
- Override its selection (making Adaptive-P pointless)
- Operate on already-transformed logits (producing undefined behavior)

**Recommended minimal chain:**

```
min_p → adaptive_p
```

Min-P removes garbage tokens. Adaptive-P selects from remaining candidates.

**Extended chain:**

```
top_k → min_p → temperature → adaptive_p
```

This chain:
1. Top-K: Optional initial truncation for efficiency
2. Min-P: Quality guardrail—removes garbage
3. Temperature: Optional distribution shaping (mild values only)
4. Adaptive-P: Final selection with probability targeting

## 5.2 Why Min-P Complements Adaptive-P

Min-P and Adaptive-P serve different, complementary purposes:

| Aspect | Min-P | Adaptive-P |
|--------|-------|------------|
| Purpose | Remove garbage | Select among quality |
| Method | Truncation | Preference |
| Output | Filtered candidates | Single selection |

**The samples.log perspective:**

All samples show tokens already filtered to p > 0.01. This filtering is min-p at work. Without it, the candidate pool would include thousands of garbage tokens at vanishingly small probabilities.

Adaptive-P assumes this cleanup has happened. If garbage tokens remain in the pool, and they happen to be at a probability close to target (after normalization), they could be selected. Min-P prevents this edge case.

**Recommended min_p values:**

| Use Case | min_p Value |
|----------|-------------|
| General text | 0.05 |
| Creative writing | 0.03 |
| Code generation | 0.1 |

> **Graph [G-16]: Combined Pipeline Effect**  
> *Full vocabulary → min-p filter → Adaptive-P transform → final selection. Show probability mass at each stage.*

## 5.3 Temperature Interaction

Temperature can be used before Adaptive-P, but with caveats:

**Mild temperature (0.8–1.2):** Generally fine. The probability adjustments are modest enough that Adaptive-P can still target effectively.

**Extreme temperature (< 0.5 or > 1.5):** Can interfere with targeting. Very low temperature creates extreme probability peaks that Adaptive-P may struggle to reshape. Very high temperature flattens distributions enough that "mid-range" becomes ambiguous.

**Temperature vs. target adjustment:**

Users often want "more creative" or "less creative" output. Both temperature and target can achieve this:

- Temperature 1.3: Flattens distribution, increases entropy
- Target 0.3: Prefers lower-probability tokens directly

The difference is controllability. Temperature affects the whole distribution uniformly. Target specifies exactly which probability range to prefer. For controlled creativity adjustment, target is the more precise tool.

**Recommendation:**

Keep temperature at 1.0 (neutral) and use target for creativity control. If temperature adjustment is desired, keep it mild (0.9–1.1) and let Adaptive-P handle the primary distribution shaping.

## 5.4 Interactions with Other Samplers

**Samplers that work well before Adaptive-P:**
- Min-P: Recommended as complementary guardrail
- Top-K: Fine for efficiency, but often unnecessary with min-p
- Top-P: Works, but somewhat redundant with Adaptive-P's targeting
- Temperature: Works if mild

**Samplers that Adaptive-P makes unnecessary:**

- **DRY / Repetition Penalty:** Adaptive-P breaks repetition chains by design. When high-probability tokens are selected repeatedly, the adaptive mechanism shifts target downward, making alternatives more attractive. External repetition penalty becomes redundant.

- **XTC:** Adaptive-P achieves XTC's goal (forceβconsideration of alternatives) more reliably and without the fat-tail redistribution problem. Users who previously relied on XTC typically disable it when using Adaptive-P.

- **Mirostat:** Both target perplexity/entropy, but through different mechanisms. Running both would create conflicting control loops. Use one or the other.

## 5.5 Implementation in llama.cpp

Adaptive-P is integrated into the llama.cpp sampler infrastructure. Basic usage:

```cpp
// Create sampler chain
struct llama_sampler * chain = llama_sampler_chain_init(params);

// Add min-p guardrail
llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05, 1));

// Add adaptive-p as final sampler
llama_sampler_chain_add(chain, 
    llama_sampler_init_adaptive_p(
        0.5f,    // target
        0.9f,    // decay  
        seed     // RNG seed
    ));
```

**Parameter configuration:**

- `target`: Desired average probability (0.0–1.0, negative to disable)
- `decay`: History decay rate (0.0–0.99)
- `seed`: Random number generator seed

**Command-line usage (example with llama-cli):**

```bash
./llama-cli -m model.gguf \
    --min-p 0.05 \
    --adaptive-p-target 0.5 \
    --adaptive-p-decay 0.9 \
    -p "Once upon a time"
```

<!-- TODO: Verify CLI parameter names
  @claude: Names depend on llama.cpp PR. Verify match.
  @loxifi: 
-->

## 5.6 Integration Challenges

**UI integration:**

Some front-ends (like SillyTavern) have fixed sampler chain configurations. Integrating Adaptive-P may require:
- Editing configuration files to add Adaptive-P to the chain
- Ensuring it appears last in the chain order
- Adding UI controls for target and decay parameters

**Sampler ordering:**

Different frameworks handle sampler ordering differently:
- Some allow user specification
- Some use fixed internal order
- Some expose limited reordering

If Adaptive-P cannot be placed last, it may not function correctly. Check framework documentation for sampler chain control.

**Debugging:**

If Adaptive-P appears to have no effect:
1. Verify it's last in the chain (another sampler may be overriding selections)
2. Check target value (negative values disable the sampler)
3. Confirm min-p or similar is removing garbage tokens
4. Review that the framework correctly passes the sampler output
