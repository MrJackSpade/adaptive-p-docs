# 4. Parameters

Adaptive-P exposes two user-configurable parameters. This section details their effects, recommended ranges, and interaction with each other.

## 4.1 Target

**Range:** 0.0 to 1.0 (negative values disable the sampler)

**Default:** 0.5

**Effect:**

Target specifies the desired average probability of selected tokens. The sampler adjusts its per-step targeting to achieve this average over time.

| Target | Behavior | Use Case |
|--------|----------|----------|
| 0.3 | Prefers surprising, lower-probability tokens | Creative writing, brainstorming |
| 0.5 | Balanced—mid-range tokens preferred | General purpose |
| 0.7 | Stays closer to model's top predictions | Factual content, coherence-critical |
| 0.9+ | Nearly deterministic | When accuracy matters most |

### Generation Showcase

The following examples show the same prompt with different target values. Notice how lower targets produce more varied, surprising word choices while higher targets stay closer to conventional phrasing.

**Prompt:** *"Write me a three paragraph horror story about a haunted bath-house written in the first person"*

**[Target 0.3 (Creative)](../samples/target_0.3_sample.md)**
> *"My blood turns to icy slurry as a cold draft, utterly impossible in the humid heat, sweeps over the nape of my neck."*

**[Target 0.5 (Balanced)](../samples/target_0.5_sample.md)**
> *"The whispers started again, not from the pipes this time, but from the empty cubicles beside me, low and wet and sounding like voices drowning in deep water."*

**[Target 0.7 (Conservative)](../samples/target_0.7_sample.md)**
> *"A cold, bony touch sliding up my calf, leaving a trail of goosebites that felt like frost."*

**[Target 0.9 (Near-deterministic)](../samples/target_0.9_sample.md)**
> *"The water wasn't just haunted; it was alive, and it had decided I was its next meal."*

Notice how target 0.3 produces unusual imagery ("icy slurry"), while 0.9 gravitates toward polished, conventional phrasing. Target 0.5 balances creativity with coherence; 0.7 executes familiar horror tropes cleanly.

**Key property: Cross-model consistency**

Unlike temperature, which produces different effects depending on input distribution shape, target behaves consistently across models. Target 0.5 on Llama produces similar *selection patterns* to target 0.5 on Mistral, even though the underlying token distributions differ.

This consistency arises because the targeting is defined in probability space, not logit space. A token at 0.5 probability means the same thing regardless of how the model arrived at that value.

**Intuition for parameter tuning:**

Users report that the target parameter feels intuitive once understood. "I want the model to pick tokens it's about 40% confident in" translates directly to target 0.4. This contrasts with temperature where "temperature 1.2" has no obvious semantic meaning.

## 4.2 Decay

**Range:** 0.0 to 0.99

**Default:** 0.9

**Effect:**

Decay controls how much historical selections influence the current calculated target. Higher decay means past selections have longer-lasting effect; lower decay makes the sampler more responsive to recent selections.

The decay value can be interpreted as "how many tokens back significantly influence the current step":

| Decay | Effective History Window |
|-------|-------------------------|
| 0.5 | ~2 tokens |
| 0.7 | ~3 tokens |
| 0.9 | ~10 tokens |
| 0.99 | ~100 tokens |

**Mathematical interpretation:**

The weight of a selection N steps ago is `decay^N`. At decay 0.9, a selection 10 steps ago has weight 0.9^10 ≈ 0.35 of the most recent selection. At decay 0.5, it has weight 0.5^10 ≈ 0.001—effectively forgotten.

**Effect on distribution shape:**

Higher decay produces sharper selection curves. With decay 0.99, the sampler tightly clusters selections around target because any deviation is strongly compensated. With decay 0.5, the sampler allows more per-step variance because it quickly "forgets" previous selections.

<table>
<tr>
<td width="50%"><img src="../charts/target_long.png" width="100%"><br><em>Target effect: Lower targets peak earlier; higher targets stay closer to natural predictions.</em></td>
<td width="50%"><img src="../charts/decay_long.png" width="100%"><br><em>Decay effect: Higher decay produces sharper peaks around target.</em></td>
</tr>
</table>

**Interaction with target:**

Decay doesn't change *where* the sampler targets—that's controlled by target. Decay changes *how tightly* it maintains that target.

- High target + high decay: Consistently conservative selections
- High target + low decay: Variable, sometimes conservative
- Low target + high decay: Consistently creative selections
- Low target + low decay: Variable, sometimes creative

Most users keep decay at default (0.9) and adjust only target.

**Why decay is configurable:**

Although 0.9 works well for most cases, edge cases require adjustment:

- **Stubbornness (decay too high):** With decay 0.99, the sampler "remembers" history for ~100 tokens. If the first 50 tokens happened to be high-probability forced choices, the sampler stubbornly tries to compensate by targeting very low probabilities for the next 50—even when that's not ideal for the current context.

- **Fishtailing (decay too low):** With decay 0.5, the sampler reacts to only the last 2-3 tokens. After one high-probability selection, it swings hard toward low probability; after one low selection, it swings back. The output "fishtails" between extremes rather than finding a stable average.

- **Elasticity:** The ideal decay provides enough elasticity to return toward target without overcorrecting. Like a spring—too stiff and it fights every movement; too loose and it oscillates wildly.

## 4.3 Internal Constants

The following values are fixed in the implementation and not user-configurable:

| Constant | Value | Purpose |
|----------|-------|---------|
| PEAK_LOGIT_VALUE | 5.0 | Maximum logit for on-target tokens |
| SHARPNESS | 10.0 | Controls transformation curve steepness |
| DISTRIBUTION_WIDTH | 0.2 | Probability distance scaling factor |

**PEAK_LOGIT_VALUE (5.0):**

Sets the logit for tokens exactly at the target probability. The specific value is less important than the *difference* between peak and suppressed logits, since softmax cares only about relative values.

**SHARPNESS (10.0):**

Controls how quickly logits drop as tokens move away from target. Higher values create a sharper peak with more aggressive suppression of distant tokens. Lower values create a broader preference band.

This value was tuned empirically across multiple models and use cases. The value 10.0 provides a good balance between:
- Sufficient discrimination (tokens at different distances receive meaningfully different logits)
- Not over-aggressive (moderately distant tokens still have non-trivial selection probability)

**DISTRIBUTION_WIDTH (0.2):**

Scales the distance calculation. Effectively defines "nearby" in probability space. At width 0.2, a token at distance 0.2 from target (one width unit away) receives notably suppressed logit.

The inverse (INV_WIDTH = 5.0) appears in the transformation to avoid runtime division.

**Why SHARPNESS is not user-configurable:**

Early development considered exposing SHARPNESS as a parameter. The decision to fix it came from two observations:

1. **Confusing interaction with target:** Users expect "higher sharpness = stronger effect" but the relationship is non-linear. SHARPNESS interacts with DISTRIBUTION_WIDTH in ways that aren't intuitive—adjusting one without the other produces unexpected results.

2. **Decay provides the needed control:** What users actually want when they'd reach for sharpness—"tighter" or "looser" adherence to target—is better achieved through decay, which has a more predictable effect.

For advanced use cases, the constants can be modified in source.

## 4.4 Disabling Adaptive-P

Setting target to a negative value (conventionally -1.0) disables the sampler entirely. In disabled mode, Adaptive-P:

1. Applies softmax to current logits
2. Samples from the resulting distribution
3. Does not modify history state

This allows Adaptive-P to remain in the sampler chain without affecting output when disabled. Users can toggle the parameter without reconfiguring the pipeline.

```cpp
if (ctx->target < 0.0f) {
    // Disabled: sample from unmodified distribution
    llama_sampler_softmax_impl(cur_p, false);
    cur_p->selected = llama_sample_dist(cur_p, ctx->rng);
    return;
}
```
