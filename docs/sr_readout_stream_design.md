# SR "Read-out Stream" — empty-adapter-stream design (proposal)

A proposed Shadow-Residual variant in which the **adapter stream carries no
activations of its own**. The only thing that ever writes into the adapter
stream is the per-layer cross-stream injection `W_cross(h_base_output)`, carried
forward through an inter-layer RMSNorm. The base stream is the full, frozen
transformer; the adapter stream is a thin, learned, low-rank *read-out* of the
base trajectory.

This document is design-only — no code yet. See
[`sr_readout_stream_design.html`](sr_readout_stream_design.html) for the diagram.

## The recurrence

```
h_adapt_{l+1} = RMSNorm_l( h_adapt_l ) + W_cross_l( h_base_output_l )
h_adapt_0     = embeds * embedding_multiplier          (clone of the base seed)
```

- `h_base_output_l` — the base stream output of layer `l` (after attention and
  MLP residuals), identical to the unadapted base model's layer-`l` output.
- `RMSNorm_l(h_adapt_l)` — the adapter stream's own carry from the layer below,
  renormalized. This **replaces** the raw running sum: the adapter stream is a
  sequence of unit-scale states, not an ever-growing residual.
- `W_cross_l(·)` — that layer's rank-R base→adapter injection (`B·A`), the sole
  trainable path.
- Head: `logits = lm_head( norm( h_adapt_N ) ) / logits_scaling` — only the
  adapter stream reaches the head, exactly as in current SR.

## What changed vs. current SR

| | Current SR (shared-base-KV) | Read-out stream (this proposal) |
| --- | --- | --- |
| Adapter attention (Q/O LoRA) | yes, delta fires | **none** — adapter never attends |
| Adapter MLP (gate/up/down LoRA) | yes, delta fires | **none** |
| Adapter residual | `h_adapt += o_adapt + mlp_adapt + W_cross(h_base)` | `h_adapt = RMSNorm(h_adapt) + W_cross(h_base_out)` |
| Trainable params / layer | cross_stream + optional attn/MLP LoRA | cross_stream `B·A` + one RMSNorm γ |
| Base stream | frozen full transformer | frozen full transformer (unchanged) |
| Cross-stream reads | `h_base` (post-MLP) | `h_base_output_l` (post-MLP) — same site |

The base stream, shared-base-K/V, and the frozen-base invariant are all
**unchanged**. The only structural difference lives on the adapter side: it stops
running its own attention/MLP and becomes a pure accumulator of `W_cross`
injections with an RMSNorm between layers.

## Why this shape

- **The adapter can only mix base information *across layers*, never *across
  positions* on its own.** With no attention on the adapter stream, token-to-token
  mixing happens entirely inside the frozen base; the adapter stacks per-position
  low-rank read-outs of successive base layers. This is the defining constraint of
  the design — a deliberate bet that a learned linear read-out of the base's
  layerwise trajectory is enough for the target tasks.
- **RMSNorm bounds the stream.** Because the carry is renormalized every layer,
  the adapter state stays unit-scale regardless of how many injections have
  accumulated or how large `h_base_output` grows in deep layers. The trade-off:
  early-layer injections are repeatedly renormalized as later ones stack on, so
  the stream is biased toward *recent* (deeper-layer) base information.
- **Tiny trainable footprint.** Per layer: `cross_stream.lora_A` `[R,H]` +
  `cross_stream.lora_B` `[H,R]` + one RMSNorm `γ` `[H]`. No Q/K/V/O/gate/up/down
  LoRA at all.

## Open design points (not yet decided)

- **Is the adapter residual non-residual on purpose?** The recurrence as written
  renormalizes the carry (`RMSNorm(h_adapt_l)`) rather than adding to a raw sum.
  An alternative keeps a true residual: `h_adapt_{l+1} = h_adapt_l +
  RMSNorm(h_adapt_l)·? + W_cross(...)`. The current spec is the non-residual
  (renormalized-carry) form.
- **RMSNorm parameterization.** One `γ` per layer (as drawn) vs. a single shared
  `γ`. Per-layer is the default.
- **PEFT expressibility.** `W_cross` is already a stock-LoRA-wrapped frozen
  zero-init `nn.Linear` (`CrossStream`), so it needs nothing new. The per-layer
  **RMSNorm γ is a new trainable parameter** that is *not* a LoRA delta — it does
  not fit the stock-PEFT "adapter = LoRA on target_modules" mould and would need
  either (a) a `modules_to_save` entry, or (b) a small custom module. This is the
  main packaging question the implementation will have to answer.
