# Shadow-Residual Architecture — Internal Explainer

An internal walkthrough of the SR module layout, on-disk files, PEFT integration (save/load), invocation flows, and what is actually trainable.

## Contents

1. [Architecture — modules and on-disk layout](#1-architecture--modules-and-on-disk-layout)
2. [Relation to PEFT — save / load / why standard PEFT is not enough](#2-relation-to-peft--save--load--why-standard-peft-is-not-enough)
3. [Invocation flows — ALORA gating and the activation sequence](#3-invocation-flows--alora-gating-and-the-activation-sequence)
4. [What gets trained — frozen vs. active](#4-what-gets-trained--frozen-vs-active)

**Legend**

- 🔵 base stream / frozen
- 🔴 adapter stream / trainable LoRA
- 🟡 cross-stream injection (B·A)
- ⚪ frozen weight
- 🟢 trainable weight

---

## 1. Architecture — modules and on-disk layout

Shadow-residual (SR) is a per-layer compute graph that maintains **two parallel hidden-state streams** through every attention/MLP block: a frozen *base stream* bit-identical to the unadapted base model (modulo a documented K/V exception), and a trainable *adapter stream* that carries LoRA deltas plus a per-layer rank-R cross-stream injection from the base. There is no final-step merge: after the last layer the adapter stream is normed and projected directly through `lm_head`.

SR lives entirely under `src/granite_switch/experimental/` — nothing in the open-source `granite-switch/` submodule changes. There are two top-level subpackages: `shadow_residual/` (the model) and `peft_shadow_residual/` (the PEFT integration). Training drivers live under `training/`.

### 1.1 Module ↔ file map

```
src/granite_switch/experimental/
├── shadow_residual/                       - the SR model itself (HF backend)
│   ├── modeling_hf.py                    ShadowResidualModel, ShadowResidualForCausalLM
│   ├── decoder_hf.py                     ShadowResidualDecoderLayer (dual-stream + early-exit)
│   ├── attention_hf.py                   ShadowResidualAttention (disjoint K/V caches)
│   ├── cross_stream.py                   CrossStream — no-op nn.Module site
│   ├── _stream_gated_linear.py           _StreamGatedLinear (nn.Linear marker subclass)
│   ├── _stream_context.py                stream_context() / current_stream() (contextvar)
│   ├── weight_transfer.py                fused->unfused QKV / gate-up slicing
│   └── config_helpers.py                 set_shadow_residual / validate_shadow_residual_config
│
├── peft_shadow_residual/                  - PEFT integration
│   ├── factory.py                        get_shadow_residual_peft_model + _build_sr_base
│   ├── load.py                           load_shadow_residual_peft_model
│   ├── cross_stream_lora.py              CrossStreamLora — LoraLayer wrapping a CrossStream site
│   └── stream_gated_lora.py              ShadowResidualLora — PeftLoraLinear with stream-gate
│
└── training/                              - HF Trainer drivers + collators
    ├── sr_train.py                       SR-specific entry point (Deprecated, to be removed and only here for reference)
    ├── train.py                          unified YAML-driven training entry
    ├── collator.py                       DataCollatorForSingleExpert{,Packed}
    ├── add_labels.py                     stamp 'labels' column from activation_sequence
    └── pack_stats.py                     pad-free packing analyzer
```

### 1.2 The class hierarchy — what inherits from what

```
GraniteMoeHybridPreTrainedModel (transformers, upstream)
            │ (inheritance)
            ▼
ShadowResidualPreTrainedModel
   config_class = GraniteSwitchConfig
            │
            ▼
ShadowResidualModel ────────────────► ShadowResidualDecoderLayer
   embed + layers[N] + norm                self_attn + gate/up/down + cross_stream
            │                                       │
            ▼                                       ├──► ShadowResidualAttention
ShadowResidualForCausalLM                          │       q_proj k_proj v_proj o_proj
   + lm_head, GenerationMixin                      │       (StreamGated)
                                                    │
                                                    ├──► CrossStream  (no-op site)
                                                    │       replaced by CrossStreamLora at PEFT-time
                                                    │
                                                    └──► _StreamGatedLinear
                                                            marker subclass of nn.Linear
```

Inheritance (vertical) and containment (horizontal). SR inherits directly from `GraniteMoeHybridPreTrainedModel` — no GraniteSwitch detour. The decoder layer holds attention, MLP projections (all `_StreamGatedLinear`), and a no-op `CrossStream` site that PEFT will later wrap.

### 1.3 Per-layer compute graph (dual-stream, cross-stream active)

```
   base stream — h_base (frozen Q/O/MLP)        adapter stream — h_adapt (trainable LoRA)
   ─────────────────────────────────────        ─────────────────────────────────────────
       h_base (from prev layer)                       h_adapt (from prev layer)
                       │                                            │
                       └────────────┬───────────────────────────────┘
                                    ▼
                  input_layernorm( cat[h_base, h_adapt] )
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   q/k/v/o_proj  (base ctx)                  q/k/v/o_proj  (adapter ctx)
   delta gated OFF; writes pkv_base          LoRA delta fires; writes past_key_values
            │                                               │
            ▼                                               ▼
   h_base += o_base * μ_res                  h_adapt += o_adapt * μ_res
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                  post_attention_layernorm( cat[...] )
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   gate/up/down  (base ctx)                  gate/up/down  (adapter ctx)
   delta gated OFF; bit-exact base MLP       LoRA delta fires
            │                                               │
            │                                               ▼
            └─── cross_stream(h_base) ──────► h_adapt += cross_stream(h_base)  [B·A]
```

One decoder layer in dual-stream mode. Layernorms run on a concatenated tensor for fused-batch efficiency, then the two streams diverge under `stream_context("base")` vs `stream_context("adapter")`. K/V caches are **disjoint**: the base stream owns `past_key_values_base` (frozen, adapter-free); the adapter stream owns the user's `past_key_values`. The end-of-layer `cross_stream(h_base)` is the only base→adapter information channel.

### 1.4 Inter-layer wiring — how stream *i* connects to stream *i+1*

The previous diagram zoomed into one layer. This one zooms *out*: the embedding feeds layer 0, each layer hands two streams to the next, and the final layer hands the adapter stream to the head. A few things become important only at this scale:

- **Streams are per-layer-pair connected.** Layer *i*'s `h_base` output feeds layer *i+1*'s `h_base` input — nothing else. Same for `h_adapt`. There is no skip from layer *i*'s adapter into layer *j*'s base (that would break the frozen-base invariant), and no skip from layer *i*'s base into layer *j*'s adapter for any *j ≠ i*.
- **Cross-stream is strictly local.** The *only* base→adapter coupling lives at the bottom of each layer: `h_adapt += cross_stream(h_base)`. It writes into *this layer's* `h_adapt` output, which is what the next layer's adapter input consumes. Information from the base stream therefore reaches the adapter stream *one layer later*, recursively.
- **Embedding is shared, not duplicated.** `embed_tokens(input_ids) × embedding_multiplier` runs once. The result *is* `h_base`; `h_adapt` is `h_base.clone()` at layer 0. The two streams diverge only inside the layer stack.
- **The head only sees the adapter stream.** After layer *N−1*, `h_base` is discarded; `norm(h_adapt)` goes through `lm_head`. The base stream's purpose is purely to feed `cross_stream(·)` at every layer; it is never read at the head.
- **K/V caches are also per-layer-paired.** Layer *i*'s base K/V live in slot *i* of `past_key_values_base`; layer *i*'s adapter K/V live in slot *i* of `past_key_values`. The two caches share *indices* but never *tensors*; one layer's base K/V never becomes another layer's adapter K/V.

```
                       embed_tokens(input_ids)
                       e *= embedding_multiplier
                       h_base = e ;  h_adapt = e.clone()
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
            h_base lane                 h_adapt lane
       (frozen, blue)                 (trainable, red)
                │                           │
   ┌────────────┴───────────────────────────┴────────────┐
   │ layer 0 — ShadowResidualDecoderLayer                │
   │   writes pkv_base[0] / pkv_adapt[0]                 │
   │   input_layernorm(cat[h_base, h_adapt])             │
   │   attn(base): Δ OFF; pkv_base[0]                    │
   │   attn(adapter): +LoRA Δ; pkv_adapt[0]              │
   │   +residual (both lanes)                            │
   │   post_attention_layernorm(cat[...])                │
   │   MLP(base): Δ OFF                                  │
   │   MLP(adapter): +LoRA Δ                             │
   │   +residual (both lanes)                            │
   │   h_adapt += cross_stream(h_base)  [B·A], R×H       │
   └────────────┬───────────────────────────┬────────────┘
                ▼                           ▼
   ┌────────────┴───────────────────────────┴────────────┐
   │ layer 1 — same wiring as layer 0                    │
   │   writes pkv_base[1] / pkv_adapt[1]                 │
   └────────────┬───────────────────────────┬────────────┘
                ▼                           ▼
                      ... layers 2 .. N-2 ...
   each layer writes its own KV slot in BOTH caches:
     pkv_base[i] (frozen, adapter-free) and pkv_adapt[i]
   layer i's h_base only feeds layer i+1's h_base
   layer i's h_adapt only feeds layer i+1's h_adapt
                │                           │
                ▼                           ▼
   ┌────────────┴───────────────────────────┴────────────┐
   │ layer N-1 (last layer)                              │
   │   writes pkv_base[N-1] / pkv_adapt[N-1]             │
   │   last per-layer base→adapter merge                 │
   └────────────┬───────────────────────────┬────────────┘
                ▼                           ▼
       h_base out (last layer)         norm(h_adapt)
       DISCARDED — not used by         lm_head( ... )
       lm_head                          / logits_scaling
                                       → logits
```

Vertical inter-layer view. The two lanes run the full height of the figure; each layer is one card with all of its intermediates (input LN → per-stream attn → +residual → post-attn LN → per-stream MLP → +residual → cross-stream merge) stacked top-to-bottom. The only base→adapter coupling is the per-layer cross-stream merge at the bottom of each layer; the lanes are otherwise parallel and never crossed across layers. After the last layer, `h_base` is dropped and only `norm(h_adapt)` reaches `lm_head`.

### 1.5 Adapter-only early-exit

When `"cross_stream"` is *not* in `target_modules`, no `CrossStream` site has been wrapped — the model gate `_any_cross_stream_wrapped(self.layers)` in `modeling_hf.py` returns `False` and the decoder takes a **single-stream path**: only `h_adapt` is computed, with one Q/K/V/O pass per layer under the default `"adapter"` context. Compute is equivalent to plain LoRA on Granite (modulo unfused QKV / gate-up reduction order).

```
   adapter-only early-exit (no cross_stream wrapped)

   h_adapt  ─►  attn (adapter ctx)        ─►  MLP (adapter ctx)            ─►  h_adapt
   (input)      single Q/K/V/O                LoRA on gate/up/down              (output)
                single KV cache               (if listed)

   No h_base maintained, no cross_stream call, no second attention pass.
   cross_stream sites stay unwrapped CrossStream no-ops; the model gate skips dual-stream entirely.
```

### 1.6 Shared base K/V (`shared_base_kv=True`)

The dual-stream graph in §1.3 has each stream compute its *own* K and V (the **disjoint-K/V** default, `config.shared_base_kv=False`): two K/V projections per layer, two physically separate caches (`past_key_values` for the adapter, `past_key_values_base` for the base). Shared-base-K/V mode (`config.shared_base_kv=True`, set via `set_shadow_residual(..., shared_base_kv=True)`) collapses that to a **single K/V, computed once from the base stream**, that both streams attend against. It is selected at runtime in `attention_hf.py` (the `if self.shared_base_kv:` branch) and is the dominant production variant — every `sr-*-sharedkv.yaml` config trains it.

What changes versus the disjoint default:

- **K and V are computed once, under `stream_context("base")`** — `k = k_proj(normed_base)`, `v = v_proj(normed_base)`. Because the call runs in the `"base"` context, any `ShadowResidualLora` wrapper short-circuits to `base_layer(x)`, so the cached K/V is `W_K·normed_base` / `W_V·normed_base` with **frozen weights only**. The cache carries *zero* adapter influence.
- **The adapter stream computes only Q** — `q_adapt = q_proj(normed_adapt)` under `"adapter"` context (Q LoRA delta fires). It does *not* compute its own K/V; `_shape_qkv` is called with `k=None, v=None` for the adapter.
- **Both Qs attend against the same shared K/V.** Per head, the adapter stream is `softmax(Q_adapt · K_baseᵀ / √d) · V_base`, then `o_proj` under `"adapter"` context (O LoRA delta fires).
- **Only one cache is used.** The shared K/V is written into `past_key_values` (the adapter cache); `past_key_values_base` is intentionally ignored in this mode.
- **K/V LoRA is structurally forbidden.** `_reject_kv_lora_when_shared` (`peft_shadow_residual/factory.py`) raises at construction time if `k_proj`/`v_proj` appear in `target_modules` — there is no place for a K/V adapter delta to land, so the combination is rejected rather than silently dropped.

```
   shared base K/V (shared_base_kv=True, cross_stream active)

   normed_base  ─►  q_proj (base ctx)   ─┐
                    k_proj (base ctx)    │   K/V = W_kv · normed_base   (FROZEN, no delta)
                    v_proj (base ctx)   ─┤   written ONCE to past_key_values
                                         │
   normed_adapt ─►  q_proj (adapter ctx) │   Q_adapt = (W_Q + ΔW_Q) · normed_adapt
                    (Q LoRA Δ fires)     │
                                         ▼
        Q_base   ──► softmax(Q_base · Kᵀ)·V  ──► o_proj (base ctx)    ─► o_base
        Q_adapt  ──► softmax(Q_adapt· Kᵀ)·V  ──► o_proj (adapter ctx) ─► o_adapt
                                         ▲
                       both Qs attend the SAME frozen K/V; one cache only.
                       past_key_values_base unused. K/V LoRA forbidden.
```

The adapter reshapes only the *query* and the *output projection*; it reads the same frozen K/V the unadapted base model would have written. The base stream's Q/K/V/O/MLP stay bit-identical to the unadapted base model, exactly as in the disjoint default. The architectural payoff: because the cache is adapter-independent, an SR adapter can attend over a KV cache populated by the frozen base — keys/values need not be recomputed when the adapter turns on.

### 1.7 The fused→unfused weight transfer

Upstream `GraniteMoeHybridForCausalLM` stores attention as a *fused* `self_attn.qkv_proj` and the MLP gate/up as a fused `shared_mlp.input_linear`. SR uses *unfused* per-projection tensors so PEFT can attach a separate LoRA per projection. `weight_transfer.transfer_base_weights(src, dst)` walks the source state-dict and slices each fused tensor into the corresponding unfused destination keys:

```
upstream                                                  SR (unfused)
self_attn.qkv_proj.weight  [Q+K+V, H]   →                 self_attn.q_proj.weight  [Q, H]
                                                          self_attn.k_proj.weight  [K, H]
                                                          self_attn.v_proj.weight  [V, H]
shared_mlp.input_linear.weight  [2I, H]  →                gate_proj.weight  [I, H]
                                                          up_proj.weight    [I, H]
shared_mlp.output_linear.weight  →                        down_proj.weight
```

The slicing is done with `torch.no_grad().copy_` into already-allocated destination tensors, so shapes must match exactly — mismatch is a hard fail. Numerics are not bit-exact with upstream because the reduction order changes (fused vs three separate matmuls); this is a documented divergence.

---

## 2. Relation to PEFT — save / load / why standard PEFT is not enough

### 2.1 The two custom-module wrappers

PEFT's `LoraModel` walks the model graph and, for each module whose attribute name matches `target_modules`, replaces it with a *LoRA wrapper*. Out of the box PEFT only knows how to wrap `nn.Linear` (and a handful of others). SR has two needs that vanilla PEFT cannot meet:

1. The **cross-stream site** is not a linear — it is a no-op `nn.Module` that has no weight at all. Standard PEFT cannot wrap it because there is no `weight` to read shapes from.
2. The **Q/K/V/O/gate/up/down projections** need a per-call gate (the *stream gate*) that toggles the LoRA delta on/off depending on which stream the decoder is computing right now. Stock `peft.tuners.lora.layer.Linear` always applies the delta.

We solve both via PEFT's `LoraConfig._register_custom_module({type: wrapper_class})` mechanism. Two registrations, both done by `_register_cross_stream(lora_config)` in `peft_shadow_residual/factory.py`:

| Target type            | Wrapper class          | What it adds                                                                                                                                                                      |
| ---------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CrossStream`          | `CrossStreamLora`      | Implements `LoraLayer` manually. Owns rank-R `lora_A`/`lora_B` ModuleDicts. `forward(x)` returns `0 + B(A(x))*scaling` — the base layer contributes nothing.                       |
| `_StreamGatedLinear`   | `ShadowResidualLora`   | Subclasses stock `peft.tuners.lora.layer.Linear`. Adds one branch at the top of `forward`: if `current_stream() == "base"` → call `base_layer(x)` and return (delta gated off); otherwise delegate to `super().forward(x, ...)`. |

> **Why a marker subclass?** PEFT dispatches `custom_module_mapping` by exact `type(module)`. We want only the *SR decoder's* Q/K/V/O/gate/up/down to get the stream-gated wrapper, not every `nn.Linear` in the model (e.g. the `lm_head`, or linears in unrelated subpackages). The marker subclass `_StreamGatedLinear` — behaviourally identical to `nn.Linear` — is the cleanest selector PEFT supports.

### 2.2 The end-to-end picture

```
1. AutoModelForCausalLM             2. ShadowResidualForCausalLM(cfg)    3. transfer_base_weights(src, dst)
   GraniteMoeHybrid (fused      ─►     empty SR shell with unfused    ─►   slice fused QKV/gate-up → unfused
   QKV/gate-up)                        _StreamGatedLinear projections      no_grad copy_, dtype-preserving
   .from_pretrained(base_id)
                                                                                       │
                                                                                       ▼
4. _register_cross_stream(cfg)      5. get_peft_model(base, cfg)        6. _disable_merge_and_unload
   CrossStream → CrossStreamLora ─►    walks target_modules, swaps   ─►   merge_and_unload would silently fold
   _StreamGatedLinear →                each match to its registered      delta into the frozen-base path,
   ShadowResidualLora                  wrapper class                     breaking the SR invariant.
   via LoraConfig._register_           freezes base, marks lora_A/B
   custom_module                       trainable
                                                                                       │
                                                                                       ▼
7a. Trainer.train() → backward only on lora_A/B          7b. model.save_pretrained(out)
    cross_stream_A, cross_stream_B                            writes out/adapter_model.safetensors
    {q,k,v,o}_proj.lora_A/B (if attn_lora=True)               writes out/adapter_config.json (LoraConfig dump)
    {gate,up,down}_proj.lora_A/B (if mlp_lora=True)           keys e.g. base_model.model.model.layers.0
    all base weights frozen via PEFT freeze pass                .cross_stream.lora_A.default.weight
```

End-to-end build path of `get_shadow_residual_peft_model(base_id, lora_config)`. Steps 1–3 build the SR base; steps 4–6 set up PEFT; step 7 trains and saves.

### 2.3 What happens on save

Saving is just `peft_model.save_pretrained(output_dir)`. PEFT's standard save logic walks every `LoraLayer` instance (which both wrappers ultimately are) and writes the `lora_A`/`lora_B` ModuleDicts to `adapter_model.safetensors`. Because `CrossStreamLora` has the same `self.lora_A[adapter] : nn.Linear(H, r)` / `self.lora_B[adapter] : nn.Linear(r, H)` layout as stock PEFT, the state-dict keys look like ordinary LoRA keys:

```
base_model.model.model.layers.0.cross_stream.lora_A.default.weight   [r, H]
base_model.model.model.layers.0.cross_stream.lora_B.default.weight   [H, r]
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight   [r, H]
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight   [Q, r]
...
```

The `adapter_config.json` is the standard `LoraConfig` dump — it records `target_modules` (including `"cross_stream"` if listed), `r`, `lora_alpha`, `rank_pattern`, and any `alora_invocation_tokens`. There is **no SR-specific sidecar file**; the checkpoint is identifiable as SR only by the presence of `"cross_stream"` in `target_modules`.

### 2.4 What happens on load

The custom-module mapping is *not* persisted in `adapter_config.json` (PEFT does not serialize `_register_custom_module` entries). That is exactly why we cannot use `PeftModel.from_pretrained` directly — without the registration, PEFT would either fail to wrap the `CrossStream` site (no `weight` to read) or wrap our `_StreamGatedLinear` with stock `Linear`, losing the stream gate. The dedicated loader `load_shadow_residual_peft_model(base_id, ckpt)` exists for exactly this reason:

1. Read `LoraConfig.from_pretrained(ckpt)`.
2. Build a fresh SR base via `_build_sr_base(...)` — same fused→unfused slicing.
3. Re-register the custom-module mapping on the freshly-loaded `LoraConfig` — *this step is the entire reason a custom loader exists*.
4. Call `PeftModel.from_pretrained(base, ckpt, config=lora_config)`, passing the registered config explicitly so PEFT does not re-read `adapter_config.json` and lose our dispatch.
5. Call `_disable_merge_and_unload` again — the freshly constructed `PeftModel` would re-enable it.

### 2.5 Why standard PEFT API does not work end-to-end

| You wanted to call                                                                          | What breaks                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_peft_model(AutoModelForCausalLM.from_pretrained(...), LoraConfig(...))`               | The model has no `cross_stream` attribute, no `_StreamGatedLinear` projections, and the upstream model uses *fused* `qkv_proj` / `shared_mlp.input_linear` (so a single LoRA covers all of Q+K+V, not what we want). You'd train a different architecture.                  |
| `PeftModel.from_pretrained(AutoModelFor..., "./out")`                                       | Same as above: there is no `cross_stream` module on the upstream model for PEFT to attach the saved `lora_A`/`lora_B` tensors to — load fails with "module not found".                                                                                                       |
| `get_peft_model(sr_base, LoraConfig(...))` (no registration)                                | PEFT cannot wrap `CrossStream` (no `weight`, would crash on `StopIteration` in `_replace_module`'s parameter probe). Our `_StreamGatedLinear`s would be wrapped with stock `Linear` — no stream gate — and the base stream would no longer be frozen-equivalent.            |
| `peft_model.merge_and_unload()`                                                              | Disabled in SR. Folding the LoRA delta into `base_layer.weight` would silently contaminate the *frozen-base* path: every `stream_context("base")` call would return `(W + ΔW)x` instead of `Wx`. The frozen-base invariant is the whole reason the architecture exists. `_disable_merge_and_unload` patches it to raise `NotImplementedError`. |

> **Bottom line.** The custom factory + custom loader are not architectural luxuries — they are the minimum glue to make PEFT work against a model whose graph contains a non-linear LoRA target (`CrossStream`) and whose linear LoRA targets need a per-call gate. Save and config dump are otherwise vanilla PEFT.

---

## 3. Invocation flows — ALORA gating and the activation sequence

There are three orthogonal axes that determine *which* LoRA delta is applied to *which* positions in *which* stream:

1. **Stream**: `"base"` vs `"adapter"`. Set by `stream_context(...)` in the SR decoder.
2. **ALORA invocation**: when `alora_invocation_tokens` is set on `LoraConfig`, peft installs `ALoraLinearVariant` on every wrapped projection; the LoRA delta is gated to positions *at and after* the rightmost match of the invocation sequence. Pre-invocation positions get the bare base output.
3. **Active adapters**: the current PEFT active adapter set. SR currently trains one adapter at a time (`adapter_name="expert"` by default).

### 3.1 The activation sequence — two distinct uses

The phrase "activation sequence" means two related but separate things in this codebase:

| Use site                                       | What it is                                                                                                                                | Effect                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add_labels.py` — preprocessing                | Token IDs marking the *prompt→completion* boundary in a tokenized dataset (e.g. `[100264, 78191, 100265]` for an assistant header).      | Stamps the `labels` column: `-100` up to and including the rightmost match, real token IDs after — loss is computed only on the completion.                                                                                                                                                                          |
| `LoraConfig(alora_invocation_tokens=[...])`    | Token IDs whose rightmost occurrence in `input_ids` marks where the adapter *turns on* at runtime.                                       | peft computes per-row `alora_offsets` (the trailing length after the match) via a pre-forward hook on every `LoraLayer`; each LoRA delta is masked off for positions before the match. `CrossStreamLora` consumes the same hook input.                                                                                |

The two are usually *related* — you'd typically pick the same boundary token IDs — but they are independent settings: you can label-mask the dataset for SFT loss without using ALORA gating, and vice-versa.

#### 3.1.1 Configuring the activation sequence — the load-bearing knobs

Three settings have to line up before ALORA gating actually fires. Missing any one of them is silent: peft falls back to unconditional LoRA and the model still trains, just with no gating.

| Setting                                          | Where                                                                                          | If missing                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alora_invocation_tokens=[t1, t2, ...]`          | `LoraConfig` (set by `sr_train.py` from `--alora_invocation_token_ids`)                        | peft never installs `ALoraLinearVariant`; every wrapped projection runs as plain LoRA on every position.                                                                                                                                                                                                                                                                                                  |
| `task_type="CAUSAL_LM"`                          | `LoraConfig` (set by `sr_train.py:150`)                                                        | **Load-bearing.** peft prints *"aLoRA is currently only supported for CAUSAL_LM task"* as a warning and skips the variant install. Caught in `tests/experimental/shadow_residual/test_alora_invocation_invariant.py` — without it, the pre-invocation logits-invariant test fails.                                                                                                                         |
| `"cross_stream"` in `target_modules`             | `LoraConfig` (set by `--cross_stream True`, default)                                           | The dual-stream forward never engages; SR runs the single-stream early-exit path. ALORA gating still works, but only in the LoRA-only sense.                                                                                                                                                                                                                                                              |

#### 3.1.2 Where the gating actually lives in code

End-to-end, when `peft_model(input_ids=...)` is called with `alora_invocation_tokens` set:

1. **Pre-forward hook** on every `LoraLayer`: `peft.tuners.lora.variants.get_alora_offsets_for_forward(...)`. Calls `calculate_alora_offsets(peft_config, active_adapter, input_ids)` (`peft/tuners/lora/variants.py:615–684`).
2. **Last-match search** — `calculate_alora_offsets` scans each row for every position whose token equals `alora_invocation_tokens[0]`, then full-tensor-equals the suffix; it picks the largest such start index (`if idx > best_match_start_idx`). The returned `alora_offsets[i]` is `seq_len - best_match_start_idx` — the count of trailing positions *at and after* the start of the last match. Returns `None` for that row if no match is found.
3. **Threading** — the hook injects `alora_offsets` into `kwargs` on every wrapped `forward(x, ...)`.
4. **Q/K/V/O/MLP wrappers** — `ShadowResidualLora.forward` (`peft_shadow_residual/stream_gated_lora.py:67–80`) inherits stock peft `Linear.forward`. Stock peft dispatches to `ALoraLinearVariant.forward` (`peft/tuners/lora/variants.py:567–612`). That variant builds `mask = pos >= (T - offsets)` and *only computes the LoRA matmul on rows where the mask is True* (line 611: `res_flat[mask_flat] += lora_B(lora_A(dropout(x_flat[mask_flat]))) * scaling`) — so the delta arithmetic itself is gated, not just multiplied by zero afterwards.
5. **Cross-stream wrapper** — `CrossStreamLora.forward` (`peft_shadow_residual/cross_stream_lora.py:165–221`) re-implements the same mask explicitly (lines 197–219). It still computes the matmul over the full sequence but multiplies by the position mask before adding to the result — functionally identical to `ALoraLinearVariant`, marginally less efficient. The duplication exists because `CrossStreamLora` is not a stock peft `Linear` subclass; it inherits `LoraLayer` directly so it picks up the pre-forward hook (and therefore `alora_offsets`) but owns its own `forward`.
6. **Stream-context interaction** — if the active stream is `"base"`, `ShadowResidualLora.forward` short-circuits to `base_layer(x)` *before* delegating to peft, so `alora_offsets` is irrelevant on base-stream calls. The two gates compose: base stream is always delta-free; adapter stream is delta-free only on pre-invocation positions.

### 3.2 Per-token activation timeline (ALORA + SR dual-stream)

A single training row, after collator inserts the canonical control token:

```
   ┌─────────────────────────────────┬─────┬──────────────────────────────────────┐
   │ prompt tokens (user / system)   │ CTL │ completion tokens (assistant)        │
   ├─────────────────────────────────┼─────┼──────────────────────────────────────┤
   │ labels = -100  (masked)         │-100 │ labels = real token IDs (loss here)  │
   ├─────────────────────────────────┴─────┼──────────────────────────────────────┤
   │ alora gate OFF: bare base output      │ alora gate ON: LoRA delta + cross    │
   │                                       │ stream applied                       │
   └───────────────────────────────────────┴──────────────────────────────────────┘
                                       ↑
                                  activation_sequence rightmost match
                                  → alora_offset = trailing_len

   What each stream sees inside the SR decoder:
   ┌───────────────────────────────────────┬──────────────────────────────────────┐
   │ base stream:    always frozen Wx (delta gated off by stream_context)         │
   ├───────────────────────────────────────┼──────────────────────────────────────┤
   │ adapter stream: Wx only here          │ adapter stream: Wx + LoRA Δ +        │
   │ (alora_offsets gates delta off pre-CTL)│ cross_stream(h_base)                 │
   └───────────────────────────────────────┴──────────────────────────────────────┘
```

Per-token timeline. The collator inserts a single canonical control token at the prompt→completion boundary derived from the labels. ALORA gating uses the same boundary at runtime (via peft's pre-forward hook) to mask the LoRA delta off pre-invocation — so the adapter stream mimics the base stream up to the control token, and only diverges after.

### 3.3 What the stream gate does to each call site

| Projection                       | Called with stream_context                                       | Wrapper class            | Behavior                                                                                                                                                                                              |
| -------------------------------- | ---------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q_proj, o_proj`                 | `"base"` (dual-stream base path)                                | `ShadowResidualLora`     | Returns `base_layer(x)` only — **delta gated off**.                                                                                                                                                   |
| `q_proj, o_proj`                 | `"adapter"` (dual-stream adapter, or early-exit)                | `ShadowResidualLora`     | Stock peft path: base + LoRA Δ, with ALORA mask if configured.                                                                                                                                        |
| `k_proj, v_proj` (disjoint K/V)  | both stream calls happen (one each), separate KV caches          | `ShadowResidualLora`     | Same as above; base call's K/V written to `past_key_values_base`, adapter call's K/V written to `past_key_values`. Two independent K/V tensors per layer.                                             |
| `k_proj, v_proj` (shared base K/V) | called **once**, under `"base"` only                           | `ShadowResidualLora`     | K/V computed from `normed_base` with delta gated off; written once to the single (adapter) cache. The adapter stream never calls K/V. K/V LoRA is forbidden in this mode (`_reject_kv_lora_when_shared`). |
| `gate_proj, up_proj, down_proj`  | `"base"` / `"adapter"` blocks                                    | `ShadowResidualLora`     | Same gating rule: base call → no delta; adapter call → delta + ALORA mask.                                                                                                                            |
| `cross_stream`                   | not in any stream context (called once per layer on h_base)      | `CrossStreamLora`        | Base layer returns zeros, so output is purely `B(A(h_base))*scaling`, optionally masked by `alora_offsets`.                                                                                            |

> **Documented K/V exception (disjoint-K/V default).** The base stream's K/V cache is filled by a call *under* `stream_context("base")`, so the K/V LoRA delta (if you trained one) is *not* applied to the base K/V — the cache stays adapter-free. The adapter K/V cache fills under `"adapter"`, so it does pick up the delta. Two independent caches, never interleaved.

> **Shared-base-K/V mode (§1.6).** When `shared_base_kv=True`, there is no adapter-side K/V at all: K and V are computed once from `normed_base` under `"base"` context and written to the single cache, so the cache is **adapter-independent regardless of what you trained** — and K/V LoRA is rejected at construction time. The adapter touches attention only through Q and O.

### 3.4 The three-mode contract (cross_stream active)

With cross_stream wrapped (the production path), the SR forward behaves as one of three modes for each token position, depending on whether `alora_invocation_tokens` is set and whether the position sits before or after the last invocation match:

| Mode                                                | Adapter Q/K/V/O/MLP                              | cross_stream(h_base) into h_adapt | Adapter K/V cache write                                                       | Base K/V cache write                |
| --------------------------------------------------- | ------------------------------------------------ | --------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------- |
| **A.** `alora_invocation_tokens` unset              | delta fires on all positions: `(W + ΔW)x`        | full delta on all positions       | filled with adapter-side values (delta included)                              | filled with base-side values (delta-free) |
| **B-pre.** ALORA on, `pos < p`                      | delta gated off: just `Wx`                       | zero (mask)                       | filled with *base-equivalent* values (`k_adapt[pos<p] == k_base[pos<p]`)       | filled with base-side values        |
| **B-post.** ALORA on, `pos ≥ p`                     | delta fires: `(W + ΔW)x`                         | full delta                        | filled with adapter-side values (delta included)                              | filled with base-side values        |

> **The K/V-cache columns above describe the disjoint-K/V default (§1.3).** Under `shared_base_kv=True` (§1.6), the "Adapter K/V cache write" and "Base K/V cache write" columns collapse: there is a single cache, filled once from the frozen base stream, identical across all three modes — the adapter never writes K/V. ALORA gating and cross-stream still apply to Q/O/MLP exactly as tabulated; only the K/V columns change.

The defining invariants of Mode B-pre (the "adapters not yet active" regime):

1. **Stream equality.** `h_adapt[pos<p] == h_base[pos<p]` at every layer's input and output (inductive across layers; base case is `h_adapt = inputs_embeds.clone() == h_base = inputs_embeds` in `modeling_hf.py:250–251`).
2. **K/V cache equality.** Adapter K/V cache values agree with base K/V cache values on pre-invocation positions, layer by layer. They live in *separate physical tensors* (`past_key_values` vs `past_key_values_base`) so reads on one don't affect the other — only the values match.
3. **No cross-stream contribution.** `cross_stream(h_base)` contributes zero to `h_adapt` on pre-invocation positions (mask in `cross_stream_lora.py:218–219`), so the base→adapter merge is genuinely off pre-invocation — nothing from W_cross propagates into the next layer's adapter input.
4. **Logits equality.** The pre-invocation logits emitted by the model with ALORA gating + non-zero `lora_B` equal the logits of the same model with `lora_B = 0` (i.e. base-equivalent), to fp tolerance.

> **Compute is *not* skipped pre-invocation.** The adapter path still runs every projection, layernorm, and attention call; it just runs them as `Wx` rather than `(W + ΔW)x`. This is required, not wasteful: the adapter K/V cache must be populated on pre-invocation positions so that post-invocation attention can read coherent keys/values for the whole sequence. The only thing that *is* skipped pre-invocation is the LoRA matmul itself (`ALoraLinearVariant` gathers post-invocation rows before doing the matmul, see §3.1.2).

### 3.5 What pins the contract — the invariant tests

The Mode B-pre invariants are by-construction consequences of the `stream_context` + `ALoraLinearVariant` + `CrossStreamLora` masking, but failure modes (someone moves a `stream_context`, peft changes variant dispatch, K/V LoRA gets retargeted, `task_type` gets dropped) would be silent. `tests/experimental/shadow_residual/test_alora_invocation_invariant.py` pins them with two CPU-only tests using a tiny randomly-initialized `GraniteSwitchConfig`:

| Test                                                          | Asserts                                                                                                                                                                                                                                       |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_pre_invocation_logits_match_base_dual_stream`           | Pre-invocation logits with ALORA + perturbed `lora_B` match the logits of the same model with `lora_B = 0` (base-equivalent control). Post-invocation logits must differ — otherwise the gating is vacuous.                                  |
| `test_pre_invocation_kv_cache_matches_base_dual_stream`       | For every layer, `past_key_values` and `past_key_values_base` agree on K and V slices over `[0, p)`. At least one layer must *disagree* over `[p, T)` — otherwise the adapter is a no-op.                                                     |

Both tests run in <5s on CPU with no HF download, no GPU. The `task_type="CAUSAL_LM"` requirement (§3.1.1) was discovered by these tests: without it, peft refuses to install `ALoraLinearVariant` and the logits-invariant assertion fails.

---

## 4. What gets trained — frozen vs active

### 4.1 Selection knobs

The trainer (`training/sr_train.py` → `GraniteSwitchTrainingArguments`) takes three boolean flags that map directly to PEFT `target_modules`:

| Flag                                          | Adds to target_modules                  | Trainable parameters added                                                                                              |
| --------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `--cross_stream True` (default)               | `"cross_stream"`                         | `cross_stream.lora_A` + `cross_stream.lora_B` per layer (rank from `--cross_stream_rank`, default 96)                    |
| `--attn_lora True`                            | `q_proj, k_proj, v_proj, o_proj`         | `lora_A`+`lora_B` on each, rank `--lora_rank` (default 32)                                                              |
| `--mlp_lora True`                             | `gate_proj, up_proj, down_proj`          | `lora_A`+`lora_B` on each, rank `--lora_rank`                                                                           |

At least one must be `True` — the dataclass `__post_init__` raises otherwise. Whether SR's dual-stream forward actually runs is determined at runtime by the model gate (see §1.4): only the presence of `"cross_stream"` in `target_modules` turns it on.

### 4.2 Frozen vs trainable (after PEFT freeze pass)

Per-decoder-layer parameter inventory (cross_stream + attn_lora selected):

```
┌─────────────────────────────────────────────┐  ┌──────────────────────────────────────────────────┐
│ FROZEN                                      │  │ TRAINABLE  (LoRA / CrossStreamLora deltas)       │
│                                             │  │                                                  │
│ embed_tokens.weight                         │  │ cross_stream.lora_A.default.weight  [R, H]       │
│ lm_head.weight (tied)                       │  │ cross_stream.lora_B.default.weight  [H, R]       │
│                                             │  │      — output added to h_adapt at end of layer   │
│ model.layers[i].input_layernorm.weight      │  │                                                  │
│ model.layers[i].post_attention_layernorm    │  │ self_attn.q_proj.lora_A.default.weight  [r, H]   │
│ model.norm.weight                           │  │ self_attn.q_proj.lora_B.default.weight  [Q, r]   │
│                                             │  │ self_attn.k_proj.lora_A/B.default.weight         │
│ self_attn.q_proj.base_layer.weight  [Q, H]  │  │ self_attn.v_proj.lora_A/B.default.weight         │
│ self_attn.k_proj.base_layer.weight  [K, H]  │  │ self_attn.o_proj.lora_A/B.default.weight         │
│ self_attn.v_proj.base_layer.weight  [V, H]  │  │      — gated OFF on stream_context("base") calls │
│ self_attn.o_proj.base_layer.weight  [H, Q]  │  │                                                  │
│                                             │  │   (only added when --mlp_lora True:)             │
│ gate_proj.base_layer.weight  [I, H]         │  │ gate_proj.lora_A/B   up_proj.lora_A/B            │
│ up_proj.base_layer.weight    [I, H]         │  │ down_proj.lora_A/B                               │
│ down_proj.base_layer.weight  [H, I]         │  │                                                  │
│                                             │  │ requires_grad = True; backward writes only here. │
│ cross_stream.base_layer (zero-output,       │  │ All initialized as: A <- Kaiming, B <- 0         │
│   no params)                                │  │   (delta starts at 0)                            │
│                                             │  │                                                  │
│ requires_grad = False (set by PEFT's        │  │                                                  │
│   freeze pass)                              │  │                                                  │
└─────────────────────────────────────────────┘  └──────────────────────────────────────────────────┘
   FROZEN  ──── no grad ────►  TRAINABLE
   FROZEN  ◄──── grad flow ────  TRAINABLE

Approximate parameter count: ~(R + r·k_attn) × H per layer  (R≈96, r≈32 typical)
```

### 4.3 Concrete training step

```
1. forward
   embed_tokens(input_ids) → inputs_embeds × embedding_multiplier
   for each layer:
       if cross_stream wrapped (any layer):
           dual-stream:
               h_base, h_adapt = layer(h_base, h_adapt, ...)
               h_adapt += cross_stream(h_base)        # B·A delta only here
       else:
           single-stream:
               h_adapt = layer(None, h_adapt, ...)    # only adapter ctx
   logits = lm_head(norm(h_adapt)) / logits_scaling
   loss = cross_entropy(logits, labels)               # labels mask out prompt

2. backward
   d(loss)/d(h_adapt) propagates through the adapter stream only.
   Reaches: q/o/k/v/gate/up/down LoRA A/B (if listed) + cross_stream A/B.
   Everything else has requires_grad=False; grads stop there.
   Base K/V caches and h_base path carry no grads (frozen Linear ops).

3. optimizer step
   AdamW updates only the parameters with requires_grad=True
   — the LoRA / CrossStreamLora ModuleDicts.
```

### 4.4 Optional: cross_stream off (LoRA-only mode)

With `--cross_stream False --attn_lora True`, only stock LoRA wrappers are installed (on `_StreamGatedLinear` projections), no `CrossStreamLora` exists, the model gate returns `False`, and SR runs the **single-stream early-exit** — one Q/K/V/O pass per layer, one MLP pass, no per-layer cross-stream injection. This is functionally LoRA on the upstream Granite model (with the documented unfused-projection numerical drift). The save format is indistinguishable from a stock PEFT LoRA checkpoint — `load_shadow_residual_peft_model` will still load it onto the SR base; `PeftModel.from_pretrained(stock_base, ...)` can also load it (but won't bit-match because of the unfused projections; loaded onto the SR base reproduces training exactly).

### 4.5 What the activation sequence does at training time

Three separate consumers, in order:

1. **Offline** — `add_labels.py` uses the activation sequence to stamp the `labels` column. Tokens up to and including the rightmost match get `-100`; everything after gets the real token ID. The cross-entropy loss therefore only reaches positions strictly after the boundary.
2. **Per-batch** — the collator (`DataCollatorForSingleExpert{,Packed}`, `training/collator.py`) finds the last `-100→non-(-100)` transition in `labels` and inserts the *canonical control token* at that position in `input_ids` (with `-100` in `labels` at the same index, so the inserted token never contributes to loss).
3. **Per-forward (only if ALORA is configured)** — if `LoraConfig(alora_invocation_tokens=[...], task_type="CAUSAL_LM")` was set (`sr_train.py:138–155` builds the config from `--alora_invocation_token_ids`; `task_type="CAUSAL_LM"` is on line 150 and is load-bearing — see §3.1.1), peft installs `ALoraLinearVariant` on every wrapped projection plus a pre-forward hook on every `LoraLayer`. The hook (`get_alora_offsets_for_forward`, `peft/tuners/lora/variants.py:706`) scans `input_ids` at runtime, calls `calculate_alora_offsets` (`variants.py:615–684`) which returns per-row `alora_offsets[i]` = (length at and after rightmost match start, or `None` if no match), and threads it into every wrapper's `forward(...)`. The variant's `forward` (`variants.py:567–612`) gathers the post-invocation rows and only runs the LoRA matmul on those. `CrossStreamLora.forward` (`peft_shadow_residual/cross_stream_lora.py:165–221`) reproduces the same masking explicitly — it inherits the hook via `LoraLayer` but owns its own delta math.

The fully-wired training-time `LoraConfig` for SR + ALORA looks like:

```python
LoraConfig(
    r=32, lora_alpha=32.0, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "cross_stream"],
    rank_pattern={"cross_stream": 96},
    bias="none",
    task_type="CAUSAL_LM",                       # required for ALORA dispatch
    alora_invocation_tokens=[100264, 78191, 100265],   # the activation sequence
)
```

> The collator's inserted control token and the ALORA invocation match are *not* the same thing, but they typically align: the activation sequence used to label the dataset can also be used as the ALORA invocation sequence, so the runtime gate flips on at the same boundary the loss computes against.

> **Three knobs, all required.** Drop any of `alora_invocation_tokens`, `task_type="CAUSAL_LM"`, or `"cross_stream"` in `target_modules`, and the result is a *silently different* training run: peft falls back to unconditional LoRA, or the dual-stream forward never engages, with no error message. The `test_alora_invocation_invariant.py` tests catch the first two; absence of `cross_stream` just means SR was never wanted in that run.

### 4.6 Reading off the wire

To inspect a trained checkpoint:

```bash
$ ls ./checkpoints
adapter_config.json   adapter_model.safetensors   training_args.bin

$ python - <<'PY'
from safetensors import safe_open
with safe_open("./checkpoints/adapter_model.safetensors", framework="pt") as f:
    for k in list(f.keys())[:8]:
        print(k, tuple(f.get_tensor(k).shape))
PY
base_model.model.model.layers.0.cross_stream.lora_A.default.weight (96, 1024)
base_model.model.model.layers.0.cross_stream.lora_B.default.weight (1024, 96)
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight (32, 1024)
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight (1024, 32)
...
```

The shapes confirm: `cross_stream` uses the rank-pattern override (R=96, square H×H); attention LoRAs use the global rank (r=32). All other base parameters live on disk in the upstream model checkpoint — they are not duplicated in `adapter_model.safetensors`.

---

Sources: `src/granite_switch/experimental/shadow_residual/{modeling_hf,decoder_hf,attention_hf,cross_stream,_stream_context,_stream_gated_linear,weight_transfer}.py`, `peft_shadow_residual/{factory,load,cross_stream_lora,stream_gated_lora}.py`, `training/{sr_train,collator,add_labels}.py`, `tests/experimental/shadow_residual/{test_alora_invocation_invariant,test_dual_kv_cache_invariant,test_shadow_residual}.py`, `peft/tuners/lora/variants.py` (ALoraLinearVariant, calculate_alora_offsets, get_alora_offsets_for_forward), `docs/SHADOW_RESIDUAL.md`.
