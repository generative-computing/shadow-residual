# Shadow Residual support for Granite 5.0 (`granitemoe`)

Plan to add Shadow Residual (SR) support for the Granite 5.0 model family
(`model_type: "granitemoe"`, `GraniteMoeForCausalLM`) — a sparse **Mixture-of-Experts**
decoder. **Iteration 1 adapts attention only** (Q/O + `cross_stream`); the sparse
expert MLP runs **frozen** so the base stream stays bit-identical to the stock
model. A dedicated section (§7) scopes what training the experts would take later.

This plan is written against the **current stock-PEFT** `shadow-residual` code on
`feature/stock-peft-sr` — the model owns the dual-stream forward, adapters are
plain `peft` LoRA, and there are **no custom LoRA classes**. Any prior 5.0/MoE
design notes that reference `stream_context`, `ShadowResidualLora`,
`CrossStreamLora`, `_register_cross_stream`, or `src/granite_switch/…` are
obsolete on the mechanics; only their *architecture* reasoning carries over and is
restated here against the real code.

---

## 1. Context — why this change

SR has trained on **dense** Granite 4.1 / 4.2 bases. Granite 5.0 is the first
**MoE** target: every layer's FFN is a sparse expert bank (`block_sparse_moe`:
router + `GraniteMoeParallelExperts`), with **no** dense `shared_mlp` and **no**
Mamba layers. The current SR decoder only knows how to build a **dense SwiGLU**
MLP (`ShadowResidualMLP` in `decoder_hf.py:49`), sized from
`shared_intermediate_size`. On a `granitemoe` base:

- `transfer_base_weights` (`weight_transfer.py`) has no `block_sparse_moe.*`
  branch, so it matches **zero** MLP keys.
- SR's dense MLP stays at **random init**, and a garbage MLP contribution is added
  into the residual stream on **both** streams, every layer — so the "frozen base
  stream is bit-identical to the base" invariant is *violated* (the base stream is
  noise, not the base). This is the same failure class already recorded in
  `build.py:66-73` (a mis-sized dense MLP blew hidden states up 78× at layer 0 on
  granite-4.1-3b); on `granitemoe` the MLP is *unmatchable by construction*, so the
  corruption is guaranteed.

**Key correctness fact:** attention-only training does **not** let us skip the MoE
MLP. The FFN executes on every token in every layer regardless of `target_modules`.
So iteration 1 must make the base MoE MLP **execute correctly and frozen** — a
strict subset of full MoE support (build the frozen expert path; add no expert
LoRA). Intended outcome: an SR adapter trains on a `granitemoe` base with LoRA on
attention (Q/O) + `cross_stream`, experts frozen, and the base stream reproduces
stock `GraniteMoeForCausalLM` logits within fp tolerance.

## 2. Target model and config

Granite 5.0 20b (`ibm-research/granite-5.0-20b-sft`, on COS at
`/danieloh_cos/antonp/models/granite-5.0-20b-sft`) — the smallest 5.0 variant,
already fetched, ~22 B total params (top-4 of 56 experts active/token).

`granitemoe` config facts that matter (verified against transformers 5.8 in-env):

| field | 20b | Notes for SR |
|---|---|---|
| `model_type` | `granitemoe` | new; parent `GraniteMoeHybridConfig` is a superset (see §3) |
| `head_dim` | 64 | attention shapes already handled |
| `num_attention_heads` / `num_key_value_heads` | 32 / 8 | GQA, same shape family as 4.x |
| `num_hidden_layers` | 40 | drives the decoder loop; no structural change |
| `intermediate_size` | 1536 | **per-expert** FFN width (not a dense MLP width) |
| `num_local_experts` | 56 | SR currently pins this to `0` — must carry it |
| `num_experts_per_tok` | 4 | top-k; must carry it |
| `shared_mlp` | **absent** | granitemoe has no dense shared MLP |
| `layer_types` / mamba | **absent** | inherently attention-only |
| `tie_word_embeddings` | false | untied lm_head already handled (4.2 work) |
| `rope_theta` | 30M | read from config into RoPE; no code change |

The larger 5.0 variants differ only in **scale knobs** (`hidden_size`,
`num_attention_heads`, `num_hidden_layers`, `num_local_experts` up to 224,
`num_experts_per_tok` up to 8, `rope_theta`). No new structural element — so
supporting 20b supports all three, modulo runtime scale (§8).

## 3. Class hierarchy — no changes

Keep `ShadowResidualConfig(GraniteMoeHybridConfig)` and
`ShadowResidualPreTrainedModel(GraniteMoeHybridPreTrainedModel)`. The hybrid config
is a **superset** that already carries `num_local_experts` / `num_experts_per_tok`
/ `router_aux_loss_coef`; SR has merely been *zeroing* the expert count. The fix is
to stop zeroing it, not to rebase the parent. This also keeps the dense-4.x path in
the same class.

**Per the user's explicit constraint: no new class.** MoE is a **per-layer mode**
inside the existing `ShadowResidualMLP`, branching on `config.num_local_experts > 0`.
One `ShadowResidualConfig`, one `ShadowResidualModel`, one
`ShadowResidualForCausalLM`, one `ShadowResidualDecoderLayer`, one
`ShadowResidualMLP`. Do **not** add a `ShadowResidualMoeBlock`.

## 4. How the frozen dual-stream MoE fits the current architecture

The current decoder (`decoder_hf.py`) has **no** `stream_context` / `disable_adapter`
toggling. The two streams are separate call sites:

```python
# decoder_hf.py:159-160 (dual-stream forward)
mlp_base  = self.mlp.forward_base_only(normed_base)   # base stream: base_layer only
mlp_adapt = self.mlp(normed_adapt)                    # adapter stream: LoRA fires
# decoder_hf.py:196 (forward_bare, no adapter)
h = h + self.mlp(normed) * self.residual_multiplier
```

`ShadowResidualMLP.forward_base_only` calls `_base_only(proj, x)` (proj.base_layer,
delta bypassed); `forward` runs the wrapped projections. This is what makes the
`GradientCheckpointingLayer` recompute exact — both are **pure functions of `x`**.

The frozen MoE block slots straight in with the **same two-method shape**:

- **In iteration 1 the experts are frozen and are NOT LoRA targets**, so the expert
  bank does not need to be `_StreamGatedLinear` and needs no `base_layer`
  indirection. `forward` and `forward_base_only` run the **same** frozen MoE
  computation — they are functionally identical in MoE mode. No stream gating, no
  op-count trickery.
- **Routing is independent per stream** (base routes on `normed_base`, adapter on
  `normed_adapt`). This is *correct*, not a compromise: the base stream's MoE output
  depends only on `normed_base` (the shared layernorm over `cat([h_base,h_adapt])`
  is per-row and splits cleanly at `[:bsz]`, `decoder_hf.py:141,156`), which is
  exactly what stock `GraniteMoeForCausalLM` sees — so `h_base` reproduces the stock
  model regardless of the adapter stream. The obsolete "route once on base, reuse for
  adapter" note does not apply to this architecture (there is no shared routing call),
  and with no expert LoRA the experts are identical in both streams, so per-stream
  routing is simply the real forward on each stream's hidden state.

**Frozen-base invariant: preserved.** The base MoE contribution is bit-for-bit the
stock model's MoE output (modulo the documented fused-projection fp caveat).

**Router / aux-loss policy:** the router is **frozen** and never trained; SR adds no
`router_aux_loss` (a deliberate non-goal — SR does not train load balancing).

**Gradient checkpointing:** routing is deterministic given the input, so
checkpoint recompute reproduces the same expert partition — recompute stays exact.
The one caveat: `expert_size = expert_size.tolist()` inside the router is a
data-dependent GPU→CPU sync and a **`torch.compile` graph-break** (upstream flags
this). SR does not compile the decoder today, so this is a documented limitation,
not a correctness issue.

## 5. Iteration 1 — the change surface (attention-only, experts frozen)

All changes are **model + weight-transfer + config**. **Zero PEFT changes**:
attention Q/O and `cross_stream` are wrapped by stock `lora.Linear` exactly as today,
and PEFT's `get_peft_model` freezes all non-LoRA params — so the plain-`nn.Parameter`
experts are left frozen automatically.

Ordered steps:

1. **`model_config.py`** — stop pinning `num_local_experts=0`. Accept it from
   `**kwargs` (the parent `GraniteMoeHybridConfig` already stores
   `num_local_experts` / `num_experts_per_tok`). Leave the
   `shared_intermediate_size` default alone — it is harmless in MoE mode (nothing
   builds a dense MLP from it there), just no longer load-bearing.

2. **`build.py::build_sr_config`** — do **not** force `shared_intermediate_size =
   intermediate_size` when `num_local_experts > 0`, and do not rewrite
   `intermediate_size` (it is the per-expert width on granitemoe). The
   `layer_types` all-attention normalization is harmless against a config with no
   `layer_types` and stays.

3. **`decoder_hf.py::ShadowResidualMLP.__init__`** — branch on
   `config.num_local_experts > 0`:
   - **MoE mode:** build a frozen `router` (`GraniteMoeTopKGating(H, num_local_experts,
     num_experts_per_tok)`), `input_linear` (`GraniteMoeParallelExperts(n_experts, H,
     2*intermediate_size)`), `output_linear` (`GraniteMoeParallelExperts(n_experts,
     intermediate_size, H)`) — plain upstream classes, plain `nn.Parameter`, **not**
     `_StreamGatedLinear`.
   - **Dense mode:** keep today's `gate_proj` / `up_proj` / `down_proj` SwiGLU.

4. **`decoder_hf.py::ShadowResidualMLP.forward` / `forward_base_only`** — in MoE
   mode, both run the mirrored `GraniteMoeMoE.forward` on their own `x`: reshape
   `[-1,H]` → `router(x)` → gather `x[batch_index]` → `input_linear(·, expert_size)`
   → `chunk(2)` SwiGLU → `output_linear(·, expert_size)` → `* batch_gates[:,None]`
   → `zeros.index_add(0, batch_index, ·)` scatter → view back. Keep the dense body
   for dense mode. Because the branch lives in **`forward`**, the bare no-adapter
   path (`forward_bare` → `self.mlp(normed)`, `decoder_hf.py:196`) is covered too.

5. **`weight_transfer.py`** — add straight-`copy_` branches (mirror the existing
   `shared_mlp` branches, incl. `.base_layer.` variants for symmetry):
   - `...block_sparse_moe.input_linear.weight` (3-D `[n_experts, 2*inter, H]`) →
     SR `mlp.input_linear.weight` — no fused reslice (upstream already stores
     gate⊕up fused at `2*inter`; SR's SwiGLU chunk consumes it identically).
   - `...block_sparse_moe.output_linear.weight` (3-D `[n_experts, H, inter]`) →
     SR `mlp.output_linear.weight`.
   - `...block_sparse_moe.router.layer.weight` (2-D) → SR `mlp.router.layer.weight`
     (the generic name-strip branch may already cover it; add explicitly for clarity).

6. **`factory.py::_materialize_and_transfer`** — **no code change required**. The
   3-D expert `nn.Parameter` and router weight have no `reset_parameters` hook, so
   `to_empty(cpu)` leaves them as garbage — but `transfer_base_weights` runs *after*
   `to_empty` on rank 0 (`factory.py:98,120`) and repopulates them via the new
   branches, so they are real before FSDP's `sync_module_states` broadcast (same
   pattern as the RoPE / `cross_stream` re-init). Recommended: extend the
   `SR_DIAG_MATERIALIZE` audit to confirm expert weights are finite post-transfer.

7. **`config_helpers.py`** — optional hardening: assert `num_experts_per_tok <=
   num_local_experts` when experts exist. The all-attention `layer_types` check is
   unchanged and passes (granitemoe has no non-attention layers).

## 6. Testing (iteration 1)

Add a tiny `GraniteMoeForCausalLM` fixture (2 layers, 4 experts, top-2, small dims)
under `tests/`. CPU where possible.

1. **Frozen-base MoE parity.** Build the SR base (`build_sr_base`) from a small
   `granitemoe`, run the **dual-stream** `h_base` and the **`forward_bare`** path,
   and assert both match a stock `GraniteMoeForCausalLM` (same weights) within fp
   tolerance (the fused-projection caveat in `CLAUDE.md` applies — small divergence
   expected, not bit-exact).
2. **Experts frozen after PEFT attach.** `get_shadow_residual_peft_model` with
   `target_modules` = `{q_proj, o_proj, cross_stream}`; assert every expert /
   router param has `requires_grad=False`, attention LoRA + cross-stream are
   trainable, forward runs clean, one train step decreases loss.
3. **Single-class / mode switch.** The same `ShadowResidualMLP` builds the MoE bank
   on a `granitemoe` fixture and the dense SwiGLU on a dense fixture — one class,
   branch on `num_local_experts`.
4. **No dense-4.x regression.** Re-run `tests/shadow_residual/` +
   `tests/peft_shadow_residual/` (frozen-base, shared-KV, bare-model invariants).
5. **Gradient-checkpoint parity.** Loss/grads match between checkpointed and
   non-checkpointed dual-stream forward on the fixture (routing determinism).

**Vela end-to-end** (the real coverage): mirror the answerability train→eval flow
(`docs/RERUN_ANSWERABILITY_4WAY.md`). One new `ans-5p0-20b` row: FSDP multi-GPU
(22 B total, like the 8b runs — `config/accelerate/fsdp_4gpu.yaml` with raised
`--num_processes`), a new train YAML pointed at the COS base with
`target_modules` = `{q_proj, o_proj, cross_stream}` (**no** MLP/expert targets),
`shared_base_kv` semantics as usual, **thinking-OFF** (no `--enable-thinking` — the
documented score-tanking gotcha). Confirm 5.0's chat template / EOS / generation
config the same way 4.2's was checked before the first real run. Baselines to land
in a sensible band: 4.1-3b 0.8972 · 4.1-8b 0.9048 · 4.2-3b 0.9092 · 4.2-8b 0.9164.

**Acceptance:** (1) frozen-base MoE parity green on the fixture; (2) dense-4.x
suite still green; (3) one 5.0 attention-only adapter trains on Vela (finite loss
from step 1 under FSDP) and evaluates in a non-degenerate accuracy band.

## 7. Later iteration — training the experts (MLP tier)

Scoped here per request; **not** in iteration 1. This is where SR gains trainable
capacity inside the sparse FFN.

- **Per-expert LoRA** over the 3-D expert weights `[n_experts, out, in]`:
  `A=[n_experts, r, in]`, `B=[n_experts, out, r]`, per-expert delta applied in the
  expert-grouped layout (sliced by `expert_size`). Param cost scales as
  `n_experts × r × (in + out)` — the main rank-budget lever, and where the 128- and
  224-expert variants get expensive (a fixed `r` costs ~2×/4× the 56-expert adapter).

- **The sharp edge — stock PEFT has no 3-D expert LoRA.** `peft.tuners.lora.layer.
  Linear` targets a 2-D `nn.Linear`; `GraniteMoeParallelExperts.weight` is a 3-D
  `nn.Parameter` with a custom `F.linear`-per-expert forward that PEFT cannot target
  by name. So *fully-stock* expert LoRA is likely **not** achievable. Options, each
  with a cost:
  1. A small custom PEFT layer for the 3-D expert tensor — reintroduces exactly the
     kind of custom LoRA module the stock-PEFT direction removed (scoped to experts).
  2. Restructure experts as `n_experts` real `nn.Linear` modules stock LoRA can
     target — memory/perf blowup at 56–224 experts.
  3. LoRA only on the **router** `layer` (a 2-D `nn.Linear`, stock-targetable) — cheap
     and stock-compatible, and under the independent-per-stream routing here it shifts
     *which experts fire* in the adapter stream (a bigger behavioral lever than
     expert-weight LoRA). Lowest-risk first prototype for the MLP tier.

- **Frozen-base tension for options 1–2.** If experts become LoRA targets, the
  base-stream expert forward must bypass the delta — but a raw `nn.Parameter` has no
  `base_layer` indirection, so `forward_base_only` would need an explicit "delta off"
  path mirroring `_base_only`. Resolving this is the core architectural decision for
  iteration 2.

- **Routing policy (kept):** frozen router, no `router_aux_loss`. Adapting the router
  (option 3) is the one exception worth prototyping; re-routing with a trained router
  reintroduces load-balance/aux-loss concerns and is deferred.

**Recommendation:** land iteration 1 (attention-only) first; for the MLP tier,
prototype **router-only LoRA (option 3)** as the stock-compatible lever before
committing to any custom 3-D-expert module.

## 8. Scale implications for the larger variants

No new code; runtime stress only. `num_hidden_layers` 40→64→72 (longer loop, more
FSDP units); `num_local_experts` 56→128→224 (frozen bank grows ~2.3×/4× — iter-1
cost is memory/FLOPs; MLP-tier LoRA params scale directly); `num_experts_per_tok`
4→8 (~2× MoE FLOPs/token, throughput not correctness); `rope_theta` and
`hidden_size` read from config. Iterate on the 20b; move to 64/72-layer variants
only after attention-only parity + a short train are green.

## 9. File map

- **`model_config.py`** — stop pinning `num_local_experts=0`.
- **`build.py`** — don't force `shared_intermediate_size` / rewrite `intermediate_size`
  when experts exist.
- **`decoder_hf.py`** — `ShadowResidualMLP` gains a `num_local_experts > 0` branch in
  `__init__` (build frozen router + parallel experts) and in `forward` /
  `forward_base_only` (mirror `GraniteMoeMoE.forward`). **No new class.**
- **`weight_transfer.py`** — three `block_sparse_moe.*` `copy_` branches.
- **`config_helpers.py`** — optional `num_experts_per_tok <= num_local_experts` check.
- **`factory.py`** — no change (transfer repopulates experts post-`to_empty`);
  optional diag assert.
- **Tests** — tiny `granitemoe` fixture; frozen-base parity; experts-frozen attach;
  mode-switch; no dense-4.x regression; grad-ckpt parity. Vela: one `ans-5p0-20b` row.
- **Class hierarchy** — unchanged (`GraniteMoeHybrid*` parents).
