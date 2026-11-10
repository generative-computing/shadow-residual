# SPDX-License-Identifier: Apache-2.0
"""Shared-A multi-task SR training: one LoRA ``A`` per (layer, module) shared by
every task, one ``B`` per task.

The PEFT model holds one stock LoRA adapter per task (same ``LoraConfig``).
:func:`setup_shared_a` makes every adapter's ``lora_A`` the *same* module and
replaces each LoRA layer's forward with a per-row routed one::

    z   = A(dropout(x))                          # computed once
    out = base_layer(x) + sum_t route[:, t] * scaling_t * B_t(z)

``route`` is the one-hot task of each batch row, set by :class:`TaskRouter`
before the forward (and unchanged until after backward, so gradient-checkpoint
recompute is exact). Every ``B_t`` stays in the graph each step — rows of other
tasks multiply it by exactly zero — so DDP needs no ``find_unused_parameters``.

Each task saves as an ordinary SR adapter (``save_pretrained`` writes
``<out>/<task>/``); the ``lora_A`` tensors are identical across the exports.
Base-stream calls go through ``base_layer`` directly (``_base_only``), so the
frozen-base invariant is unaffected.
"""

from __future__ import annotations

import types
from typing import Sequence

import torch


class TaskRouter:
    """Holds the current batch's one-hot task routing, shared by all LoRA layers."""

    def __init__(self, task_names: Sequence[str]):
        self.task_names = list(task_names)
        self.route: torch.Tensor | None = None

    def set(self, task_ids: torch.Tensor, dtype: torch.dtype | None = None) -> None:
        route = torch.nn.functional.one_hot(task_ids.long(), len(self.task_names))
        self.route = route.to(dtype or torch.float32)

    def clear(self) -> None:
        self.route = None


def _routed_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if self.disable_adapters:
        return self.base_layer(x, *args, **kwargs)
    router: TaskRouter = self._sr_task_router
    route = router.route
    if route is None:
        raise RuntimeError("shared-A LoRA layer called without a task route set")
    if route.shape[0] != x.shape[0]:
        raise RuntimeError(
            f"task route has {route.shape[0]} rows but the layer input has {x.shape[0]}"
        )
    result = self.base_layer(x, *args, **kwargs)
    names = router.task_names
    lora_A = self.lora_A[names[0]]
    dropout = self.lora_dropout[names[0]]
    z = lora_A(dropout(x.to(lora_A.weight.dtype)))
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    delta = None
    for t, name in enumerate(names):
        w = route[:, t].to(z.dtype).view(shape) * self.scaling[name]
        term = self.lora_B[name](z) * w
        delta = term if delta is None else delta + term
    return result + delta.to(result.dtype)


def _lora_layers(model):
    from peft.tuners.lora.layer import LoraLayer

    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def setup_shared_a(peft_model, task_names: Sequence[str], lora_config, dtype=None) -> TaskRouter:
    """Turn a PEFT model holding adapter ``task_names[0]`` into a shared-A model.

    Adds the other tasks as adapters with ``lora_config``, ties every adapter's
    ``lora_A`` to the first task's, marks all LoRA params trainable, and installs
    the routed forward on every LoRA layer. Returns the router to drive it.
    """
    names = list(task_names)
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError(f"need >= 2 distinct task names, got {names}")
    if names[0] not in peft_model.peft_config:
        raise ValueError(f"model has no adapter {names[0]!r}; build it with that adapter_name")
    for name in names[1:]:
        peft_model.add_adapter(name, lora_config)
    if dtype is not None:
        peft_model.to(dtype=dtype)

    router = TaskRouter(names)
    layers = _lora_layers(peft_model)
    if not layers:
        raise ValueError("no LoRA layers found")
    for layer in layers:
        shared = layer.lora_A[names[0]]
        for name in names[1:]:
            if layer.r[name] != layer.r[names[0]]:
                raise ValueError("shared A needs the same rank for every task")
            layer.lora_A[name] = shared
        layer._sr_task_router = router
        layer.forward = types.MethodType(_routed_forward, layer)

    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    return router


class MultiTaskCollator:
    """Wraps a collator and adds the batch's ``task_ids`` tensor."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: list[dict]) -> dict:
        task_ids = torch.tensor([int(f["task_id"]) for f in features], dtype=torch.long)
        batch = self.base_collator(features)
        batch["task_ids"] = task_ids
        return batch


def make_multitask_trainer_class(base_cls):
    """A trainer subclass that routes each batch by its ``task_ids`` before the forward."""

    class MultiTaskTrainer(base_cls):
        def __init__(self, *args, task_router: TaskRouter, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_router = task_router

        def _set_signature_columns_if_needed(self):
            super()._set_signature_columns_if_needed()
            if "task_id" not in self._signature_columns:
                self._signature_columns = list(self._signature_columns) + ["task_id"]

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            task_ids = inputs.pop("task_ids")
            self.task_router.set(task_ids)
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

    return MultiTaskTrainer


def parse_task_specs(specs: Sequence[str]) -> list[tuple[str, str, str]]:
    """``NAME:TRAIN_JSONL:VAL_JSONL`` strings -> ``(name, train, val)``."""
    out = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"--task expects NAME:TRAIN:VAL, got {spec!r}")
        out.append((parts[0], parts[1], parts[2]))
    names = [n for n, _, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate task names: {names}")
    return out


__all__ = [
    "TaskRouter",
    "setup_shared_a",
    "MultiTaskCollator",
    "make_multitask_trainer_class",
    "parse_task_specs",
]
