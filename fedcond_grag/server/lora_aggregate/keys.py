"""Shared helpers for locating LoRA adapter tensors inside a PEFT model's
state dict. peft names the pieces of a wrapped `nn.Linear` as:

    <module>.base_layer.weight      -- frozen original weight
    <module>.lora_A.default.weight  -- down-projection (adapter name "default")
    <module>.lora_B.default.weight  -- up-projection

All aggregation strategies key off these substrings rather than exact
suffixes so they keep working if the adapter is ever named something other
than "default" (multi-adapter setups).
"""
from __future__ import annotations

import torch.nn as nn


def has_lora(model: nn.Module) -> bool:
    return getattr(model, "peft_config", None) is not None


def lora_state_dict(model: nn.Module) -> dict:
    return {
        k: v.clone()
        for k, v in model.state_dict().items()
        if ".lora_A." in k or ".lora_B." in k
    }


def lora_a_keys(state_dict: dict) -> list[str]:
    return [k for k in state_dict if ".lora_A." in k]


def b_key_of(a_key: str) -> str:
    return a_key.replace(".lora_A.", ".lora_B.")


def base_key_of(a_key: str) -> str:
    return a_key.split(".lora_A.")[0] + ".base_layer.weight"


_QUANTIZED_PARAM_TYPES = {"Params4bit", "Int8Params"}


def is_quantized_param(param) -> bool:
    return type(param).__name__ in _QUANTIZED_PARAM_TYPES
