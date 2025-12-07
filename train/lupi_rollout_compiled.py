# coding=utf-8
"""Compiled LUPI-aware Speculative KD rollout with optional torch.compile.

This is the EXPERIMENTAL version WITH torch.compile support.
For the stable version without compilation, see lupi_rollout_optimized.py.

Key features:
1. All optimizations from lupi_rollout_optimized.py
2. Optional torch.compile for Student draft and Teacher verification
3. Static KV-cache option for better torch.compile compatibility

WARNING: torch.compile may cause issues during training.
Use this version only after testing with lupi_rollout_optimized.py.

Reference:
- https://huggingface.co/docs/transformers/en/perf_torch_compile
- https://huggingface.co/docs/transformers/en/llm_optims
"""

from typing import List, Optional, Tuple
import os

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

# Global flag for torch.compile (can be overridden via environment variable)
USE_COMPILE = os.environ.get("LUPI_USE_COMPILE", "0") == "1"


def _crop_past_key_values(past_key_values, max_length: int):
    """Crop KV-cache to a specified maximum length.

    Args:
        past_key_values: DynamicCache or Tuple of (key, value) tensors per layer.
        max_length: Maximum sequence length to keep.

    Returns:
        Cropped past_key_values (same type as input).
    """
    if past_key_values is None:
        return None

    # Handle DynamicCache (used by modern Transformers models like Qwen3)
    if isinstance(past_key_values, DynamicCache):
        past_key_values.crop(max_length)
        return past_key_values

    # Handle legacy tuple format
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append((
            past_key_values[idx][0][:, :, :max_length, :],
            past_key_values[idx][1][:, :, :max_length, :],
        ))
    return tuple(new_past)


@torch.no_grad()
def _sample_top_p(
    logits: Tensor,
    top_p: float,
) -> Tensor:
    """Top-p (nucleus) sampling for a single step."""
    if top_p >= 1.0:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(0)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    logits_filtered = logits.clone()
    logits_filtered[0, sorted_indices[sorted_indices_to_remove]] = float("-inf")

    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)


def _vectorized_topk_check(
    logits: Tensor,
    draft_ids: Tensor,
    top_k: int,
    temperature: float,
) -> Tensor:
    """Vectorized top-K membership check.

    Args:
        logits: [n, V] logits tensor (verify logits).
        draft_ids: [n] draft token ids to verify.
        top_k: Number of top tokens to consider.
        temperature: Temperature scaling.

    Returns:
        is_accepted: [n] boolean tensor.
    """
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    effective_top_k = min(top_k, probs.shape[-1])
    topk_indices = torch.topk(probs, k=effective_top_k, dim=-1).indices

    draft_expanded = draft_ids.unsqueeze(1)
    is_accepted = (topk_indices == draft_expanded).any(dim=1)

    return is_accepted


# Conditionally compiled functions
def _get_compiled_functions(use_compile: bool):
    """Get compiled or non-compiled versions of internal functions."""

    @torch.no_grad()
    def _student_draft_step(
        student_model,
        input_ids: Tensor,
        past_key_values,
        temperature: float,
        top_p: float,
    ) -> Tuple[int, Tuple, Tensor]:
        """Single Student draft step."""
        student_kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": True,
        }
        if past_key_values is not None:
            student_kwargs["past_key_values"] = past_key_values

        student_out = student_model(**student_kwargs)
        new_past = student_out.past_key_values
        logits = student_out.logits[:, -1, :] / max(temperature, 1e-6)

        next_tok = _sample_top_p(logits, top_p)
        tok_val = next_tok.item()

        return tok_val, new_past, next_tok

    @torch.no_grad()
    def _teacher_batch_forward(
        teacher_model,
        draft_tensor: Tensor,
        past_key_values,
    ) -> Tuple[Tensor, Tuple]:
        """Batch Teacher forward pass."""
        teacher_out = teacher_model(
            input_ids=draft_tensor,
            attention_mask=torch.ones_like(draft_tensor),
            past_key_values=past_key_values,
            use_cache=True,
        )
        return teacher_out.logits, teacher_out.past_key_values

    if use_compile:
        try:
            compiled_student = torch.compile(
                _student_draft_step,
                mode="reduce-overhead",
                fullgraph=False,
            )
            compiled_teacher = torch.compile(
                _teacher_batch_forward,
                mode="reduce-overhead",
                fullgraph=False,
            )
            return compiled_student, compiled_teacher
        except Exception as e:
            print(f"[WARNING] torch.compile failed: {e}. Using non-compiled version.")
            return _student_draft_step, _teacher_batch_forward
    else:
        return _student_draft_step, _teacher_batch_forward


@torch.no_grad()
def _parallel_teacher_verify_compiled(
    teacher_model: PreTrainedModel,
    teacher_past: Tuple,
    first_logits: Tensor,
    draft_ids: List[int],
    top_k: int,
    teacher_temperature: float,
    teacher_top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    teacher_forward_fn,
) -> Tuple[List[int], Tuple, Tensor, dict]:
    """Parallel Teacher verification with optional compiled forward.

    Output logits structure after parallel forward of [d1, d2, ..., d_gamma]:
      all_logits[0, i] = P(next | prefix, d1, ..., d_{i+1})
    So:
      all_logits[0, 0] = P(d2 | prefix, d1)     → for verifying d2
      all_logits[0, 1] = P(d3 | prefix, d1, d2) → for verifying d3
      ...
      all_logits[0, gamma-1] = P(next | prefix, d1, ..., d_gamma) → bonus/next_logits
    """
    gamma = len(draft_ids)

    # 1) Verify first token using first_logits
    probs_0 = F.softmax(first_logits / max(teacher_temperature, 1e-6), dim=-1)
    topk_idx_0 = torch.topk(probs_0, k=min(top_k, probs_0.shape[-1])).indices[0]

    first_accepted = draft_ids[0] in topk_idx_0.tolist()

    if not first_accepted:
        replacement = _sample_top_p(first_logits / max(teacher_temperature, 1e-6), teacher_top_p)
        accepted_token = replacement.item()

        token_tensor = torch.tensor([[accepted_token]], device=device, dtype=dtype)
        all_logits, new_past = teacher_forward_fn(teacher_model, token_tensor, teacher_past)
        next_logits = all_logits[:, -1, :]

        stats = {
            'n_matches': 0,
            'gamma': gamma,
            'acceptance_rate': 0.0,
            'first_rejected': True,
        }
        return [accepted_token], new_past, next_logits, stats

    # 2) Parallel forward pass
    draft_tensor = torch.tensor([draft_ids], device=device, dtype=dtype)
    all_logits, teacher_out_past = teacher_forward_fn(teacher_model, draft_tensor, teacher_past)

    # 3) Vectorized acceptance check (FIXED: correct indexing)
    # - d1 was already verified using first_logits
    # - d2, d3, ..., d_gamma need to be verified using all_logits[0, 0:-1]
    if gamma > 1:
        # Verify draft_ids[1:] using all_logits[0, :-1]
        verify_logits = all_logits[0, :-1, :]  # [gamma-1, V]
        draft_ids_to_verify = torch.tensor(draft_ids[1:], device=device, dtype=torch.long)  # [gamma-1]
        is_accepted_rest = _vectorized_topk_check(verify_logits, draft_ids_to_verify, top_k, teacher_temperature)

        # Combine: d1 (already accepted) + rest
        is_accepted = torch.cat([
            torch.tensor([True], device=device, dtype=torch.bool),
            is_accepted_rest
        ])  # [gamma]
    else:
        is_accepted = torch.tensor([True], device=device, dtype=torch.bool)

    # 4) Find first rejection
    rejection_mask = ~is_accepted
    cumsum = rejection_mask.cumsum(dim=0)
    n_matches = int((cumsum == 0).sum().item())

    # 5) Determine final tokens
    if n_matches == gamma:
        accepted_tokens = list(draft_ids)
        new_past = teacher_out_past
        next_logits = all_logits[:, -1, :]  # bonus token position
    else:
        # n_matches tokens accepted (d1, ..., d_{n_matches})
        accepted_tokens = list(draft_ids[:n_matches])

        # Teacher resampling (FIXED: correct index)
        # Resample from P(d_{n_matches+1} | prefix, d1, ..., d_{n_matches})
        # = all_logits[0, n_matches-1]
        resample_logits = all_logits[0, n_matches - 1, :].unsqueeze(0)
        replacement = _sample_top_p(resample_logits / max(teacher_temperature, 1e-6), teacher_top_p)
        accepted_tokens.append(replacement.item())

        # Crop KV-cache (FIXED: correct length)
        if isinstance(teacher_past, DynamicCache):
            prefix_len = teacher_past.get_seq_length()
        else:
            prefix_len = teacher_past[0][0].shape[2]
        new_cache_len = prefix_len + n_matches  # Only accepted tokens
        new_past = _crop_past_key_values(teacher_out_past, new_cache_len)

        token_tensor = torch.tensor([[accepted_tokens[-1]]], device=device, dtype=dtype)
        final_logits, final_past = teacher_forward_fn(teacher_model, token_tensor, new_past)
        new_past = final_past
        next_logits = final_logits[:, -1, :]

    stats = {
        'n_matches': n_matches,
        'gamma': gamma,
        'acceptance_rate': n_matches / gamma if gamma > 0 else 0.0,
        'first_rejected': False,
    }

    return accepted_tokens, new_past, next_logits, stats


@torch.no_grad()
def lupi_skd_rollout_compiled(
    teacher_model: PreTrainedModel,
    student_model: PreTrainedModel,
    teacher_input_ids: Tensor,
    student_input_ids: Tensor,
    max_new_tokens: int,
    top_k: int,
    gamma: int,
    teacher_temperature: float,
    student_temperature: float,
    teacher_top_p: float,
    student_top_p: float,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    end_of_string_ls: Optional[List[str]] = None,
    verbose: bool = False,
    print_every_blocks: int = 10,
    use_compile: bool = True,
) -> Tuple[Tensor, Tensor, List[int], str, dict]:
    """Compiled LUPI-aware SKD rollout with optional torch.compile.

    This is the EXPERIMENTAL version. For stable usage, use lupi_skd_rollout_optimized.

    Additional Args (compared to lupi_skd_rollout_optimized):
        use_compile: Whether to use torch.compile for internal functions.
            Default is True, but can be disabled if issues occur.

    Returns:
        Same as lupi_skd_rollout_optimized.

    Note:
        - First call may be slow due to compilation overhead.
        - If compilation fails, automatically falls back to non-compiled version.
        - Set environment variable LUPI_USE_COMPILE=1 to enable globally.
    """
    device = next(teacher_model.parameters()).device
    dtype = teacher_input_ids.dtype

    teacher_ids = teacher_input_ids.to(device).clone()
    student_ids = student_input_ids.to(device).clone()

    generated_tokens: List[int] = []

    # Get compiled or non-compiled functions
    student_draft_fn, teacher_forward_fn = _get_compiled_functions(use_compile)

    # Statistics
    debug_stats = {
        'total_blocks': 0,
        'total_drafted': 0,
        'total_accepted': 0,
        'total_rejected': 0,
        'teacher_forward_calls': 0,
        'student_forward_calls': 0,
        'block_stats': [],
        'use_compile': use_compile,
    }

    # Initialize Teacher KV-cache
    teacher_pref_out = teacher_model(
        input_ids=teacher_ids,
        attention_mask=torch.ones_like(teacher_ids),
        use_cache=True,
    )
    teacher_past = teacher_pref_out.past_key_values
    teacher_next_logits = teacher_pref_out.logits[:, -1, :]
    debug_stats['teacher_forward_calls'] += 1

    block_idx = 0

    while len(generated_tokens) < max_new_tokens:
        block_idx += 1
        debug_stats['total_blocks'] = block_idx

        # ============================================================
        # 1) Student generates gamma draft tokens
        # ============================================================
        draft_ids: List[int] = []
        draft_student_ids = student_ids.clone()
        block_student_past = None
        next_student_input = draft_student_ids

        for _ in range(gamma):
            tok_val, block_student_past, next_tok = student_draft_fn(
                student_model,
                next_student_input,
                block_student_past,
                student_temperature,
                student_top_p,
            )
            debug_stats['student_forward_calls'] += 1

            draft_ids.append(tok_val)
            next_tok_2d = next_tok.view(1, 1)
            draft_student_ids = torch.cat([draft_student_ids, next_tok_2d], dim=1)
            next_student_input = next_tok_2d

            if tokenizer is not None and tok_val == tokenizer.eos_token_id:
                break
            if end_of_string_ls and tokenizer is not None:
                text = tokenizer.decode(draft_ids, skip_special_tokens=True)
                if any(stop in text for stop in end_of_string_ls):
                    break

        if len(draft_ids) == 0:
            break

        debug_stats['total_drafted'] += len(draft_ids)

        # ============================================================
        # 2) Parallel Teacher verification
        # ============================================================
        accepted_tokens, teacher_past, teacher_next_logits, verify_stats = _parallel_teacher_verify_compiled(
            teacher_model=teacher_model,
            teacher_past=teacher_past,
            first_logits=teacher_next_logits,
            draft_ids=draft_ids,
            top_k=top_k,
            teacher_temperature=teacher_temperature,
            teacher_top_p=teacher_top_p,
            device=device,
            dtype=dtype,
            teacher_forward_fn=teacher_forward_fn,
        )

        debug_stats['teacher_forward_calls'] += 1
        if verify_stats['n_matches'] < len(draft_ids):
            debug_stats['teacher_forward_calls'] += 1

        n_matches = verify_stats['n_matches']
        debug_stats['total_accepted'] += n_matches
        debug_stats['total_rejected'] += len(draft_ids) - n_matches
        debug_stats['block_stats'].append(verify_stats)

        if len(accepted_tokens) == 0:
            break

        # ============================================================
        # 3) Append accepted tokens
        # ============================================================
        new_tokens_tensor = torch.tensor(
            accepted_tokens, device=device, dtype=dtype
        ).view(1, -1)

        teacher_ids = torch.cat([teacher_ids, new_tokens_tensor], dim=1)
        student_ids = torch.cat([student_ids, new_tokens_tensor], dim=1)
        generated_tokens.extend(accepted_tokens)

        # ============================================================
        # 4) Student cache management
        # ============================================================
        if n_matches < len(draft_ids):
            if block_student_past is not None and n_matches > 0:
                student_prefix_len = block_student_past[0][0].shape[2] - len(draft_ids)
                valid_cache_len = student_prefix_len + n_matches
                block_student_past = _crop_past_key_values(block_student_past, valid_cache_len)
            else:
                block_student_past = None

        if verbose and (block_idx % print_every_blocks == 0 or len(generated_tokens) >= max_new_tokens):
            compile_str = "[COMPILED]" if use_compile else "[NON-COMPILED]"
            if tokenizer is not None:
                partial_text = tokenizer.decode(generated_tokens[-200:], skip_special_tokens=True)
            else:
                partial_text = ""
            print(f"{compile_str} [Block {block_idx}] total_generated={len(generated_tokens)}, "
                  f"n_matches={n_matches}/{len(draft_ids)}")
            if partial_text:
                print(f"  ...{partial_text}")
            print("-" * 40)

        # ============================================================
        # 5) Global stop conditions
        # ============================================================
        if tokenizer is not None and generated_tokens[-1] == tokenizer.eos_token_id:
            break
        if end_of_string_ls and tokenizer is not None:
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if any(stop in text for stop in end_of_string_ls):
                break
        if len(generated_tokens) >= max_new_tokens:
            break

    # Final decoding
    if tokenizer is not None:
        new_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        new_str = ""

    # Aggregate statistics
    if debug_stats['total_drafted'] > 0:
        debug_stats['avg_acceptance_rate'] = debug_stats['total_accepted'] / debug_stats['total_drafted']
    else:
        debug_stats['avg_acceptance_rate'] = 0.0

    return teacher_ids, student_ids, generated_tokens, new_str, debug_stats
