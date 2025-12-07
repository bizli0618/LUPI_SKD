# coding=utf-8
"""Optimized LUPI-aware Speculative KD rollout with parallel Teacher verification.

This is the stable version WITHOUT torch.compile.
For the experimental version with torch.compile, see lupi_rollout_compiled.py.

Key optimizations:
1. Parallel Teacher verification: gamma tokens verified in 1 forward pass
2. KV-Cache crop: preserve valid cache on rejection
3. Vectorized acceptance check: batch top-K membership test

Reference implementations:
- transformers/utils.py:3789-3816 (parallel verification)
- transformers/utils.py:4033-4107 (acceptance logic)
- transformers/candidate_generator.py:361-409 (KV-cache crop)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache


def _crop_past_key_values(past_key_values, max_length: int):
    """Crop KV-cache to a specified maximum length.

    Reference: transformers/candidate_generator.py:361-409

    Args:
        past_key_values: DynamicCache or Tuple of (key, value) tensors per layer.
            Each tensor has shape [batch, heads, seq_len, head_dim].
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
    """Top-p (nucleus) sampling for a single step.

    Args:
        logits: [1, V] logit tensor for the last token position.
        top_p: nucleus probability threshold in (0, 1].

    Returns:
        next_token_id: [1] int64 tensor with sampled token id.
    """
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


@torch.no_grad()
def _parallel_teacher_verify(
    teacher_model: PreTrainedModel,
    teacher_past: Tuple,
    first_logits: Tensor,
    draft_ids: List[int],
    top_k: int,
    teacher_temperature: float,
    teacher_top_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[int], Tuple, Tensor, dict]:
    """Verify gamma draft tokens with parallel Teacher forward pass.

    Reference: transformers/utils.py:3789-3816, 4033-4107

    Args:
        teacher_model: Teacher model for verification.
        teacher_past: Current Teacher KV-cache.
        first_logits: [1, V] logits for verifying the first draft token.
        draft_ids: List of gamma draft token ids from Student.
        top_k: Top-K acceptance threshold.
        teacher_temperature: Temperature for Teacher probability distribution.
        teacher_top_p: Top-p for Teacher resampling on rejection.
        device: Target device.
        dtype: Target dtype.

    Returns:
        accepted_tokens: List[int] - Final accepted token ids.
        new_teacher_past: Updated Teacher KV-cache.
        next_logits: [1, V] logits for next block.
        stats: dict with debugging info (n_matches, gamma, acceptance_rate).
    """
    gamma = len(draft_ids)

    # 1) Verify first token using first_logits (already computed)
    probs_0 = F.softmax(first_logits / max(teacher_temperature, 1e-6), dim=-1)
    topk_idx_0 = torch.topk(probs_0, k=min(top_k, probs_0.shape[-1])).indices[0]

    first_accepted = draft_ids[0] in topk_idx_0.tolist()

    if not first_accepted:
        # First token rejected → Teacher resamples
        replacement = _sample_top_p(first_logits / max(teacher_temperature, 1e-6), teacher_top_p)
        accepted_token = replacement.item()

        # Advance Teacher by one step
        token_tensor = torch.tensor([[accepted_token]], device=device, dtype=dtype)
        out = teacher_model(
            input_ids=token_tensor,
            attention_mask=torch.ones_like(token_tensor),
            past_key_values=teacher_past,
            use_cache=True,
        )
        stats = {
            'n_matches': 0,
            'gamma': gamma,
            'acceptance_rate': 0.0,
            'first_rejected': True,
        }
        return [accepted_token], out.past_key_values, out.logits[:, -1, :], stats

    # 2) Parallel forward pass for all draft tokens
    # Input: [d1, d2, ..., d_gamma] with KV-cache containing prefix
    # Output logits structure:
    #   all_logits[0, i] = P(next | prefix, d1, ..., d_{i+1})
    # So:
    #   all_logits[0, 0] = P(d2 | prefix, d1)     → for verifying d2
    #   all_logits[0, 1] = P(d3 | prefix, d1, d2) → for verifying d3
    #   ...
    #   all_logits[0, gamma-2] = P(d_gamma | prefix, d1, ..., d_{gamma-1}) → for verifying d_gamma
    #   all_logits[0, gamma-1] = P(next | prefix, d1, ..., d_gamma)        → bonus/next_logits
    draft_tensor = torch.tensor([draft_ids], device=device, dtype=dtype)

    teacher_out = teacher_model(
        input_ids=draft_tensor,
        attention_mask=torch.ones_like(draft_tensor),
        past_key_values=teacher_past,
        use_cache=True,
    )
    # all_logits: [1, gamma, V]
    all_logits = teacher_out.logits

    # 3) Vectorized acceptance check
    # - d1 was already verified using first_logits (passed check to reach here)
    # - d2, d3, ..., d_gamma need to be verified using all_logits[0, 0:-1]
    # Reference: transformers/utils.py:4062 uses p[0, :-1, :] (excludes last for bonus token)
    if gamma > 1:
        # Logits for verifying d2, d3, ..., d_gamma
        # all_logits[0, 0] verifies d2, all_logits[0, 1] verifies d3, etc.
        verify_logits = all_logits[0, :-1, :]  # [gamma-1, V]
        probs = F.softmax(verify_logits / max(teacher_temperature, 1e-6), dim=-1)

        effective_top_k = min(top_k, probs.shape[-1])
        topk_indices = torch.topk(probs, k=effective_top_k, dim=-1).indices  # [gamma-1, top_k]

        # Tokens to verify: draft_ids[1:] = [d2, d3, ..., d_gamma]
        draft_ids_to_verify = torch.tensor(draft_ids[1:], device=device, dtype=torch.long).unsqueeze(1)  # [gamma-1, 1]
        is_accepted_rest = (topk_indices == draft_ids_to_verify).any(dim=1)  # [gamma-1]

        # Combine: d1 (already accepted) + rest
        is_accepted = torch.cat([
            torch.tensor([True], device=device, dtype=torch.bool),
            is_accepted_rest
        ])  # [gamma]
    else:
        # gamma == 1: only d1, already verified with first_logits
        is_accepted = torch.tensor([True], device=device, dtype=torch.bool)

    # 4) Find first rejection position
    # n_matches = number of consecutively accepted tokens (starting from d1)
    rejection_mask = ~is_accepted
    cumsum = rejection_mask.cumsum(dim=0)
    n_matches = int((cumsum == 0).sum().item())

    # 5) Determine final tokens
    if n_matches == gamma:
        # All tokens accepted
        accepted_tokens = list(draft_ids)
        new_past = teacher_out.past_key_values
        next_logits = all_logits[:, -1, :]  # bonus token position
    else:
        # n_matches tokens accepted (d1, ..., d_{n_matches})
        # Rejection at d_{n_matches+1}
        accepted_tokens = list(draft_ids[:n_matches])

        # Teacher resampling from P(d_{n_matches+1} | prefix, d1, ..., d_{n_matches})
        # This is all_logits[0, n_matches-1] since:
        #   all_logits[0, 0] = P(d2|prefix,d1) for n_matches=1
        #   all_logits[0, 1] = P(d3|prefix,d1,d2) for n_matches=2
        # Reference: transformers/utils.py:4091 uses p[:, n_matches, :]
        # In our case, equivalent is all_logits[0, n_matches-1] because:
        #   - SKD's new_logits[:, 0] = P(d1|prefix) = our first_logits
        #   - SKD's new_logits[:, i] = our all_logits[0, i-1] for i > 0
        resample_logits = all_logits[0, n_matches - 1, :].unsqueeze(0)
        replacement = _sample_top_p(resample_logits / max(teacher_temperature, 1e-6), teacher_top_p)
        accepted_tokens.append(replacement.item())

        # Crop KV-cache: keep only prefix + accepted tokens (d1, ..., d_{n_matches})
        # The parallel forward added all gamma tokens to cache, we need to remove rejected ones
        # Reference: transformers/candidate_generator.py:361-409
        if isinstance(teacher_past, DynamicCache):
            prefix_len = teacher_past.get_seq_length()
        else:
            prefix_len = teacher_past[0][0].shape[2]
        new_cache_len = prefix_len + n_matches  # Only accepted tokens
        new_past = _crop_past_key_values(teacher_out.past_key_values, new_cache_len)

        # Get next logits by advancing Teacher with the resampled token
        token_tensor = torch.tensor([[accepted_tokens[-1]]], device=device, dtype=dtype)
        final_out = teacher_model(
            input_ids=token_tensor,
            attention_mask=torch.ones_like(token_tensor),
            past_key_values=new_past,
            use_cache=True,
        )
        new_past = final_out.past_key_values
        next_logits = final_out.logits[:, -1, :]

    stats = {
        'n_matches': n_matches,
        'gamma': gamma,
        'acceptance_rate': n_matches / gamma if gamma > 0 else 0.0,
        'first_rejected': False,
    }

    return accepted_tokens, new_past, next_logits, stats


@torch.no_grad()
def lupi_skd_rollout_optimized(
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
) -> Tuple[Tensor, Tensor, List[int], str, dict]:
    """Optimized LUPI-aware SKD rollout with parallel Teacher verification.

    Key optimizations vs lupi_rollout.py:
    1. Parallel Teacher verification: O(1) forward passes instead of O(gamma)
    2. KV-Cache preservation: crop instead of full discard on rejection
    3. Vectorized acceptance check: batch top-K membership test

    Args:
        teacher_model: Teacher model (sees Q+K).
        student_model: Student model (sees Q only).
        teacher_input_ids: Tokenized teacher input [1, T_teacher] (Q+K).
        student_input_ids: Tokenized student input [1, T_student] (Q only).
        max_new_tokens: Maximum number of tokens to generate.
        top_k: Top-K threshold for acceptance.
        gamma: Block size (number of draft tokens per speculative block).
        teacher_temperature: Temperature for teacher sampling.
        student_temperature: Temperature for student sampling.
        teacher_top_p: Teacher top-p used when resampling.
        student_top_p: Student top-p used for draft tokens.
        tokenizer: Tokenizer instance (for stop-string decoding).
        end_of_string_ls: List of stop strings.
        verbose: Whether to print progress.
        print_every_blocks: Print interval when verbose=True.

    Returns:
        teacher_full_ids: [1, T_teacher + |Y|] tensor (Q+K+Y).
        student_full_ids: [1, T_student + |Y|] tensor (Q+Y).
        generated_tokens: List[int] of generated token ids (Y).
        new_str: Decoded generated string.
        debug_stats: dict with statistics for debugging.
    """
    device = next(teacher_model.parameters()).device
    dtype = teacher_input_ids.dtype

    teacher_ids = teacher_input_ids.to(device).clone()
    student_ids = student_input_ids.to(device).clone()

    generated_tokens: List[int] = []

    # Statistics for debugging
    debug_stats = {
        'total_blocks': 0,
        'total_drafted': 0,
        'total_accepted': 0,
        'total_rejected': 0,
        'teacher_forward_calls': 0,
        'student_forward_calls': 0,
        'block_stats': [],
    }

    # Initialize Teacher KV-cache with prefix
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
            student_kwargs = {
                "input_ids": next_student_input,
                "attention_mask": torch.ones_like(next_student_input),
                "use_cache": True,
            }
            if block_student_past is not None:
                student_kwargs["past_key_values"] = block_student_past

            student_out = student_model(**student_kwargs)
            block_student_past = student_out.past_key_values
            student_logits = student_out.logits[:, -1, :] / max(student_temperature, 1e-6)
            debug_stats['student_forward_calls'] += 1

            next_tok = _sample_top_p(student_logits, student_top_p)
            tok_val = next_tok.item()
            draft_ids.append(tok_val)

            next_tok_2d = next_tok.view(1, 1)
            draft_student_ids = torch.cat([draft_student_ids, next_tok_2d], dim=1)
            next_student_input = next_tok_2d

            # Early stop on EOS
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
        accepted_tokens, teacher_past, teacher_next_logits, verify_stats = _parallel_teacher_verify(
            teacher_model=teacher_model,
            teacher_past=teacher_past,
            first_logits=teacher_next_logits,
            draft_ids=draft_ids,
            top_k=top_k,
            teacher_temperature=teacher_temperature,
            teacher_top_p=teacher_top_p,
            device=device,
            dtype=dtype,
        )

        # Update forward call count (1 batch call + possible 1-2 single calls)
        debug_stats['teacher_forward_calls'] += 1
        if verify_stats['n_matches'] < len(draft_ids):
            debug_stats['teacher_forward_calls'] += 1  # Resample step

        n_matches = verify_stats['n_matches']
        debug_stats['total_accepted'] += n_matches
        debug_stats['total_rejected'] += len(draft_ids) - n_matches
        debug_stats['block_stats'].append(verify_stats)

        if len(accepted_tokens) == 0:
            break

        # ============================================================
        # 3) Append accepted tokens to prefixes
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
        # If rejection occurred, crop Student cache to valid length
        if n_matches < len(draft_ids):
            if block_student_past is not None and n_matches > 0:
                # Student cache has: prefix + gamma tokens
                # Valid: prefix + n_matches tokens
                student_prefix_len = block_student_past[0][0].shape[2] - len(draft_ids)
                valid_cache_len = student_prefix_len + n_matches
                block_student_past = _crop_past_key_values(block_student_past, valid_cache_len)
            else:
                # No valid draft tokens or first rejected
                block_student_past = None

        if verbose and (block_idx % print_every_blocks == 0 or len(generated_tokens) >= max_new_tokens):
            if tokenizer is not None:
                partial_text = tokenizer.decode(generated_tokens[-200:], skip_special_tokens=True)
            else:
                partial_text = ""
            print(f"[Block {block_idx}] total_generated={len(generated_tokens)}, "
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

    # Compute aggregate statistics
    if debug_stats['total_drafted'] > 0:
        debug_stats['avg_acceptance_rate'] = debug_stats['total_accepted'] / debug_stats['total_drafted']
    else:
        debug_stats['avg_acceptance_rate'] = 0.0

    return teacher_ids, student_ids, generated_tokens, new_str, debug_stats
