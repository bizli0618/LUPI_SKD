# coding=utf-8
"""Adaptive Gamma LUPI-aware Speculative KD rollout.

This module implements 'Adaptive Gamma' (Dynamic Speculative Block Size)
to solve the issue of low acceptance rates during early training.

Key Logic:
- Dynamically adjusts the number of draft tokens (`current_gamma`) based on
  the Student's success rate in the previous block.
- If Student matches all tokens -> Increase gamma (Trust Student).
- If Student fails early -> Decrease gamma (Restrict Student).

Based on lupi_rollout_optimized.py.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache


def _crop_past_key_values(past_key_values, max_length: int):
    """Crop KV-cache to a specified maximum length."""
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        past_key_values.crop(max_length)
        return past_key_values

    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append((
            past_key_values[idx][0][:, :, :max_length, :],
            past_key_values[idx][1][:, :, :max_length, :],
        ))
    return tuple(new_past)


@torch.no_grad()
def _sample_top_p(logits: Tensor, top_p: float) -> Tensor:
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


@torch.no_grad()
def _parallel_teacher_verify(
    teacher_model: PreTrainedModel,
    teacher_past: Tuple,
    first_logits: Tensor,
    draft_ids: List[int],
    top_k: int,
    teacher_temperature: float,
    teacher_top_p: float,
    teacher_min_prob: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[int], Tuple, Tensor, dict]:
    """Verify gamma draft tokens with parallel Teacher forward pass."""
    gamma = len(draft_ids)

    # 1) Verify first token using first_logits
    probs_0 = F.softmax(first_logits / max(teacher_temperature, 1e-6), dim=-1)
    topk_idx_0 = torch.topk(probs_0, k=min(top_k, probs_0.shape[-1])).indices[0]

    is_in_topk_0 = draft_ids[0] in topk_idx_0.tolist()
    token_prob_0 = probs_0[0, draft_ids[0]].item()
    is_prob_sufficient_0 = token_prob_0 >= teacher_min_prob

    first_accepted = is_in_topk_0 and is_prob_sufficient_0

    if not first_accepted:
        replacement = _sample_top_p(first_logits / max(teacher_temperature, 1e-6), teacher_top_p)
        accepted_token = replacement.item()

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
            'correction': accepted_token, # Debug info
            'rejected_token': draft_ids[0] # Debug info
        }
        return [accepted_token], out.past_key_values, out.logits[:, -1, :], stats

    # 2) Parallel forward pass
    draft_tensor = torch.tensor([draft_ids], device=device, dtype=dtype)
    teacher_out = teacher_model(
        input_ids=draft_tensor,
        attention_mask=torch.ones_like(draft_tensor),
        past_key_values=teacher_past,
        use_cache=True,
    )
    all_logits = teacher_out.logits

    # 3) Vectorized acceptance check
    if gamma > 1:
        verify_logits = all_logits[0, :-1, :]
        probs = F.softmax(verify_logits / max(teacher_temperature, 1e-6), dim=-1)
        effective_top_k = min(top_k, probs.shape[-1])
        topk_indices = torch.topk(probs, k=effective_top_k, dim=-1).indices

        draft_ids_to_verify = torch.tensor(draft_ids[1:], device=device, dtype=torch.long).unsqueeze(1)
        
        # Check Top-K
        is_in_topk = (topk_indices == draft_ids_to_verify).any(dim=1)
        
        # Check Probability Threshold
        token_probs = probs.gather(1, draft_ids_to_verify).squeeze(1)
        is_prob_sufficient = token_probs >= teacher_min_prob
        
        is_accepted_rest = is_in_topk & is_prob_sufficient

        is_accepted = torch.cat([
            torch.tensor([True], device=device, dtype=torch.bool),
            is_accepted_rest
        ])
    else:
        is_accepted = torch.tensor([True], device=device, dtype=torch.bool)

    # 4) Find first rejection
    rejection_mask = ~is_accepted
    cumsum = rejection_mask.cumsum(dim=0)
    n_matches = int((cumsum == 0).sum().item())

    # 5) Determine final tokens
    if n_matches == gamma:
        accepted_tokens = list(draft_ids)
        new_past = teacher_out.past_key_values
        next_logits = all_logits[:, -1, :]
        correction = None
        rejected_token = None
    else:
        accepted_tokens = list(draft_ids[:n_matches])
        # Resample from all_logits[0, n_matches - 1] (distribution for the failed position)
        resample_logits = all_logits[0, n_matches - 1, :].unsqueeze(0)
        replacement = _sample_top_p(resample_logits / max(teacher_temperature, 1e-6), teacher_top_p)
        accepted_tokens.append(replacement.item())
        
        correction = replacement.item()
        rejected_token = draft_ids[n_matches]

        if isinstance(teacher_past, DynamicCache):
            prefix_len = teacher_past.get_seq_length()
        else:
            prefix_len = teacher_past[0][0].shape[2]
        new_cache_len = prefix_len + n_matches
        new_past = _crop_past_key_values(teacher_out.past_key_values, new_cache_len)

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
        'correction': correction,
        'rejected_token': rejected_token
    }

    return accepted_tokens, new_past, next_logits, stats


@torch.no_grad()
def lupi_skd_rollout_adaptive(
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
    teacher_min_prob: float = 0.0,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    end_of_string_ls: Optional[List[str]] = None,
    verbose: bool = False,
    print_every_blocks: int = 10,
) -> Tuple[Tensor, Tensor, List[int], str, dict]:
    """Adaptive Gamma LUPI-aware SKD rollout with detailed debugging."""
    # Detect devices independently
    teacher_device = next(teacher_model.parameters()).device
    student_device = next(student_model.parameters()).device
    dtype = teacher_input_ids.dtype

    # Initial placement
    teacher_ids = teacher_input_ids.to(teacher_device).clone()
    student_ids = student_input_ids.to(student_device).clone()

    generated_tokens: List[int] = []

    # Statistics
    debug_stats = {
        'total_blocks': 0,
        'total_drafted': 0,
        'total_accepted': 0,
        'total_rejected': 0,
        'teacher_forward_calls': 0,
        'student_forward_calls': 0,
        'block_stats': [],
        'gamma_history': [],
    }

    # Initialize Teacher
    teacher_pref_out = teacher_model(
        input_ids=teacher_ids,
        attention_mask=torch.ones_like(teacher_ids),
        use_cache=True,
    )
    teacher_past = teacher_pref_out.past_key_values
    teacher_next_logits = teacher_pref_out.logits[:, -1, :]
    debug_stats['teacher_forward_calls'] += 1

    # --- ADAPTIVE GAMMA STATE ---
    max_gamma = gamma
    current_gamma = max_gamma
    
    block_idx = 0

    if verbose:
        print(f"\n[Start Rollout] Max Gamma: {max_gamma}, Top-K: {top_k}")

    while len(generated_tokens) < max_new_tokens:
        block_idx += 1
        debug_stats['total_blocks'] = block_idx
        debug_stats['gamma_history'].append(current_gamma)

        # 1) Student generates 'current_gamma' draft tokens
        draft_ids: List[int] = []
        draft_student_ids = student_ids.clone()
        block_student_past = None
        next_student_input = draft_student_ids

        for _ in range(current_gamma):
            student_kwargs = {
                "input_ids": next_student_input, # Already on student_device
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

            next_tok_2d = next_tok.view(1, 1) # On student_device
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

        # 2) Parallel Teacher Verification
        # Verification happens on Teacher Device
        accepted_tokens, teacher_past, teacher_next_logits, verify_stats = _parallel_teacher_verify(
            teacher_model=teacher_model,
            teacher_past=teacher_past,
            first_logits=teacher_next_logits,
            draft_ids=draft_ids,
            top_k=top_k,
            teacher_temperature=teacher_temperature,
            teacher_top_p=teacher_top_p,
            teacher_min_prob=teacher_min_prob,
            device=teacher_device,
            dtype=dtype,
        )

        debug_stats['teacher_forward_calls'] += 1
        if verify_stats['n_matches'] < len(draft_ids):
            debug_stats['teacher_forward_calls'] += 1

        n_matches = verify_stats['n_matches']
        debug_stats['total_accepted'] += n_matches
        debug_stats['total_rejected'] += len(draft_ids) - n_matches
        debug_stats['block_stats'].append(verify_stats)

        # Detailed Debugging
        if verbose and (block_idx % print_every_blocks == 0 or verify_stats['correction'] is not None):
            should_log = True # Always log if there's a correction or interval met
            if should_log:
                draft_text = tokenizer.decode(draft_ids, skip_special_tokens=False) if tokenizer else str(draft_ids)
                
                status_icon = "✅" if n_matches == len(draft_ids) else "❌"
                print(f"[Block {block_idx}] Gamma: {len(draft_ids)} | Match: {n_matches}/{len(draft_ids)} {status_icon}")
                print(f"  Draft: '{draft_text}'")
                
                if verify_stats['correction'] is not None:
                    # Correction happened
                    rejected_txt = tokenizer.decode([verify_stats['rejected_token']], skip_special_tokens=False) if tokenizer else str(verify_stats['rejected_token'])
                    corrected_txt = tokenizer.decode([verify_stats['correction']], skip_special_tokens=False) if tokenizer else str(verify_stats['correction'])
                    print(f"  ⚠️ REJECTED at pos {n_matches}: '{rejected_txt}' -> CORRECTED to '{corrected_txt}'")
                
        # 3) ADAPTIVE GAMMA UPDATE
        prev_gamma = current_gamma
        if n_matches == len(draft_ids):
            current_gamma = min(max_gamma, current_gamma + 2)
        else:
            current_gamma = max(1, n_matches + 1)
        
        if verbose and prev_gamma != current_gamma:
             print(f"  🔄 Adaptive Gamma: {prev_gamma} -> {current_gamma}")

        if len(accepted_tokens) == 0:
            break

        # 4) Update State
        # Create tensor on Teacher device first (source of truth)
        new_tokens_tensor_t = torch.tensor(accepted_tokens, device=teacher_device, dtype=dtype).view(1, -1)
        
        # Move to Student device for Student history
        new_tokens_tensor_s = new_tokens_tensor_t.to(student_device)
        
        teacher_ids = torch.cat([teacher_ids, new_tokens_tensor_t], dim=1)
        student_ids = torch.cat([student_ids, new_tokens_tensor_s], dim=1)
        
        generated_tokens.extend(accepted_tokens)

        # Student cache management (Need to fix cache on Student Device)
        if n_matches < len(draft_ids):
            if block_student_past is not None and n_matches > 0:
                # We need to crop Student Cache.
                # Student cache is on student_device.
                # Check draft length and match length.
                
                # Logic:
                # Student generated 'gamma' tokens.
                # We accepted 'n_matches'.
                # The cache has Grown by 'gamma'.
                # We want to keep 'prefix + n_matches'.
                
                # Get prefix len from cache structure
                if isinstance(block_student_past, DynamicCache):
                     current_len = block_student_past.get_seq_length()
                else:
                     current_len = block_student_past[0][0].shape[2]
                
                # current_len includes the full draft.
                # We want to roll back to (current_len - gamma + n_matches).
                valid_cache_len = current_len - len(draft_ids) + n_matches
                
                block_student_past = _crop_past_key_values(block_student_past, valid_cache_len)
            else:
                block_student_past = None

        # Global stop
        if tokenizer is not None and generated_tokens[-1] == tokenizer.eos_token_id:
            break
        if end_of_string_ls and tokenizer is not None:
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if any(stop in text for stop in end_of_string_ls):
                break
        if len(generated_tokens) >= max_new_tokens:
            break

    new_str = tokenizer.decode(generated_tokens, skip_special_tokens=True) if tokenizer else ""

    if debug_stats['total_drafted'] > 0:
        debug_stats['avg_acceptance_rate'] = debug_stats['total_accepted'] / debug_stats['total_drafted']
    else:
        debug_stats['avg_acceptance_rate'] = 0.0

    return teacher_ids, student_ids, generated_tokens, new_str, debug_stats