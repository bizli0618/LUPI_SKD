# coding=utf-8
"""
Deep Debugging Module for LUPI-SKD.
Provides token-level granularity on Student's drafting and Teacher's verification process.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

# ANSI Colors for visibility
COLOR_RESET = "\033[0m"
COLOR_STUDENT = "\033[94m"  # Blue
COLOR_TEACHER = "\033[92m"  # Green
COLOR_REJECT = "\033[91m"   # Red
COLOR_ACCEPT = "\033[96m"   # Cyan
COLOR_WARN = "\033[93m"     # Yellow

def _crop_past_key_values(past_key_values, max_length: int):
    if past_key_values is None: return None
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
def lupi_skd_rollout_adaptive_debug(
    teacher_model: PreTrainedModel,
    student_model: PreTrainedModel,
    teacher_input_ids: torch.Tensor,
    student_input_ids: torch.Tensor,
    max_new_tokens: int,
    top_k: int,
    gamma: int,
    teacher_temperature: float,
    student_temperature: float,
    teacher_top_p: float,
    student_top_p: float,
    teacher_min_prob: float = 0.0,
    tokenizer: PreTrainedTokenizerBase = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], str, dict]:
    """
    A heavy-weight debugging rollout function.
    Logs extensive details about token probabilities, ranks, and choices.
    """
    device = next(teacher_model.parameters()).device
    teacher_ids = teacher_input_ids.to(device).clone()
    student_ids = student_input_ids.to(device).clone()
    
    generated_tokens = []
    
    # Initialize Teacher
    t_out = teacher_model(input_ids=teacher_ids, use_cache=True)
    teacher_past = t_out.past_key_values
    teacher_logits = t_out.logits[:, -1, :]
    
    current_gamma = gamma
    step_count = 0
    
    print(f"\n{COLOR_WARN}=== STARTING DEEP DEBUG ROLLOUT (Top-10 Analysis) ==={COLOR_RESET}")
    print(f"Initial Gamma: {gamma}, Top-K: {top_k}, Min-Prob: {teacher_min_prob}")

    while len(generated_tokens) < max_new_tokens:
        step_count += 1
        print(f"\n{COLOR_WARN}[Step {step_count}] Current Gamma: {current_gamma}{COLOR_RESET}")
        
        # --- 1. Student Draft Phase ---
        draft_tokens = []
        
        s_input = student_ids
        s_past = None
        
        print(f"  {COLOR_STUDENT}Student Drafting...{COLOR_RESET}")
        
        for i in range(current_gamma):
            s_kwargs = {"input_ids": s_input, "use_cache": True}
            if s_past is not None: s_kwargs["past_key_values"] = s_past
            
            s_out = student_model(**s_kwargs)
            s_past = s_out.past_key_values
            s_logits = s_out.logits[:, -1, :] / max(student_temperature, 1e-6)
            
            # Debugging Student Choice
            s_probs_all = F.softmax(s_logits, dim=-1)
            s_top10 = torch.topk(s_probs_all, k=10, dim=-1)
            
            # Sampling
            if student_top_p < 1.0:
                # Nucleus sampling logic (simplified for debug)
                sorted_logits, sorted_indices = torch.sort(s_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                start_idx = (cum_probs > student_top_p).nonzero()
                if start_idx.numel() > 0:
                    idx = start_idx[0, 1].item()
                    sorted_logits[0, idx+1:] = float('-inf')
                sampled_idx = torch.multinomial(F.softmax(sorted_logits, dim=-1), 1)
                next_token = sorted_indices[0, sampled_idx.item()].item()
            else:
                next_token = torch.multinomial(s_probs_all, 1).item()
            
            draft_tokens.append(next_token)
            
            # Log Student Info
            token_str = tokenizer.decode([next_token]).replace('\n', '\\n')
            token_prob = s_probs_all[0, next_token].item()
            token_rank = (s_probs_all[0] > token_prob).sum().item() + 1
            
            print(f"    Draft[{i}]: '{token_str}' (ID:{next_token}) | Prob: {token_prob:.4f} | Rank: {token_rank}")
            
            # Detailed Top-10 Candidates
            cand_strs = []
            for idx in range(10):
                tid = s_top10.indices[0][idx].item()
                tprob = s_top10.values[0][idx].item()
                tstr = tokenizer.decode([tid]).replace('\n','\\n').replace(' ','')
                cand_strs.append(f"{tstr}({tprob:.2f})")
            print(f"      -> Top-10: {', '.join(cand_strs)}")
            
            s_input = torch.tensor([[next_token]], device=device)
            
            # Stop tokens
            if next_token == tokenizer.eos_token_id:
                break
        
        # --- 2. Teacher Verification Phase ---
        print(f"  {COLOR_TEACHER}Teacher Verifying...{COLOR_RESET}")
        
        # Run Teacher on draft
        draft_tensor = torch.tensor([draft_tokens], device=device)
        t_draft_out = teacher_model(input_ids=draft_tensor, past_key_values=teacher_past, use_cache=True)
        
        # Check verification
        check_logits = torch.cat([teacher_logits.unsqueeze(1), t_draft_out.logits[:, :-1, :]], dim=1)
        
        matches = 0
        reject_info = None
        
        for i, token_id in enumerate(draft_tokens):
            step_logits = check_logits[0, i, :] / max(teacher_temperature, 1e-6)
            t_probs = F.softmax(step_logits, dim=-1)
            
            # Check Top-K
            t_topk = torch.topk(t_probs, k=top_k, dim=-1)
            t_top10 = torch.topk(t_probs, k=10, dim=-1) # For debug view
            
            is_in_topk = token_id in t_topk.indices.tolist()
            t_prob = t_probs[token_id].item()
            is_prob_sufficient = t_prob >= teacher_min_prob
            
            is_accepted = is_in_topk and is_prob_sufficient
            
            # Get stats
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
            t_rank = (t_probs > t_prob).sum().item() + 1
            top1_id = t_topk.indices[0].item()
            top1_str = tokenizer.decode([top1_id]).replace('\n', '\\n')
            
            if is_accepted:
                matches += 1
                print(f"    {COLOR_ACCEPT}✓ Pos[{i}]{COLOR_RESET} '{token_str}' ACCEPTED | T.Rank: {t_rank} | T.Prob: {t_prob:.4f}")
            else:
                reason = "Rank" if not is_in_topk else "Low Prob"
                print(f"    {COLOR_REJECT}✗ Pos[{i}]{COLOR_RESET} '{token_str}' REJECTED ({reason}) | T.Rank: {t_rank} | T.Prob: {t_prob:.4f}")
                print(f"      -> Teacher Top-1: '{top1_str}' (Prob: {t_probs[top1_id]:.4f})")
                
                # Detailed Top-10 Teacher Preferences
                t_cand_strs = []
                for idx in range(10):
                    tid = t_top10.indices[idx].item()
                    tprob = t_top10.values[idx].item()
                    tstr = tokenizer.decode([tid]).replace('\n','\\n').replace(' ','')
                    
                    # Mark acceptance status
                    mark = ""
                    if tid in t_topk.indices.tolist():
                        if tprob >= teacher_min_prob:
                            mark = "*" # Accepted
                        else:
                            mark = "~" # Top-K but Low Prob
                    
                    t_cand_strs.append(f"{mark}{tstr}({tprob:.2f})")
                print(f"      -> Teacher Top-10 (*=OK, ~=LowProb): {', '.join(t_cand_strs)}")
                
                reject_info = (i, step_logits)
                break
        
        # --- 3. Update & Correction ---
        if matches == len(draft_tokens):
            accepted = draft_tokens
            generated_tokens.extend(accepted)
            teacher_past = t_draft_out.past_key_values
            teacher_logits = t_draft_out.logits[:, -1, :]
            
            new_gamma = min(gamma, current_gamma + 2)
            print(f"  {COLOR_ACCEPT}All Matched! Increasing Gamma {current_gamma} -> {new_gamma}{COLOR_RESET}")
            current_gamma = new_gamma
            
            acc_tensor = torch.tensor([accepted], device=device)
            teacher_ids = torch.cat([teacher_ids, acc_tensor], dim=1)
            student_ids = torch.cat([student_ids, acc_tensor], dim=1)
            
        else:
            idx, rejected_logits = reject_info
            accepted = draft_tokens[:idx]
            
            sorted_logits, sorted_indices = torch.sort(rejected_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            start_idx = (cum_probs > teacher_top_p).nonzero()
            if start_idx.numel() > 0:
                cut = start_idx[0, 0].item()
                sorted_logits[cut+1:] = float('-inf')
            
            sampled_rank = torch.multinomial(F.softmax(sorted_logits, dim=-1), 1).item()
            correction_idx = sorted_indices[sampled_rank].item()
            correction_str = tokenizer.decode([correction_idx]).replace('\n', '\\n')
            
            print(f"  {COLOR_REJECT}Correction:{COLOR_RESET} '{correction_str}' (replacing rejected token)")
            
            accepted.append(correction_idx)
            generated_tokens.extend(accepted)
            
            prefix_len = teacher_ids.shape[1]
            if matches > 0:
                cropped_past = _crop_past_key_values(t_draft_out.past_key_values, prefix_len + matches)
            else:
                cropped_past = teacher_past 
            
            corr_tensor = torch.tensor([[correction_idx]], device=device)
            t_corr_out = teacher_model(input_ids=corr_tensor, past_key_values=cropped_past, use_cache=True)
            
            teacher_past = t_corr_out.past_key_values
            teacher_logits = t_corr_out.logits[:, -1, :]
            
            acc_tensor = torch.tensor([accepted], device=device)
            teacher_ids = torch.cat([teacher_ids, acc_tensor], dim=1)
            student_ids = torch.cat([student_ids, acc_tensor], dim=1)
            
            new_gamma = max(1, matches + 1)
            print(f"  {COLOR_REJECT}Mismatch! Decreasing Gamma {current_gamma} -> {new_gamma}{COLOR_RESET}")
            current_gamma = new_gamma

        if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id:
            print("EOS Token Generated. Stopping.")
            break
            
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return teacher_ids, student_ids, generated_tokens, decoded_text, {}
