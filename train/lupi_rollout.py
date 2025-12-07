# coding=utf-8
"""LUPI-aware Speculative KD rollout with block speculative decoding (gamma > 1)."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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
        # Pure softmax sampling
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(0)

    # Sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # [1, V]
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative prob above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set filtered logits to -inf
    logits_filtered = logits.clone()
    logits_filtered[0, sorted_indices[sorted_indices_to_remove]] = float("-inf")

    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)


@torch.no_grad()
def lupi_skd_rollout_block(
    teacher_model: PreTrainedModel,
    student_model: PreTrainedModel,
    teacher_input_ids: Tensor,   # [1, T_teacher]  (Q+K)
    student_input_ids: Tensor,   # [1, T_student]  (Q)
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
) -> Tuple[Tensor, Tensor, List[int], str]:
    """LUPI-aware SKD rollout with block speculative decoding (gamma tokens per block).

    Args:
        teacher_model: Teacher model (sees Q+K).
        student_model: Student model (sees Q only).
        teacher_input_ids: Tokenized teacher input [1, T_teacher] (Q+K).
        student_input_ids: Tokenized student input [1, T_student] (Q only).
        max_new_tokens: Maximum number of tokens to generate.
        top_k: Top-K threshold for acceptance.
        gamma: Block size (number of draft tokens per speculative block).
        teacher_temperature: Temperature for teacher sampling (rollout only).
        student_temperature: Temperature for student sampling (rollout only).
        teacher_top_p: Teacher top-p (nucleus sampling) used when resampling.
        student_top_p: Student top-p used for draft tokens.
        tokenizer: Tokenizer instance (for optional stop-string decoding).
        end_of_string_ls: List of stop strings. If any appears in decoded text, rollout stops.

    Returns:
        teacher_full_ids: [1, T_teacher + |Y|] tensor (Q+K+Y).
        student_full_ids: [1, T_student + |Y|] tensor (Q+Y).
        generated_tokens: List[int] of generated token ids (Y).
        new_str: Decoded generated string.
    """
    # 더 안전한 device 선택 (모델이 FSDP/Accelerate일 수 있으므로)
    device = next(teacher_model.parameters()).device

    teacher_ids = teacher_input_ids.to(device).clone()   # [1, T_teacher]
    student_ids = student_input_ids.to(device).clone()   # [1, T_student]

    generated_tokens: List[int] = []

    # Teacher prefix를 한 번만 통과시켜 KV-cache와 첫 분포를 확보
    with torch.no_grad():
        teacher_pref_out = teacher_model(
            input_ids=teacher_ids,
            attention_mask=torch.ones_like(teacher_ids),
            use_cache=True,
        )
    teacher_past = teacher_pref_out.past_key_values
    teacher_next_logits = teacher_pref_out.logits[:, -1, :]  # [1, V]

    block_idx = 0
    # 메인 루프: 블록 단위로 진행
    while len(generated_tokens) < max_new_tokens:
        block_idx += 1

        # 1) 학생이 gamma개 draft 토큰 제안 (cache 사용)
        draft_ids: List[int] = []
        draft_student_ids = student_ids.clone()
        block_student_past = None
        next_student_input = draft_student_ids  # 첫 step은 전체 prefix

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

            next_tok = _sample_top_p(student_logits, student_top_p)
            tok_val = next_tok.item()
            draft_ids.append(tok_val)

            next_tok_2d = next_tok.view(1, 1)
            draft_student_ids = torch.cat([draft_student_ids, next_tok_2d], dim=1)
            next_student_input = next_tok_2d  # 이후 step은 직전 토큰만 전달

            # EOS/stop-string 체크 (옵션, 블록 안에서도 끊을 수 있게)
            if tokenizer is not None and tok_val == tokenizer.eos_token_id:
                break
            if end_of_string_ls and tokenizer is not None:
                text = tokenizer.decode(draft_ids, skip_special_tokens=True)
                if any(stop in text for stop in end_of_string_ls):
                    break

        if len(draft_ids) == 0:
            break  # 아무것도 못 뽑았으면 종료

        # 2) 교사 검증: KV-cache 기반 순차 검증
        final_tokens_this_block: List[int] = []
        block_fully_accepted = True

        for tok_id in draft_ids:
            logits_step = teacher_next_logits / max(teacher_temperature, 1e-6)
            probs_step = F.softmax(logits_step, dim=-1)
            topk_probs, topk_idx = torch.topk(probs_step, k=min(top_k, probs_step.shape[-1]))

            if (topk_idx == tok_id).any().item():
                accepted_token = tok_id
            else:
                block_fully_accepted = False
                replacement = _sample_top_p(logits_step, teacher_top_p)
                accepted_token = replacement.item()

            token_tensor = torch.tensor([[accepted_token]], device=device, dtype=teacher_ids.dtype)
            with torch.no_grad():
                teacher_step_out = teacher_model(
                    input_ids=token_tensor,
                    attention_mask=torch.ones_like(token_tensor),
                    past_key_values=teacher_past,
                    use_cache=True,
                )
            teacher_past = teacher_step_out.past_key_values
            teacher_next_logits = teacher_step_out.logits[:, -1, :]

            final_tokens_this_block.append(accepted_token)

            if not block_fully_accepted:
                break

        if len(final_tokens_this_block) == 0:
            break

        # 3) accept된/교체된 토큰들을 prefix에 한 번에 append
        new_tokens_tensor = torch.tensor(
            final_tokens_this_block, device=device, dtype=teacher_ids.dtype
        ).view(1, -1)  # [1, L_final]

        teacher_ids = torch.cat([teacher_ids, new_tokens_tensor], dim=1)
        student_ids = torch.cat([student_ids, new_tokens_tensor], dim=1)
        generated_tokens.extend(final_tokens_this_block)

        if verbose and (block_idx % print_every_blocks == 0 or len(generated_tokens) >= max_new_tokens):
            if tokenizer is not None:
                partial_text = tokenizer.decode(generated_tokens[-200:], skip_special_tokens=True)
            else:
                partial_text = ""
            print(f"[Block {block_idx}] total_generated={len(generated_tokens)}")
            if partial_text:
                print(f"  ...{partial_text}")
            print("-" * 40)

        # replacement가 있었다면 학생 cache는 무효 처리
        if not block_fully_accepted:
            block_student_past = None

        # 4) global stop 조건
        if tokenizer is not None and generated_tokens[-1] == tokenizer.eos_token_id:
            break
        if end_of_string_ls and tokenizer is not None:
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if any(stop in text for stop in end_of_string_ls):
                break
        if len(generated_tokens) >= max_new_tokens:
            break

    # 최종 디코딩
    if tokenizer is not None:
        new_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        new_str = ""

    return teacher_ids, student_ids, generated_tokens, new_str
