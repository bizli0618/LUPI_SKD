import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import wandb
except ImportError:
    class DummyWandb:
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
    wandb = DummyWandb()
from tqdm import tqdm

from train.ddp_lupi_dataset import LUPIDataset, LUPICollator
from train.lupi_rollout_adaptive import lupi_skd_rollout_adaptive

@dataclass
class LUPISKDConfig:
    # Model paths
    teacher_model_name: str = "Qwen/Qwen3-32B"
    student_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"

    # Data
    data_path: str = "data/korean_culture_train_200_en.json" # Full English dataset
    train_val_split: float = 0.9

    # Training
    epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1 # User requested 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine" # Changed to cosine

    # SKD Parameters
    max_new_tokens: int = 4096
    gamma: int = 20
    top_k: int = 5
    teacher_temperature: float = 0.7
    teacher_top_p: float = 1.0
    teacher_min_prob: float = 0.05
    student_temperature: float = 0.7
    student_top_p: float = 1.0

    # Resources
    seed: int = 42
    wandb_project: str = "LUPI-SKD-DDP"
    checkpoint_dir: str = "checkpoints/lupi_skd"

    # Resume from checkpoint
    resume_from_checkpoint: str = None  # e.g., "checkpoints/lupi_skd/step_44"
    resume_step: int = 0                # Global step to resume from (set automatically if resume_from_checkpoint is set)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def compute_kl_loss(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
    loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
    return loss_fct(student_log_probs, teacher_log_probs)

def evaluate(model, teacher_model, dataloader, device_s, device_t, rank):
    """Compute Validation Loss (KL)."""
    model.eval()
    total_loss = 0
    steps = 0
    
    if rank == 0: print(f"[{rank}] Starting Evaluation...")
    
    with torch.no_grad():
        for batch in dataloader:
            t_input_ids = batch['teacher_input_ids'].to(device_t)
            s_input_ids = batch['student_input_ids'].to(device_s)
            
            # For validation, we don't do rollout (too slow). 
            # We just check KL on the prompt + ground truth if available?
            # BUT LUPIDataset only gives prompts. We need generated tokens to compute loss.
            # So we MUST do rollout or use a fixed target.
            # Let's do a short rollout for validation (max 1024) to measure alignment.
            
            # Note: Using 'lupi_skd_rollout_adaptive' for val is okay but expensive.
            # Let's limit max_new_tokens for val to 512 to save time.
            
            _, _, gen_tokens, _, _ = lupi_skd_rollout_adaptive(
                teacher_model=teacher_model,
                student_model=model.module if isinstance(model, DDP) else model,
                teacher_input_ids=t_input_ids,
                student_input_ids=s_input_ids,
                max_new_tokens=512, # Shorter for val
                top_k=5, gamma=10, # Conservative settings
                teacher_temperature=0.7, student_temperature=0.7,
                teacher_top_p=1.0, student_top_p=1.0,
                verbose=False
            )
            
            if not gen_tokens: continue
            
            gen_tensor = torch.tensor([gen_tokens], dtype=torch.long).to(device_t)
            full_t_input = torch.cat([t_input_ids, gen_tensor], dim=1)
            gen_tensor_s = gen_tensor.to(device_s)
            full_s_input = torch.cat([s_input_ids, gen_tensor_s], dim=1)
            
            t_out = teacher_model(full_t_input)
            t_logits = t_out.logits[:, t_input_ids.shape[1]-1:-1, :]
            
            s_out = model(full_s_input)
            s_logits = s_out.logits[:, s_input_ids.shape[1]-1:-1, :]
            
            loss = compute_kl_loss(s_logits, t_logits.to(device_s))
            total_loss += loss.item()
            steps += 1
    
    model.train()
    return total_loss / steps if steps > 0 else 0.0

def main():
    local_rank, rank, world_size = setup_ddp()
    
    teacher_device = torch.device(f"cuda:{local_rank * 2}")
    student_device = torch.device(f"cuda:{local_rank * 2 + 1}")
    torch.cuda.set_device(student_device)
    
    config = LUPISKDConfig()

    # Override from environment variables
    if os.environ.get("RESUME_CHECKPOINT"):
        config.resume_from_checkpoint = os.environ["RESUME_CHECKPOINT"]
        if rank == 0:
            print(f"[{rank}] RESUME_CHECKPOINT env var detected: {config.resume_from_checkpoint}")
    
    # Set seeds for reproducibility (Important for Split)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if rank == 0:
        print(f"[{rank}] DDP Init. World Size: {world_size}")
        print(f"[{rank}] Config: {config}")

    # 2. Model Loading
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if rank == 0: print(f"[{rank}] Loading Teacher...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to(teacher_device).eval()
    
    # Load Student (from checkpoint if resuming)
    if config.resume_from_checkpoint:
        if rank == 0: print(f"[{rank}] Resuming Student from checkpoint: {config.resume_from_checkpoint}")
        student_model = AutoModelForCausalLM.from_pretrained(
            config.resume_from_checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="kernels-community/vllm-flash-attn3",
        ).to(student_device).train()
        # Extract step number from checkpoint path if not set
        if config.resume_step == 0:
            ckpt_name = Path(config.resume_from_checkpoint).name
            if ckpt_name.startswith("step_"):
                config.resume_step = int(ckpt_name.split("_")[1])
                if rank == 0: print(f"[{rank}] Auto-detected resume_step: {config.resume_step}")
    else:
        if rank == 0: print(f"[{rank}] Loading Student from scratch: {config.student_model_name}")
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="kernels-community/vllm-flash-attn3",
        ).to(student_device).train()
    
    student_ddp = DDP(student_model, device_ids=[local_rank * 2 + 1], output_device=local_rank * 2 + 1, find_unused_parameters=False)
    
    # 3. Data Splitting
    full_dataset = LUPIDataset(config.data_path, tokenizer)
    total_size = len(full_dataset)
    indices = list(range(total_size))
    # Shuffle indices with fixed seed
    random.shuffle(indices)
    
    split = int(total_size * config.train_val_split)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    if rank == 0:
        print(f"[{rank}] Data Split: Train {len(train_dataset)}, Val {len(val_dataset)}")
    
    collator = LUPICollator(tokenizer)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collator)
    
    # Val dataloader: Only rank 0 needs to evaluate? Or distribute? 
    # Let's distribute val to speed it up, then average metrics.
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, collate_fn=collator)
    
    optimizer = AdamW(student_ddp.parameters(), lr=config.learning_rate)
    
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler_type, 
        optimizer=optimizer, 
        num_warmup_steps=int(config.warmup_ratio * total_steps), 
        num_training_steps=total_steps
    )
    
    if rank == 0:
        # WandB init with resume support
        wandb_resume = "allow" if config.resume_from_checkpoint else None
        run_name = f"full-ddp-lupi-5ep" + (f"-resume-{config.resume_step}" if config.resume_step > 0 else "")
        wandb.init(project=config.wandb_project, name=run_name, resume=wandb_resume)
        print(f"[{rank}] Total Training Steps: {total_steps}")
        if config.resume_from_checkpoint:
            print(f"[{rank}] Resuming from step {config.resume_step}")

    # Initialize global_step (resume if specified)
    global_step = config.resume_step
    save_interval = int(steps_per_epoch * 0.5) # Save every 0.5 epoch

    # Calculate which epoch to start from
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    skip_steps_in_epoch = global_step % steps_per_epoch if steps_per_epoch > 0 else 0
    if rank == 0 and config.resume_from_checkpoint:
        print(f"[{rank}] Starting from epoch {start_epoch}, skipping {skip_steps_in_epoch} steps in first epoch")
    
    for epoch in range(start_epoch, config.epochs):
        train_sampler.set_epoch(epoch)
        student_ddp.train()

        progress_bar = tqdm(train_dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{config.epochs}")

        for step, batch in enumerate(progress_bar):
            # Skip already processed steps when resuming
            if epoch == start_epoch and step < skip_steps_in_epoch * config.gradient_accumulation_steps:
                continue
            t_input_ids = batch['teacher_input_ids'].to(teacher_device)
            s_input_ids = batch['student_input_ids'].to(student_device)
            
            with torch.no_grad():
                _, _, gen_tokens, gen_text, stats = lupi_skd_rollout_adaptive(
                    teacher_model=teacher_model,
                    student_model=student_model, # Use raw model
                    teacher_input_ids=t_input_ids,
                    student_input_ids=s_input_ids,
                    max_new_tokens=config.max_new_tokens,
                    top_k=config.top_k,
                    gamma=config.gamma,
                    teacher_temperature=config.teacher_temperature,
                    student_temperature=config.student_temperature,
                    teacher_top_p=config.teacher_top_p,
                    student_top_p=config.student_top_p,
                    teacher_min_prob=config.teacher_min_prob,
                    tokenizer=tokenizer,
                    verbose=(rank == 0 and step % 10 == 0)
                )
            
            if not gen_tokens: continue

            gen_tensor = torch.tensor([gen_tokens], dtype=torch.long).to(teacher_device)
            full_t_input = torch.cat([t_input_ids, gen_tensor], dim=1)
            gen_tensor_s = gen_tensor.to(student_device)
            full_s_input = torch.cat([s_input_ids, gen_tensor_s], dim=1)
            
            with torch.no_grad():
                t_out = teacher_model(full_t_input)
                t_logits_gen = t_out.logits[:, t_input_ids.shape[1]-1:-1, :]
            
            s_out = student_ddp(full_s_input)
            s_logits_gen = s_out.logits[:, s_input_ids.shape[1]-1:-1, :]
            
            t_logits_gen = t_logits_gen.to(student_device)
            loss = compute_kl_loss(s_logits_gen, t_logits_gen)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_ddp.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if rank == 0:
                    # --- Advanced WandB Logging for Research ---
                    # Extract block stats
                    block_stats = stats.get('block_stats', [])
                    gamma_history = stats.get('gamma_history', [])
                    
                    metrics = {
                        "train/loss": loss.item() * config.gradient_accumulation_steps,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    }
                    
                    if gamma_history:
                        metrics.update({
                            "gamma/mean": np.mean(gamma_history),
                            "gamma/max": np.max(gamma_history),
                            "gamma/min": np.min(gamma_history),
                            "gamma/std": np.std(gamma_history),
                        })
                    
                    if block_stats:
                        # Acceptance Rates
                        acc_rates = [b['acceptance_rate'] for b in block_stats]
                        metrics['acc_rate/mean'] = np.mean(acc_rates)
                        
                        # First Token Acceptance
                        first_rejected_count = sum(1 for b in block_stats if b.get('first_rejected', False))
                        metrics['acc_rate/first_token'] = 1.0 - (first_rejected_count / len(block_stats))
                        
                        # Full Match Rate
                        full_match_count = sum(1 for b in block_stats if b['n_matches'] == b['gamma'])
                        metrics['acc_rate/full_match'] = full_match_count / len(block_stats)
                        
                        # Token Efficiency
                        total_drafted = stats.get('total_drafted', 0)
                        total_accepted = stats.get('total_accepted', 0)
                        total_rejected = stats.get('total_rejected', 0)
                        metrics['tokens/total_drafted'] = total_drafted
                        metrics['tokens/total_accepted'] = total_accepted
                        metrics['tokens/total_rejected'] = total_rejected
                        metrics['tokens/acceptance_ratio_global'] = total_accepted / total_drafted if total_drafted > 0 else 0
                    
                    # [DEBUG LOGGING] Print key metrics to stdout
                    print(f"  [Step {global_step}] Loss: {metrics['train/loss']:.4f} | "
                          f"Gamma(Avg): {metrics.get('gamma/mean', 0):.2f} | "
                          f"Acc(Avg): {metrics.get('acc_rate/mean', 0):.2%} | "
                          f"GenLen: {len(gen_tokens)}")
                    
                    wandb.log(metrics)
                
                # Save & Validate every 0.5 Epoch
                if global_step % save_interval == 0:
                    # Validation
                    val_loss = evaluate(student_ddp, teacher_model, val_dataloader, student_device, teacher_device, rank)
                    
                    # Aggregate Val Loss across ranks
                    val_loss_tensor = torch.tensor(val_loss).to(student_device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss_avg = val_loss_tensor.item() / world_size
                    
                    if rank == 0:
                        print(f"\n[Step {global_step}] Validation Loss: {val_loss_avg:.4f}")
                        wandb.log({"val/loss": val_loss_avg})
                        
                        # Save
                        save_path = os.path.join(config.checkpoint_dir, f"step_{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        print(f"[{rank}] Saving checkpoint to {save_path}...")
                        student_ddp.module.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        print(f"[{rank}] Saved.")
                        
    cleanup_ddp()

if __name__ == "__main__":
    main()