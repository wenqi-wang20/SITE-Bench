"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import accelerate
import torchvision
import random

#from magma.image_processing_magma import MagmaImageProcessor
#from magma.processing_magma import MagmaProcessor
#from magma.modeling_magma import MagmaForConditionalGeneration

import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenVLPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, QwenVLPaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer, TiktokenActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, Llama3RLDSBatchTransform, QwenVLRLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                            # Fine-tuning batch size
    max_steps: int = 60000                                        # Max number of fine-tuning steps
    save_steps: int = 3000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 4                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = False                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "fewshot-libero"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "microsoft-research"                          # Name of entity to log under

    # fmt: on

    vision_tower_trainable: bool = True
    ddp: bool = True

    image_resolution: int = 224

    # lr
    constant_lr: bool = False

    note: str = "no"

    max_grad_norm: float = -1.0

    # image encoder
    base_img_size: int = 224

    save_by_epoch_and_max_epoch: int = -1

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        "+1111"
        f"+ia-{cfg.image_aug}"
        f"+ir-{cfg.image_resolution}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}-cst-{cfg.constant_lr}"
        f"+base_im_-{cfg.base_img_size}"
        f"+vision-fronzen-{not cfg.vision_tower_trainable}"
        f"+n-{cfg.note}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    # processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # vla = AutoModelForVision2Seq.from_pretrained(
    #     cfg.vla_path,
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=quantization_config,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )
    # model_id = "/mnt/code/cache/vla-models/magma-llama-3-8b-instruct-hf" 
    #processor = MagmaProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True) 
    #image_processor = MagmaImageProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True, base_img_size=cfg.base_img_size) 
    #vla = MagmaForConditionalGeneration.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16,trust_remote_code=True) 

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    image_processor = processor.image_processor
    vla = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.vla_path, 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ) 

    # for stage in vla.vision_tower.clip_vision_model.trunk.stages:
    #     for block in stage.blocks:
    #         # block.gamma = block.weight.clone().contiguous()
    #         block.gamma.data.copy_(block.weight.data)


    if not cfg.vision_tower_trainable:
        for param in vla.vision_tower.parameters():
            param.requires_grad = False

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla.language_model = get_peft_model(vla.language_model, lora_config)
        # vla.print_trainable_parameters()

    if cfg.ddp:
        # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
        # Create Optimizer =>> note that we default to a simple constant learning rate!
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
        if not cfg.constant_lr:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps)


    # Create Action Tokenizer
    action_tokenizer = TiktokenActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    from prismatic.vla.datasets import DummyDataset
    
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=image_processor.preprocess,
    #     prompt_builder_fn=MegamaPromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---

    height, width = 224, 224

    batch_transform = QwenVLRLDSBatchTransform(
        action_tokenizer,
        processor,
        image_transform=image_processor.preprocess,
        prompt_builder_fn=QwenVLPromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        # resize_resolution=(vla.module.config.image_size, vla.module.config.image_size),
        resize_resolution=(cfg.image_resolution, cfg.image_resolution),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = QwenVLPaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    if not cfg.ddp:
        # Create Optimizer =>> note that we default to a simple constant learning rate!
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

        from accelerate import Accelerator
        accelerator = Accelerator(dispatch_batches=False)
        if not cfg.constant_lr:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps * 9)
            vla, optimizer, dataloader, lr_scheduler  = accelerator.prepare(
                vla, optimizer, dataloader, lr_scheduler
            )
        else:
            vla, optimizer, dataloader = accelerator.prepare(
                vla, optimizer, dataloader
            )

        print(len(dataloader))

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Train!
    epoch = 0
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for step_idx, batch in enumerate(dataloader):
            # with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    image_grid_thw=batch["image_grid_thw"].to(device_id),
                    labels=batch["labels"],
                    #image_sizes=batch["image_sizes"],
                )
                loss = output.loss                    

            # Backward!
            if cfg.ddp:
                loss.backward()
            else:
                accelerator.backward(loss)

            print(output.logits.shape)
            print(batch["labels"].shape)
            # Compute Accuracy and L1 Loss for Logging
            action_len = batch["labels"].size(1)
            action_logits = output.logits[:, -action_len : -1]
            # action_logits = output.logits[:, 599 : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)


            # Compute Gradient Norm
            if cfg.max_grad_norm > 0.0:
                if cfg.ddp:
                    grad_norm = torch.nn.utils.clip_grad_norm_(vla.parameters(), cfg.max_grad_norm)
                else:
                    grad_norm = accelerator.clip_grad_norm_(vla.parameters(), cfg.max_grad_norm)
                # grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            else:
                grad_norm = 0.0

            # Get Learning Rate
            lr = optimizer.param_groups[0]['lr']

            # Push Metrics to W&B (every 10 steps)
            if distributed_state.is_main_process and step_idx % 10 == 0:
                wandb.log(
                    {"train_loss": loss, "action_accuracy": action_accuracy, "l1_loss": action_l1_loss, "epoch": epoch, "grad_norm": grad_norm, 'lr': lr}, step=step_idx
                )
                print(grad_norm)

            # Optimizer Step
            if (step_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                if not cfg.constant_lr:
                    lr_scheduler.step()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if step_idx > 0 and step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    save_dir = f'{save_dir}/step-{step_idx}-epoch-{epoch}'

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)

                    if cfg.ddp:
                        if cfg.use_lora:
                            vla.module.language_model.save_pretrained(save_dir)
                        else:
                            vla.module.save_pretrained(save_dir)
                    else:
                        # accelerator.save_state(output_dir=save_dir)
                        unwrapped_model = accelerator.unwrap_model(vla)
                        # unwrapped_model.to('cpu')
                        # torch.cuda.empty_cache()
                        state_dict = accelerator.get_state_dict(vla)

                        unwrapped_model.save_pretrained(
                        save_dir, 
                        is_main_process=accelerator.is_main_process, 
                        save_function=accelerator.save, 
                        state_dict=state_dict
                        )
                    
                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> TODO (kpertsch, siddk) :: This is inefficient; probably want to do this post-hoc...
                    if cfg.use_lora:
                        base_vla = MagmaForConditionalGeneration.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16,trust_remote_code=True, low_cpu_mem_usage=True) 
                        merged_vla = PeftModel.from_pretrained(base_vla, save_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        merged_vla.save_pretrained(f"{run_dir}/step-{step_idx}-epoch-{epoch}", safe_serialization=False)

                # Block on Main Process Checkpointing
                dist.barrier()
            if step_idx % len(dataloader) == 0:
                if cfg.save_by_epoch_and_max_epoch > 0 and epoch >= 15 and epoch % 3 == 0:
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    save_dir = f'{save_dir}/epoch-{epoch}'

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)

                    if cfg.ddp:
                        if cfg.use_lora:
                            vla.module.language_model.save_pretrained(save_dir)
                        else:
                            vla.module.save_pretrained(save_dir)
                    else:
                        # accelerator.save_state(output_dir=save_dir)
                        unwrapped_model = accelerator.unwrap_model(vla)
                        # unwrapped_model.to('cpu')
                        # torch.cuda.empty_cache()
                        state_dict = accelerator.get_state_dict(vla)

                        unwrapped_model.save_pretrained(
                        save_dir, 
                        is_main_process=accelerator.is_main_process, 
                        save_function=accelerator.save, 
                        state_dict=state_dict
                        )
                epoch += 1
                if cfg.save_by_epoch_and_max_epoch > 0 and epoch > cfg.save_by_epoch_and_max_epoch:
                    break
            
            if step_idx >= cfg.max_steps:
                print(f"Stop training at {step_idx}")
                break
if __name__ == "__main__":
    finetune()
