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

import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModel, BitsAndBytesConfig, AutoConfig,AutoImageProcessor, LlavaOnevisionForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, DummyDataset, QwenVLRLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

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
    batch_size: int = 4                                            # Fine-tuning batch size
    max_steps: int = 60000                                        # Max number of fine-tuning steps
    save_steps: int = 3000                                          # Interval for checkpoint saving
    learning_rate: float = 1e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = False                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "microsoft-research"                          # Name of entity to log under

    # fmt: on

    ddp: bool = True

    note: str = "no"

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
        "+1106"
        f"+ia-{cfg.image_aug}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
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

    #print('cfg.vla_path: ', cfg.vla_path)
    #sys.exit()

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    if 'llava-onevision-qwen2-7b-ov-hf' in cfg.vla_path:
        vla = LlavaOnevisionForConditionalGeneration.from_pretrained(
            cfg.vla_path, 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        image_processor_transform = processor.image_processor.preprocess
        height = 384
        width = 384
    elif 'InternVL2_5-8B' in cfg.vla_path:
        vla = AutoModel.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print('wtf 0: ', dir(processor.image_processor))
        sys.exit()
    elif 'Qwen2.5-VL-7B-Instruct' in cfg.vla_path:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print('wtf 1: ', dir(processor.image_processor))
        print('')
        print('wtf 2: ', processor.image_processor.size)
        sys.exit()
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        image_processor_transform = processor.image_processor.apply_transform
        #height = vla.module.config.image_sizes[0]
        #width = vla.module.config.image_sizes[-1]

        height = vla.config.image_sizes[0]
        width = vla.config.image_sizes[-1]

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
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    ##########################
    '''import requests
    from PIL import Image

    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": 'test'},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, vla.dtype)

    for i in inputs:
        print(i, ' : ', inputs[i].shape)
    print('image_sizes: ', inputs['image_sizes'])
    print('image_sizes: ', inputs['image_sizes'].dtype)
    print('actual img size: ', raw_image.size)
    print('prompt: ', prompt)

    output = vla.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))
    sys.exit()'''
    ##########################

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---

    '''batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )'''

    if 'llava-onevision-qwen2-7b-ov-hf' in cfg.vla_path:
        batch_transform = QwenVLRLDSBatchTransform(
            action_tokenizer,
            #processor.tokenizer,
            processor,
            image_transform=image_processor_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        )
    else:
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=image_processor_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
            #chat_template_func=processor.apply_chat_template
        )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        # resize_resolution=(vla.module.config.image_size, vla.module.config.image_size),
        resize_resolution=(height, width), # vla.module.config.image_sizes[0], vla.module.config.image_sizes[-1]
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    '''vla_dataset = TrialDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        # resize_resolution=(vla.module.config.image_size, vla.module.config.image_size),
        resize_resolution=(height, width), # vla.module.config.image_sizes[0], vla.module.config.image_sizes[-1]
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )'''

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
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
            with torch.autocast("cuda", dtype=torch.bfloat16):

                if 'llava-onevision-qwen2-7b-ov-hf' in cfg.vla_path:
                        output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        image_sizes=batch["image_sizes"],
                        labels=batch["labels"],
                    )
                else:
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                loss = output.loss

            # Backward!
            loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
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

            # Push Metrics to W&B (every 10 steps)
            if distributed_state.is_main_process and step_idx % 10 == 0:
                wandb.log(
                    {"train_loss": loss, "action_accuracy": action_accuracy, "l1_loss": action_l1_loss, "epoch": epoch}, step=step_idx
                )

            # Optimizer Step
            if (step_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
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
                    vla.module.save_pretrained(save_dir)

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> TODO (kpertsch, siddk) :: This is inefficient; probably want to do this post-hoc...
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, save_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        merged_vla.save_pretrained(save_dir)

                # Block on Main Process Checkpointing
                dist.barrier()
            if (step_idx+1) % len(dataloader) == 0:
                if cfg.save_by_epoch_and_max_epoch > 0 and epoch > 24 and epoch % 3 == 0:
                    print(f"Saving Model Checkpoint for Step {step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    save_dir = f'{save_dir}/epoch-{epoch}'

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                epoch += 1

                if cfg.save_by_epoch_and_max_epoch > 0 and epoch > cfg.save_by_epoch_and_max_epoch:
                    break
                
            if step_idx >= cfg.max_steps:
                print(f"Stop training at {step_idx}")
                break

if __name__ == "__main__":
    finetune()
