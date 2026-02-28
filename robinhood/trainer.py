"""
robinhood.trainer — Unsloth-based SFT training from robinhood datasets

Takes a dataset produced by the robinhood pipeline (or any JSON in the
platform format) and finetunes a student model using Unsloth + LoRA.

Supports:
    - Any HuggingFace / local base model loadable by Unsloth
    - LoRA with configurable rank, alpha, target modules
    - Response-only training (masks input tokens so only the output is trained)
    - 4-bit quantized training for large models
    - Wandb logging
    - Checkpoint saving at configurable intervals

Usage::

    python -m robinhood.trainer \\
        --base-model unsloth/Qwen3-14B \\
        --dataset ./robinhood_output/dataset/train.json \\
        --output-dir ./trained_model

Or from Python::

    from robinhood.trainer import DistillTrainer, TrainConfig
    trainer = DistillTrainer(TrainConfig(
        base_model="unsloth/Qwen3-14B",
        dataset_path="./robinhood_output/dataset/train.json",
    ))
    trainer.train()
"""

import argparse
import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class LoraConfig:
    """LoRA adapter configuration."""
    rank: int = 16
    alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Full training configuration."""
    # Model
    base_model: str = "unsloth/Qwen3-14B"
    max_seq_length: int = 8000
    load_in_4bit: bool = True

    # Data
    dataset_path: str = "./robinhood_output/dataset/train.json"
    val_dataset_path: Optional[str] = None
    system_prompt_override: Optional[str] = None

    # LoRA
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Training
    num_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.1
    optim: str = "adamw_8bit"
    bf16: bool = True
    fp16: bool = False
    seed: int = 42

    # Saving / logging
    output_dir: str = "./robinhood_trained"
    save_steps: int = 100
    logging_steps: int = 5
    report_to: str = "none"
    run_name: Optional[str] = None

    # Response-only training (mask input tokens)
    train_on_responses_only: bool = True

    # --- Curriculum learning (Light-R1 style) ---
    curriculum_order: bool = False

    # --- REDI contrastive refinement (Stage 2) ---
    dpo_dataset_path: Optional[str] = None
    dpo_beta: float = 0.1
    dpo_epochs: float = 1.0
    dpo_learning_rate: float = 5e-5

    @classmethod
    def from_file(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = json.load(f)
        lora_data = data.pop("lora", {})
        lora = LoraConfig(**{k: v for k, v in lora_data.items() if k in LoraConfig.__dataclass_fields__})
        return cls(
            lora=lora,
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )

    def to_file(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


def _load_robinhood_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a robinhood-format dataset (JSON list of samples)."""
    with open(path) as f:
        data = json.load(f)
    print(f"[TRAINER] Loaded {len(data)} samples from {path}")
    return data


def _format_for_sft(
    task_data: List[Dict[str, Any]],
    tokenizer,
    system_prompt_override: Optional[str] = None,
) -> "Dataset":
    """
    Convert robinhood dataset to Unsloth SFT format.

    Each sample has:
        input: [{role, content}, ...]
        output: {choices: [{message: {content, role}}]}

    We merge input + output into a single message list, apply the chat
    template, and return a HuggingFace Dataset with a "text" column.
    """
    from datasets import Dataset

    formatted = []
    for datum in task_data:
        messages = copy.deepcopy(datum["input"])

        try:
            assistant_msg = copy.deepcopy(datum["output"]["choices"][0]["message"])
            messages.append(assistant_msg)
        except (KeyError, IndexError, TypeError) as e:
            print(f"[TRAINER] Skipping malformed sample: {e}")
            continue

        if system_prompt_override:
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = system_prompt_override

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        formatted.append({"text": text})

    print(f"[TRAINER] Formatted {len(formatted)}/{len(task_data)} samples for SFT")
    return Dataset.from_list(formatted)


def _get_template_prefixes(tokenizer):
    """
    Extract instruction and response prefixes for train_on_responses_only.

    Uses a sentinel-based approach to find where assistant content starts
    in the rendered chat template.
    """
    sys_msg = "Respond to the following query"
    user_msg = "Hello!"
    sentinel = "<|ROBINHOOD_MARKER|>"

    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    # Instruction prefix
    marked = [{"role": "system", "content": f"<|INSTR|>{sys_msg}"}] + msgs[1:]
    rendered_marked = tokenizer.apply_chat_template(
        marked, tokenize=False, add_generation_prompt=False,
    )
    instruction_prefix = rendered_marked.partition("<|INSTR|>")[0]

    # Response prefix
    rendered_base = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    msgs_with_asst = msgs + [{"role": "assistant", "content": sentinel}]
    rendered_asst = tokenizer.apply_chat_template(
        msgs_with_asst, tokenize=False, add_generation_prompt=False,
    )

    if sentinel not in rendered_asst:
        return instruction_prefix, ""

    before_sentinel = rendered_asst.partition(sentinel)[0]
    response_prefix = before_sentinel[len(rendered_base):]

    return instruction_prefix, response_prefix


def _load_dpo_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a DPO-pairs JSONL dataset."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[TRAINER] Loaded {len(pairs)} DPO pairs from {path}")
    return pairs


def _format_for_dpo(
    pairs: List[Dict[str, Any]],
    tokenizer,
) -> "Dataset":
    """
    Convert robinhood DPO pairs to the format expected by trl.DPOTrainer.

    Each pair has: prompt (messages), chosen (str), rejected (str).
    """
    from datasets import Dataset

    formatted = []
    for pair in pairs:
        prompt_msgs = pair.get("prompt", [])
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )

        formatted.append({
            "prompt": prompt_text,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    print(f"[TRAINER] Formatted {len(formatted)} DPO pairs")
    return Dataset.from_list(formatted)


class DistillTrainer:
    """
    Two-stage trainer matching the Chinese distillation methodology:

    **Stage 1 — Curriculum SFT** (Light-R1 style):
        Standard supervised finetuning on verified traces, optionally ordered
        easy-to-hard based on difficulty scores from the verification stage.

    **Stage 2 — REDI Contrastive Refinement** (optional):
        DPO-based refinement using chosen/rejected pairs generated from the
        rejection sampling stage.  The rejected traces that standard
        distillation throws away become valuable training signal.

    Usage::

        trainer = DistillTrainer(TrainConfig(
            base_model="unsloth/Qwen3-14B",
            dataset_path="./robinhood_output/dataset/train.json",
            curriculum_order=True,
            dpo_dataset_path="./robinhood_output/dataset/dpo_pairs.jsonl",
        ))
        trainer.train()
    """

    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()

    def train(self) -> str:
        """
        Run the full training loop (Stage 1 + optional Stage 2).

        Returns:
            Path to the saved adapter directory.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.to_file(os.path.join(cfg.output_dir, "train_config.json"))

        has_dpo = cfg.dpo_dataset_path and os.path.exists(cfg.dpo_dataset_path)
        stages = "2 stages (SFT → DPO)" if has_dpo else "1 stage (SFT)"

        print(f"\n{'='*60}")
        print(f"  ROBINHOOD TRAINER — {stages}")
        print(f"  Base model:  {cfg.base_model}")
        print(f"  Dataset:     {cfg.dataset_path}")
        print(f"  LoRA:        r={cfg.lora.rank}, alpha={cfg.lora.alpha}")
        print(f"  Curriculum:  {'yes (easy→hard)' if cfg.curriculum_order else 'no (shuffled)'}")
        if has_dpo:
            print(f"  DPO pairs:   {cfg.dpo_dataset_path}")
        print(f"  Output:      {cfg.output_dir}")
        print(f"{'='*60}\n")

        adapter_path = self._stage1_curriculum_sft()

        if has_dpo:
            adapter_path = self._stage2_contrastive(adapter_path)

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Adapter: {adapter_path}")
        print(f"{'='*60}\n")

        return adapter_path

    # ------------------------------------------------------------------
    # Stage 1: Curriculum SFT
    # ------------------------------------------------------------------

    def _stage1_curriculum_sft(self) -> str:
        """
        Standard SFT with optional curriculum ordering.

        When ``curriculum_order`` is True, samples are presented easy-to-hard
        (sorted by difficulty metadata from the verification stage), matching
        Light-R1's progressive difficulty scheduling.
        """
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig

        cfg = self.config

        print("[STAGE 1] Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.load_in_4bit,
        )

        print(f"[STAGE 1] Applying LoRA (r={cfg.lora.rank}, alpha={cfg.lora.alpha})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.rank,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
            use_rslora=False,
            loftq_config=None,
        )

        print("[STAGE 1] Loading dataset...")
        train_data = _load_robinhood_dataset(cfg.dataset_path)

        if cfg.curriculum_order:
            train_data.sort(
                key=lambda s: s.get("metadata", {}).get("difficulty", 0.5)
            )
            difficulties = [s.get("metadata", {}).get("difficulty", 0.5) for s in train_data]
            if difficulties:
                print(
                    f"[STAGE 1] Curriculum ordering: difficulty range "
                    f"{min(difficulties):.2f} → {max(difficulties):.2f}"
                )

        train_dataset = _format_for_sft(
            train_data, tokenizer, cfg.system_prompt_override,
        )

        val_dataset = None
        if cfg.val_dataset_path and os.path.exists(cfg.val_dataset_path):
            val_data = _load_robinhood_dataset(cfg.val_dataset_path)
            val_dataset = _format_for_sft(
                val_data, tokenizer, cfg.system_prompt_override,
            )

        sft_output = os.path.join(cfg.output_dir, "stage1_sft")
        sft_args = SFTConfig(
            output_dir=sft_output,
            num_train_epochs=cfg.num_epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            optim=cfg.optim,
            bf16=cfg.bf16,
            fp16=cfg.fp16,
            seed=cfg.seed,
            save_steps=cfg.save_steps,
            logging_steps=cfg.logging_steps,
            report_to=cfg.report_to,
            run_name=f"{cfg.run_name or 'robinhood'}_sft",
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_args,
        )

        if cfg.train_on_responses_only:
            try:
                from unsloth.chat_templates import train_on_responses_only as apply_response_mask
                instr_prefix, resp_prefix = _get_template_prefixes(tokenizer)
                if resp_prefix:
                    trainer = apply_response_mask(
                        trainer,
                        instruction_part=instr_prefix,
                        response_part=resp_prefix,
                    )
                    print(f"[STAGE 1] Response-only masking enabled")
                else:
                    print("[STAGE 1] Could not detect response prefix, training on full sequence")
            except ImportError:
                print("[STAGE 1] train_on_responses_only not available, training on full sequence")

        print(f"\n[STAGE 1] Starting curriculum SFT...")
        print(f"  Samples:          {len(train_dataset)}")
        print(f"  Epochs:           {cfg.num_epochs}")
        print(f"  Effective batch:  {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
        print(f"  Learning rate:    {cfg.learning_rate}")
        print(f"  Curriculum order: {cfg.curriculum_order}")
        print()

        trainer.train()

        adapter_dir = os.path.join(sft_output, "final_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[STAGE 1] SFT adapter saved to {adapter_dir}")

        return adapter_dir

    # ------------------------------------------------------------------
    # Stage 2: REDI Contrastive Refinement
    # ------------------------------------------------------------------

    def _stage2_contrastive(self, sft_adapter_path: str) -> str:
        """
        REDI-style contrastive refinement using DPO on chosen/rejected pairs.

        Loads the SFT adapter from Stage 1, then applies DPO training using
        pairs where the *chosen* response is the verified-correct trace and
        the *rejected* response is a trace that failed verification for the
        same prompt.  This teaches the model to prefer correct reasoning
        over plausible-but-wrong reasoning.
        """
        from unsloth import FastLanguageModel
        from trl import DPOTrainer, DPOConfig

        cfg = self.config

        print(f"\n[STAGE 2] Loading SFT adapter from {sft_adapter_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_adapter_path,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.load_in_4bit,
        )

        print("[STAGE 2] Loading DPO pairs...")
        dpo_pairs = _load_dpo_dataset(cfg.dpo_dataset_path)
        dpo_dataset = _format_for_dpo(dpo_pairs, tokenizer)

        dpo_output = os.path.join(cfg.output_dir, "stage2_dpo")
        dpo_args = DPOConfig(
            output_dir=dpo_output,
            num_train_epochs=cfg.dpo_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.dpo_learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            beta=cfg.dpo_beta,
            optim=cfg.optim,
            bf16=cfg.bf16,
            fp16=cfg.fp16,
            seed=cfg.seed,
            save_steps=cfg.save_steps,
            logging_steps=cfg.logging_steps,
            report_to=cfg.report_to,
            run_name=f"{cfg.run_name or 'robinhood'}_dpo",
            max_length=cfg.max_seq_length,
            gradient_checkpointing=True,
        )

        dpo_trainer = DPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dpo_dataset,
            args=dpo_args,
        )

        print(f"\n[STAGE 2] Starting REDI contrastive refinement...")
        print(f"  DPO pairs:       {len(dpo_dataset)}")
        print(f"  Epochs:          {cfg.dpo_epochs}")
        print(f"  Beta:            {cfg.dpo_beta}")
        print(f"  Learning rate:   {cfg.dpo_learning_rate}")
        print()

        dpo_trainer.train()

        adapter_dir = os.path.join(dpo_output, "final_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[STAGE 2] DPO-refined adapter saved to {adapter_dir}")

        return adapter_dir


def main():
    parser = argparse.ArgumentParser(
        description="robinhood trainer — Finetune a model on a robinhood distillation dataset"
    )
    parser.add_argument(
        "--config", type=str, help="Path to training config JSON file"
    )
    parser.add_argument(
        "--base-model", type=str, default="unsloth/Qwen3-14B",
        help="Base model to finetune (default: unsloth/Qwen3-14B)"
    )
    parser.add_argument(
        "--dataset", type=str, default="./robinhood_output/dataset/train.json",
        help="Path to training dataset JSON"
    )
    parser.add_argument(
        "--val-dataset", type=str, default=None,
        help="Path to validation dataset JSON"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./robinhood_trained",
        help="Output directory for adapter and checkpoints"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=float, default=3.0,
        help="Number of training epochs (default: 3.0)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device train batch size (default: 2)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=8000,
        help="Maximum sequence length (default: 8000)"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--no-response-only", action="store_true",
        help="Train on full sequence instead of response-only"
    )
    parser.add_argument(
        "--report-to", type=str, default="none",
        choices=["none", "wandb"],
        help="Logging backend (default: none)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for the training run"
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Sort training data easy-to-hard (Light-R1 curriculum learning)"
    )
    parser.add_argument(
        "--dpo-dataset", type=str, default=None,
        help="Path to DPO pairs JSONL for REDI contrastive Stage 2"
    )
    parser.add_argument(
        "--dpo-beta", type=float, default=0.1,
        help="DPO beta parameter (default: 0.1)"
    )
    parser.add_argument(
        "--dpo-epochs", type=float, default=1.0,
        help="DPO training epochs (default: 1.0)"
    )
    parser.add_argument(
        "--dpo-lr", type=float, default=5e-5,
        help="DPO learning rate (default: 5e-5)"
    )

    args = parser.parse_args()

    if args.config:
        config = TrainConfig.from_file(args.config)
    else:
        config = TrainConfig(
            base_model=args.base_model,
            dataset_path=args.dataset,
            val_dataset_path=args.val_dataset,
            output_dir=args.output_dir,
            lora=LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha),
            num_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
            train_on_responses_only=not args.no_response_only,
            report_to=args.report_to,
            run_name=args.run_name,
            curriculum_order=args.curriculum,
            dpo_dataset_path=args.dpo_dataset,
            dpo_beta=args.dpo_beta,
            dpo_epochs=args.dpo_epochs,
            dpo_learning_rate=args.dpo_lr,
        )

    trainer = DistillTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
