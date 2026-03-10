"""
robinhood.cli — Unified command-line interface

Provides the ``robinhood`` command with three subcommands::

    robinhood collect   — collect reasoning traces and build a dataset
    robinhood train     — finetune a student model on an existing dataset
    robinhood reformat  — reformat previously collected traces

Install and run::

    pip install -e .
    robinhood collect --model claude-sonnet-4-20250514 --samples-per-category 50
    robinhood train   --base-model unsloth/Qwen3-14B --dataset ./train.json
"""

import argparse
import sys
import time

from robinhood import __version__


# ------------------------------------------------------------------
# Consent
# ------------------------------------------------------------------

_CONSENT_TEXT = """\
╔══════════════════════════════════════════════════════════════════════╗
║                         robinhood — consent                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Before collecting reasoning traces, please confirm:                 ║
║                                                                      ║
║  1. I have reviewed my provider's terms of service and my intended   ║
║     use of model outputs for training is permitted.                  ║
║                                                                      ║
║  2. I will respect the provider's published rate limits and will     ║
║     not use multiple accounts, region bypass, or any other means     ║
║     to circumvent access restrictions.                               ║
║                                                                      ║
║  3. I understand that student models trained via distillation may    ║
║     not inherit the teacher's safety training, and I will evaluate   ║
║     safety independently before deployment.                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _require_consent(skip: bool) -> None:
    if skip:
        return
    print(_CONSENT_TEXT)
    try:
        answer = input("Do you confirm? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in ("y", "yes"):
        print("\nConsent not given. Exiting.")
        sys.exit(0)


# ------------------------------------------------------------------
# Shared CLI helpers
# ------------------------------------------------------------------

def _add_compliance_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("compliance")
    group.add_argument(
        "--compliance", action="store_true",
        help="Conservative concurrency (5) and rate limits (30 RPM)",
    )
    group.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive consent prompt",
    )


# ------------------------------------------------------------------
# robinhood collect
# ------------------------------------------------------------------

def _build_collect_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "collect",
        help="Collect reasoning traces from a teacher model and build a dataset",
        description="Run the full pipeline: prompt generation → trace collection "
                    "→ verification → formatting → export (→ optional training).",
    )

    g = p.add_argument_group("provider")
    g.add_argument("--provider", choices=["anthropic", "openai", "openrouter"],
                   help="LLM provider (default: auto-detected)")
    g.add_argument("--api-key", help="API key (default: from env var)")
    g.add_argument("--model", default="claude-sonnet-4-20250514",
                   help="Teacher model (default: claude-sonnet-4-20250514)")

    g = p.add_argument_group("prompts")
    g.add_argument("--samples-per-category", type=int, default=10,
                   help="Prompts per category (default: 10)")
    g.add_argument("--categories", nargs="*",
                   help="Prompt categories to include (default: all)")
    g.add_argument("--prompt-file", help="Load prompts from file")
    g.add_argument("--skill-file", help="Skill definition file (JSON/YAML)")
    g.add_argument("--prompts-per-skill", type=int, default=20,
                   help="Prompts to synthesize per skill (default: 20)")
    g.add_argument("--synthesis-model", help="Model for prompt synthesis")

    g = p.add_argument_group("collection")
    g.add_argument("--thinking-budget", type=int, default=10000,
                   help="Thinking token budget (default: 10000)")
    g.add_argument("--rate-limit", type=int, default=50,
                   help="Requests per minute (default: 50)")
    g.add_argument("--max-concurrent", type=int, default=10,
                   help="Max concurrent requests (default: 10)")

    g = p.add_argument_group("rejection sampling & verification")
    g.add_argument("--samples-per-prompt", type=int, default=1,
                   help="Traces per prompt for rejection sampling (default: 1)")
    g.add_argument("--judge-model", help="Model for LLM-as-judge")
    g.add_argument("--min-judge-score", type=float, default=6.0,
                   help="Min judge score (default: 6.0)")
    g.add_argument("--difficulty-min", type=float, default=0.05,
                   help="Min difficulty to keep (default: 0.05)")
    g.add_argument("--difficulty-max", type=float, default=0.95,
                   help="Max difficulty to keep (default: 0.95)")

    g = p.add_argument_group("formatting & export")
    g.add_argument("--format", default="thinking_and_output",
                   choices=["thinking_and_output", "output_only",
                            "reasoning_augmented", "multi_turn"],
                   help="Dataset format (default: thinking_and_output)")
    g.add_argument("--export", default="platform",
                   choices=["platform", "huggingface"],
                   help="Export format (default: platform)")
    g.add_argument("--curriculum", action="store_true",
                   help="Sort training data easy-to-hard")
    g.add_argument("--output-dir", default="./robinhood_output",
                   help="Output directory (default: ./robinhood_output)")
    g.add_argument("--run-name", help="Run name (default: auto)")
    g.add_argument("--config", help="Pipeline config JSON file")

    g = p.add_argument_group("training (optional, pass --train to enable)")
    g.add_argument("--train", action="store_true",
                   help="Train a student model after export")
    g.add_argument("--base-model", default="unsloth/Qwen3-14B",
                   help="Student model (default: unsloth/Qwen3-14B)")
    g.add_argument("--lora-rank", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=16)
    g.add_argument("--train-epochs", type=float, default=3.0)
    g.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization")
    g.add_argument("--dpo-beta", type=float, default=0.1)
    g.add_argument("--dpo-epochs", type=float, default=1.0)
    g.add_argument("--dpo-lr", type=float, default=5e-5)

    _add_compliance_flags(p)
    return p


def _run_collect(args) -> None:
    _require_consent(args.yes)

    from robinhood.pipeline import PipelineConfig, RobinhoodPipeline

    if args.config:
        config = PipelineConfig.from_file(args.config)
    else:
        if args.skill_file:
            prompt_source = "skills"
        elif args.prompt_file:
            prompt_source = args.prompt_file
        else:
            prompt_source = "templates"

        model = args.model
        model_short = (model.split("/")[-1].split("-")[0] if "/" in model
                       else model.split("-")[1] if "-" in model else model)
        run_name = args.run_name or f"robinhood_{model_short}_{int(time.time())}"

        config = PipelineConfig(
            output_dir=args.output_dir,
            run_name=run_name,
            prompt_source=prompt_source,
            skill_file=args.skill_file,
            prompts_per_skill=args.prompts_per_skill,
            synthesis_model=args.synthesis_model,
            categories=args.categories,
            samples_per_category=args.samples_per_category,
            provider=args.provider,
            api_key=args.api_key,
            model=model,
            thinking_budget=args.thinking_budget,
            format_mode=args.format,
            export_format=args.export,
            rate_limit_rpm=args.rate_limit,
            max_concurrent=args.max_concurrent,
            samples_per_prompt=args.samples_per_prompt,
            judge_model=args.judge_model,
            min_judge_score=args.min_judge_score,
            difficulty_min=args.difficulty_min,
            difficulty_max=args.difficulty_max,
            curriculum_order=args.curriculum,
            compliance_mode=args.compliance,
            train_after_export=args.train,
            base_model=args.base_model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            train_epochs=args.train_epochs,
            load_in_4bit=not args.no_4bit,
            dpo_beta=args.dpo_beta,
            dpo_epochs=args.dpo_epochs,
            dpo_lr=args.dpo_lr,
        )

    pipeline = RobinhoodPipeline(config=config)
    pipeline.run()


# ------------------------------------------------------------------
# robinhood train
# ------------------------------------------------------------------

def _build_train_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "train",
        help="Finetune a student model on an existing dataset",
        description="Two-stage training: curriculum SFT (Stage 1) then optional "
                    "REDI contrastive DPO (Stage 2) using rejected traces.",
    )

    g = p.add_argument_group("model")
    g.add_argument("--base-model", default="unsloth/Qwen3-14B",
                   help="Base model to finetune (default: unsloth/Qwen3-14B)")
    g.add_argument("--max-seq-length", type=int, default=8000)
    g.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="./robinhood_output/dataset/train.json",
                   help="Training dataset JSON")
    g.add_argument("--val-dataset", help="Validation dataset JSON")

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-rank", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=16)

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=float, default=3.0)
    g.add_argument("--batch-size", type=int, default=2)
    g.add_argument("--grad-accum", type=int, default=4)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--curriculum", action="store_true",
                   help="Sort data easy-to-hard")
    g.add_argument("--no-response-only", action="store_true",
                   help="Train on full sequence, not response-only")

    g = p.add_argument_group("REDI contrastive (Stage 2)")
    g.add_argument("--dpo-dataset",
                   help="DPO pairs JSONL for contrastive refinement")
    g.add_argument("--dpo-beta", type=float, default=0.1)
    g.add_argument("--dpo-epochs", type=float, default=1.0)
    g.add_argument("--dpo-lr", type=float, default=5e-5)

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="./robinhood_trained")
    g.add_argument("--report-to", default="none", choices=["none", "wandb"])
    g.add_argument("--run-name")
    g.add_argument("--config", help="Training config JSON file")

    return p


def _run_train(args) -> None:
    from robinhood.trainer import DistillTrainer, TrainConfig, LoraConfig

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


# ------------------------------------------------------------------
# robinhood reformat
# ------------------------------------------------------------------

def _build_reformat_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "reformat",
        help="Reformat previously collected traces (no API calls)",
        description="Convert existing traces into a different format mode "
                    "without re-collecting from the provider.",
    )
    p.add_argument("traces", help="Path to traces JSON file")
    p.add_argument("--format", default="thinking_and_output",
                   choices=["thinking_and_output", "output_only",
                            "reasoning_augmented", "multi_turn"],
                   help="Target format (default: thinking_and_output)")
    p.add_argument("--export", default="platform",
                   choices=["platform", "huggingface"])
    p.add_argument("--output-dir", default="./robinhood_output")
    return p


def _run_reformat(args) -> None:
    from robinhood.pipeline import run_from_existing_traces
    run_from_existing_traces(
        traces_path=args.traces,
        output_dir=args.output_dir,
        format_mode=args.format,
        export_format=args.export,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="robinhood",
        description=(
            "Reasoning trace collection & dataset formatting. "
            "Collect traces from teacher models, build verified datasets, "
            "train student models."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"robinhood {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    _build_collect_parser(sub)
    _build_train_parser(sub)
    _build_reformat_parser(sub)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "collect": _run_collect,
        "train": _run_train,
        "reformat": _run_reformat,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
