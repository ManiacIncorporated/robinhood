"""
robinhood.pipeline — End-to-end distillation pipeline

Orchestrates the full workflow for creating finetuning datasets from
teacher-model reasoning traces.

Pipeline stages (full closed-loop mode):
    1. Prompt Generation — templates or skill files
    2. Multi-Answer Trace Collection — N completions per prompt
    3. Dual Verification & Rejection Sampling — deterministic + LLM-as-judge
    4. Difficulty Scoring & Filtering — pass-rate based
    5. Dataset Formatting — 4 SFT format modes
    6. DPO Pair Export — chosen/rejected for contrastive stage
    7. Curriculum-Ordered Export — easy-to-hard
    8. Two-Stage Training — curriculum SFT → DPO refinement

Legacy mode (samples_per_prompt=1, no verification) works exactly as before.

Users are responsible for ensuring their use of model outputs complies
with the relevant provider's terms of service.

Usage:
    python -m robinhood --config config.json
    python -m robinhood --model claude-sonnet-4-20250514 --samples 1000
    python -m robinhood --model claude-sonnet-4-20250514 --samples-per-prompt 8 --curriculum --train
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from robinhood.trace_collector import (
    TraceCollector,
    ClaudeTraceCollector,
    CollectionConfig,
    ThinkingConfig,
    CollectedTrace,
)
from robinhood.dataset_formatter import (
    DatasetFormatter,
    FormatterConfig,
)
from robinhood.prompt_sources import (
    generate_prompts,
    generate_prompts_from_file,
)
from robinhood.skills import (
    SkillSet,
    SkillPromptSynthesizer,
    generate_prompts_from_skills,
)
from robinhood.verification import (
    TraceVerifier,
    VerificationConfig,
)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    # Output
    output_dir: str = "./robinhood_output"
    run_name: str = "robinhood_run"

    # Prompt generation
    prompt_source: str = "templates"  # "templates", "skills", or path to file
    skill_file: Optional[str] = None
    prompts_per_skill: int = 20
    synthesis_model: Optional[str] = None  # defaults to same as collection model
    categories: Optional[List[str]] = None
    samples_per_category: int = 10
    prompt_seed: int = 42

    # Provider / auth — works with anthropic, openai, or openrouter
    provider: Optional[str] = None   # auto-detected from model if None
    api_key: Optional[str] = None    # falls back to env vars per provider

    # Model & collection
    model: str = "claude-sonnet-4-20250514"
    thinking_budget: int = 10000
    max_output_tokens: int = 4096
    batch_size: int = 5
    max_concurrent: int = 10
    rate_limit_rpm: int = 50

    # Formatting
    format_mode: str = "thinking_and_output"
    max_thinking_tokens: Optional[int] = None
    include_system_prompt: bool = True
    train_split_ratio: float = 0.9

    # Quality filtering
    min_thinking_chars: int = 50
    min_output_chars: int = 10
    max_output_chars: int = 50000

    # --- Rejection sampling & verification ---
    samples_per_prompt: int = 1
    judge_model: Optional[str] = None
    min_judge_score: float = 6.0
    difficulty_min: float = 0.05
    difficulty_max: float = 0.95

    # --- Curriculum ordering ---
    curriculum_order: bool = False

    # --- Compliance mode ---
    # When True, enforces conservative defaults: low concurrency (5),
    # strict rate-limit (30 RPM), and skips any retry on 429/rate-limit errors.
    compliance_mode: bool = False

    # Export
    export_format: str = "platform"  # "platform" or "huggingface"

    # Training (when --train is used)
    train_after_export: bool = False
    base_model: str = "unsloth/Qwen3-14B"
    lora_rank: int = 16
    lora_alpha: int = 16
    train_epochs: float = 3.0
    train_batch_size: int = 2
    train_grad_accum: int = 4
    train_lr: float = 3e-4
    load_in_4bit: bool = True

    # --- REDI contrastive Stage 2 ---
    dpo_beta: float = 0.1
    dpo_epochs: float = 1.0
    dpo_lr: float = 5e-5

    @classmethod
    def from_file(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_file(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


_COMPLIANCE_DEFAULTS = {
    "rate_limit_rpm": 30,
    "max_concurrent": 5,
    "batch_size": 3,
}


class RobinhoodPipeline:
    """
    End-to-end pipeline for creating reasoning trace distillation datasets.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        if self.config.compliance_mode:
            self._apply_compliance_defaults()
        self._setup_output_dir()

    def _apply_compliance_defaults(self):
        """
        Override concurrency/rate-limit settings with conservative values.

        Compliance mode ensures the pipeline stays well within published
        provider rate limits and uses minimal concurrency, even if the
        user-supplied config specifies higher values.
        """
        c = self.config
        for key, safe_val in _COMPLIANCE_DEFAULTS.items():
            if getattr(c, key) > safe_val:
                setattr(c, key, safe_val)
        print(
            f"[COMPLIANCE] Compliance mode active — "
            f"rate_limit={c.rate_limit_rpm} RPM, "
            f"max_concurrent={c.max_concurrent}, "
            f"batch_size={c.batch_size}"
        )

    def _setup_output_dir(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.traces_dir = os.path.join(self.config.output_dir, "traces")
        self.dataset_dir = os.path.join(self.config.output_dir, "dataset")
        os.makedirs(self.traces_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

    def _is_full_pipeline(self) -> bool:
        """True when rejection sampling + verification is enabled."""
        return self.config.samples_per_prompt > 1

    def run(self) -> Dict[str, Any]:
        """
        Execute the full pipeline and return summary statistics.

        When ``samples_per_prompt > 1`` the full closed-loop pipeline runs:
            Prompts → Multi-sample collection → Verification & rejection
            sampling → Difficulty scoring → Formatting → DPO pair export
            → Curriculum-ordered export → Two-stage training

        Otherwise the legacy (single-sample) pipeline runs.
        """
        stats = {}
        pipeline_start = time.time()

        config_path = os.path.join(self.config.output_dir, "pipeline_config.json")
        self.config.to_file(config_path)
        from robinhood.providers import detect_provider
        provider = detect_provider(self.config.model, self.config.provider)

        full = self._is_full_pipeline()
        mode = "FULL CLOSED-LOOP" if full else "STANDARD"
        n_stages = self._count_stages()

        print(f"\n{'='*70}")
        print(f"  ROBINHOOD PIPELINE ({mode}): {self.config.run_name}")
        print(f"  Provider:           {provider}")
        print(f"  Model:              {self.config.model}")
        print(f"  Format:             {self.config.format_mode}")
        if full:
            print(f"  Samples/prompt:     {self.config.samples_per_prompt}")
            print(f"  Difficulty range:   [{self.config.difficulty_min:.2f}, {self.config.difficulty_max:.2f}]")
            print(f"  Curriculum order:   {self.config.curriculum_order}")
        if self.config.skill_file:
            print(f"  Skill file:         {self.config.skill_file}")
        print(f"  Output:             {self.config.output_dir}")
        print(f"{'='*70}\n")

        stage = 0

        # --- Stage: Prompt generation ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Generating prompts...")
        stage_start = time.time()
        prompts = self._generate_prompts()
        stats["prompts_generated"] = len(prompts)
        stats["prompt_time"] = time.time() - stage_start
        print(f"  -> {len(prompts)} prompts generated ({stats['prompt_time']:.1f}s)\n")

        prompts_path = os.path.join(self.config.output_dir, "prompts.json")
        with open(prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)

        if full:
            return self._run_full_pipeline(prompts, stats, stage, n_stages, pipeline_start)
        else:
            return self._run_legacy_pipeline(prompts, stats, stage, n_stages, pipeline_start)

    def _count_stages(self) -> int:
        full = self._is_full_pipeline()
        n = 1  # prompt generation
        n += 1  # collection
        if full:
            n += 1  # verification
        else:
            n += 1  # basic quality filter
        n += 1  # format
        n += 1  # export
        if self.config.train_after_export:
            n += 1  # train
        return n

    # ------------------------------------------------------------------
    # Full closed-loop pipeline (samples_per_prompt > 1)
    # ------------------------------------------------------------------

    def _run_full_pipeline(
        self,
        prompts: List[Dict[str, Any]],
        stats: Dict[str, Any],
        stage: int,
        n_stages: int,
        pipeline_start: float,
    ) -> Dict[str, Any]:

        # --- Stage: Multi-answer trace collection ---
        stage += 1
        n = self.config.samples_per_prompt
        print(
            f"[PIPELINE] Stage {stage}/{n_stages}: "
            f"Collecting {n}x traces per prompt ({len(prompts) * n} total)..."
        )
        stage_start = time.time()
        traces_by_prompt = self._collect_multi_traces(prompts)
        total_traces = sum(len(v) for v in traces_by_prompt.values())
        stats["traces_collected"] = total_traces
        stats["prompts_with_traces"] = len(traces_by_prompt)
        stats["collection_time"] = time.time() - stage_start
        print(
            f"  -> {total_traces} traces across {len(traces_by_prompt)} prompts "
            f"({stats['collection_time']:.1f}s)\n"
        )

        if not traces_by_prompt:
            print("[PIPELINE] No traces collected. Exiting.")
            return stats

        # --- Stage: Verification + rejection sampling + difficulty scoring ---
        stage += 1
        print(
            f"[PIPELINE] Stage {stage}/{n_stages}: "
            f"Verification, rejection sampling & difficulty scoring..."
        )
        stage_start = time.time()

        traces_by_prompt_dicts = {
            pid: [t.to_dict() if hasattr(t, "to_dict") else t for t in tlist]
            for pid, tlist in traces_by_prompt.items()
        }

        selected, rejected, difficulty_scores = self._verify_and_select(traces_by_prompt_dicts)
        stats["selected_traces"] = len(selected)
        stats["rejected_traces"] = len(rejected)
        stats["verification_time"] = time.time() - stage_start

        if difficulty_scores:
            diffs = list(difficulty_scores.values())
            stats["difficulty_mean"] = sum(diffs) / len(diffs)
            stats["difficulty_min"] = min(diffs)
            stats["difficulty_max"] = max(diffs)

        print(
            f"  -> {len(selected)} selected, {len(rejected)} rejected "
            f"({stats['verification_time']:.1f}s)\n"
        )

        if not selected:
            print("[PIPELINE] No traces passed verification. Exiting.")
            return stats

        # Save rejected traces for inspection / DPO
        rejected_path = os.path.join(self.traces_dir, "rejected_traces.json")
        with open(rejected_path, "w") as f:
            json.dump(rejected, f, indent=2)

        # --- Stage: Format dataset ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Formatting dataset...")
        stage_start = time.time()
        formatted = self._format_dataset(selected)
        stats["samples_formatted"] = len(formatted)
        stats["format_time"] = time.time() - stage_start
        print(f"  -> {len(formatted)} samples formatted ({stats['format_time']:.1f}s)\n")

        # --- Stage: Export (SFT + DPO pairs) ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Exporting dataset & DPO pairs...")
        stage_start = time.time()
        output_paths = self._export_dataset(
            formatted, curriculum_order=self.config.curriculum_order,
        )

        # Export DPO pairs from selected/rejected
        if rejected:
            formatter = DatasetFormatter(config=FormatterConfig(
                format_mode=self.config.format_mode,
            ))
            dpo_path = formatter.export_dpo_pairs(
                selected, rejected, self.dataset_dir,
            )
            output_paths["dpo_pairs"] = dpo_path

        stats["output_paths"] = output_paths
        stats["export_time"] = time.time() - stage_start
        print(f"  -> Export complete ({stats['export_time']:.1f}s)\n")

        # --- Stage (optional): Two-stage training ---
        if self.config.train_after_export:
            stage += 1
            print(
                f"[PIPELINE] Stage {stage}/{n_stages}: "
                f"Two-stage training (SFT → DPO)..."
            )
            stage_start = time.time()
            adapter_path = self._train(output_paths)
            stats["adapter_path"] = adapter_path
            stats["train_time"] = time.time() - stage_start
            print(f"  -> Training complete ({stats['train_time']:.1f}s)\n")

        stats["total_time"] = time.time() - pipeline_start
        self._print_summary(stats)
        self._save_stats(stats)
        return stats

    # ------------------------------------------------------------------
    # Legacy pipeline (samples_per_prompt == 1)
    # ------------------------------------------------------------------

    def _run_legacy_pipeline(
        self,
        prompts: List[Dict[str, Any]],
        stats: Dict[str, Any],
        stage: int,
        n_stages: int,
        pipeline_start: float,
    ) -> Dict[str, Any]:

        # --- Stage: Collect traces ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Collecting reasoning traces...")
        stage_start = time.time()
        traces = self._collect_traces(prompts)
        stats["traces_collected"] = len(traces)
        stats["collection_time"] = time.time() - stage_start
        print(f"  -> {len(traces)} traces collected ({stats['collection_time']:.1f}s)\n")

        if not traces:
            print("[PIPELINE] No traces collected. Exiting.")
            return stats

        # --- Stage: Quality filtering ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Filtering traces for quality...")
        stage_start = time.time()
        filtered_traces = self._filter_traces(traces)
        stats["traces_after_filter"] = len(filtered_traces)
        stats["traces_filtered_out"] = len(traces) - len(filtered_traces)
        stats["filter_time"] = time.time() - stage_start
        print(
            f"  -> {len(filtered_traces)} traces passed quality filter "
            f"({stats['traces_filtered_out']} removed, {stats['filter_time']:.1f}s)\n"
        )

        # --- Stage: Format dataset ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Formatting dataset...")
        stage_start = time.time()
        trace_dicts = [t if isinstance(t, dict) else t.to_dict() for t in filtered_traces]
        formatted = self._format_dataset(trace_dicts)
        stats["samples_formatted"] = len(formatted)
        stats["format_time"] = time.time() - stage_start
        print(f"  -> {len(formatted)} samples formatted ({stats['format_time']:.1f}s)\n")

        # --- Stage: Export ---
        stage += 1
        print(f"[PIPELINE] Stage {stage}/{n_stages}: Exporting dataset...")
        stage_start = time.time()
        output_paths = self._export_dataset(formatted)
        stats["output_paths"] = output_paths
        stats["export_time"] = time.time() - stage_start
        print(f"  -> Dataset exported ({stats['export_time']:.1f}s)\n")

        # --- Stage (optional): Train ---
        if self.config.train_after_export:
            stage += 1
            print(f"[PIPELINE] Stage {stage}/{n_stages}: Training student model...")
            stage_start = time.time()
            adapter_path = self._train(output_paths)
            stats["adapter_path"] = adapter_path
            stats["train_time"] = time.time() - stage_start
            print(f"  -> Training complete ({stats['train_time']:.1f}s)\n")

        stats["total_time"] = time.time() - pipeline_start
        self._print_summary(stats)
        self._save_stats(stats)
        return stats

    def _generate_prompts(self) -> List[Dict[str, Any]]:
        if self.config.skill_file or self.config.prompt_source == "skills":
            skill_file = self.config.skill_file or self.config.prompt_source
            if skill_file == "skills":
                raise ValueError(
                    "prompt_source='skills' requires skill_file to be set"
                )
            synth_model = self.config.synthesis_model or self.config.model
            prompts_path = os.path.join(self.config.output_dir, "synthesized_prompts.json")
            return generate_prompts_from_skills(
                skill_file=skill_file,
                prompts_per_skill=self.config.prompts_per_skill,
                provider=self.config.provider,
                api_key=self.config.api_key,
                synthesis_model=synth_model,
                save_path=prompts_path,
            )
        elif self.config.prompt_source == "templates":
            return generate_prompts(
                categories=self.config.categories,
                samples_per_category=self.config.samples_per_category,
                seed=self.config.prompt_seed,
            )
        else:
            return generate_prompts_from_file(self.config.prompt_source)

    def _collect_multi_traces(
        self, prompts: List[Dict[str, Any]],
    ) -> Dict[str, List[CollectedTrace]]:
        """Collect N traces per prompt for rejection sampling."""
        collection_config = CollectionConfig(
            model=self.config.model,
            thinking=ThinkingConfig(
                enabled=True,
                budget_tokens=self.config.thinking_budget,
            ),
            max_output_tokens=self.config.max_output_tokens,
            batch_size=self.config.batch_size,
            max_concurrent=self.config.max_concurrent,
            rate_limit_rpm=self.config.rate_limit_rpm,
            samples_per_prompt=self.config.samples_per_prompt,
            compliance_mode=self.config.compliance_mode,
            provider=self.config.provider,
            api_key=self.config.api_key,
        )
        collector = TraceCollector(config=collection_config)
        raw_path = os.path.join(self.traces_dir, "raw_multi_traces.json")
        return collector.collect_multi_sync(
            prompts,
            samples_per_prompt=self.config.samples_per_prompt,
            save_path=raw_path,
        )

    def _verify_and_select(
        self, traces_by_prompt: Dict[str, List[Dict[str, Any]]],
    ):
        """Run dual verification + difficulty scoring."""
        verifier = TraceVerifier(config=VerificationConfig(
            judge_model=self.config.judge_model or self.config.model,
            judge_provider=self.config.provider,
            judge_api_key=self.config.api_key,
            min_judge_score=self.config.min_judge_score,
            difficulty_min=self.config.difficulty_min,
            difficulty_max=self.config.difficulty_max,
        ))
        return verifier.verify_and_select(
            traces_by_prompt,
            fallback_model=self.config.model,
        )

    def _collect_traces(self, prompts: List[Dict[str, Any]]) -> List[CollectedTrace]:
        collection_config = CollectionConfig(
            model=self.config.model,
            thinking=ThinkingConfig(
                enabled=True,
                budget_tokens=self.config.thinking_budget,
            ),
            max_output_tokens=self.config.max_output_tokens,
            batch_size=self.config.batch_size,
            max_concurrent=self.config.max_concurrent,
            rate_limit_rpm=self.config.rate_limit_rpm,
            compliance_mode=self.config.compliance_mode,
            provider=self.config.provider,
            api_key=self.config.api_key,
        )

        collector = TraceCollector(config=collection_config)
        traces_path = os.path.join(self.traces_dir, "raw_traces.json")
        traces = collector.collect_traces_sync(prompts, save_path=traces_path)

        # Log collection stats
        cstats = collector.get_collection_stats()
        print(f"  Collection stats: {json.dumps(cstats, indent=2)}")

        return traces

    def _filter_traces(self, traces) -> list:
        """Apply quality filters to collected traces."""
        filtered = []
        for t in traces:
            if isinstance(t, dict):
                thinking = t.get("thinking_text", "")
                output = t.get("output_text", "")
            else:
                thinking = t.thinking_text
                output = t.output_text

            if len(thinking) < self.config.min_thinking_chars:
                continue
            if len(output) < self.config.min_output_chars:
                continue
            if len(output) > self.config.max_output_chars:
                continue
            filtered.append(t)
        return filtered

    def _format_dataset(self, trace_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatter_config = FormatterConfig(
            format_mode=self.config.format_mode,
            max_thinking_tokens=self.config.max_thinking_tokens,
            include_system_prompt=self.config.include_system_prompt,
            train_split_ratio=self.config.train_split_ratio,
        )
        formatter = DatasetFormatter(config=formatter_config)
        return formatter.format_traces(trace_dicts)

    def _train(self, output_paths: Dict[str, str]) -> str:
        from robinhood.trainer import DistillTrainer, TrainConfig, LoraConfig

        train_path = output_paths.get("train")
        val_path = output_paths.get("validation")
        dpo_path = output_paths.get("dpo_pairs")

        if not train_path:
            raise ValueError("No train.json found in exported paths — cannot train")

        train_dir = os.path.join(self.config.output_dir, "trained")
        train_config = TrainConfig(
            base_model=self.config.base_model,
            dataset_path=train_path,
            val_dataset_path=val_path,
            output_dir=train_dir,
            lora=LoraConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            ),
            num_epochs=self.config.train_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.train_grad_accum,
            learning_rate=self.config.train_lr,
            load_in_4bit=self.config.load_in_4bit,
            max_seq_length=self.config.max_output_tokens,
            curriculum_order=self.config.curriculum_order,
            dpo_dataset_path=dpo_path,
            dpo_beta=self.config.dpo_beta,
            dpo_epochs=self.config.dpo_epochs,
            dpo_learning_rate=self.config.dpo_lr,
        )

        trainer = DistillTrainer(config=train_config)
        return trainer.train()

    def _export_dataset(
        self,
        formatted: List[Dict[str, Any]],
        curriculum_order: bool = False,
    ) -> Dict[str, str]:
        formatter_config = FormatterConfig(
            format_mode=self.config.format_mode,
            max_thinking_tokens=self.config.max_thinking_tokens,
            include_system_prompt=self.config.include_system_prompt,
            train_split_ratio=self.config.train_split_ratio,
        )
        formatter = DatasetFormatter(config=formatter_config)

        if self.config.export_format == "huggingface":
            path = formatter.export_huggingface(formatted, self.dataset_dir)
            return {"dataset": path}
        else:
            return formatter.export_for_platform(
                formatted, self.dataset_dir,
                curriculum_order=curriculum_order,
            )

    def _print_summary(self, stats: Dict[str, Any]):
        full = self._is_full_pipeline()
        print(f"\n{'='*70}")
        print(f"  PIPELINE SUMMARY: {self.config.run_name}")
        print(f"{'='*70}")
        print(f"  Prompts generated:    {stats.get('prompts_generated', 0)}")
        print(f"  Traces collected:     {stats.get('traces_collected', 0)}")

        if full:
            print(f"  Prompts with traces:  {stats.get('prompts_with_traces', 0)}")
            print(f"  Selected (verified):  {stats.get('selected_traces', 0)}")
            print(f"  Rejected (for DPO):   {stats.get('rejected_traces', 0)}")
            if "difficulty_mean" in stats:
                print(
                    f"  Difficulty (mean):    {stats['difficulty_mean']:.3f} "
                    f"[{stats['difficulty_min']:.3f} – {stats['difficulty_max']:.3f}]"
                )
        else:
            print(f"  Traces filtered out:  {stats.get('traces_filtered_out', 0)}")

        print(f"  Samples formatted:    {stats.get('samples_formatted', 0)}")
        print(f"  Total time:           {stats.get('total_time', 0):.1f}s")
        print(f"  Output directory:     {self.config.output_dir}")
        if stats.get("output_paths"):
            for name, path in stats["output_paths"].items():
                print(f"    {name}: {path}")
        if stats.get("adapter_path"):
            print(f"  Trained adapter:      {stats['adapter_path']}")
            print(f"  Training time:        {stats.get('train_time', 0):.1f}s")
        print(f"{'='*70}\n")

    def _save_stats(self, stats: Dict[str, Any]):
        stats_path = os.path.join(self.config.output_dir, "run_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)


def run_from_existing_traces(
    traces_path: str,
    output_dir: str,
    format_mode: str = "thinking_and_output",
    export_format: str = "platform",
) -> Dict[str, Any]:
    """
    Run formatting and export on previously collected traces.

    Useful for reformatting traces without re-collecting them.
    """
    print(f"[PIPELINE] Loading traces from {traces_path}...")
    traces = ClaudeTraceCollector.load_traces(traces_path)
    print(f"  -> Loaded {len(traces)} traces")

    formatter = DatasetFormatter(config=FormatterConfig(format_mode=format_mode))
    formatted = formatter.format_traces(traces)
    print(f"  -> Formatted {len(formatted)} samples")

    os.makedirs(output_dir, exist_ok=True)
    if export_format == "huggingface":
        paths = {"dataset": formatter.export_huggingface(formatted, output_dir)}
    else:
        paths = formatter.export_for_platform(formatted, output_dir)

    return {"samples": len(formatted), "paths": paths}


_CONSENT_PROMPT = """\
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


def _prompt_consent() -> bool:
    """Display the consent prompt and return True if the user confirms."""
    print(_CONSENT_PROMPT)
    try:
        answer = input("Do you confirm? [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="robinhood — Reasoning trace collection & dataset formatting"
    )
    parser.add_argument(
        "--config", type=str, help="Path to pipeline config JSON file"
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["anthropic", "openai", "openrouter"],
        help="LLM provider (default: auto-detected from model name / env vars)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (default: read from ANTHROPIC_API_KEY, OPENAI_API_KEY, "
             "or OPENROUTER_API_KEY env var depending on provider)"
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514). Examples: "
             "gpt-4o, anthropic/claude-sonnet-4, deepseek/deepseek-r1"
    )
    parser.add_argument(
        "--samples-per-category", type=int, default=10,
        help="Number of prompts per category (default: 10)"
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=10000,
        help="Thinking token budget (default: 10000)"
    )
    parser.add_argument(
        "--format", type=str, default="thinking_and_output",
        choices=["thinking_and_output", "output_only", "reasoning_augmented", "multi_turn"],
        help="Dataset format mode (default: thinking_and_output)"
    )
    parser.add_argument(
        "--export", type=str, default="platform",
        choices=["platform", "huggingface"],
        help="Export format (default: platform)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./robinhood_output",
        help="Output directory (default: ./robinhood_output)"
    )
    parser.add_argument(
        "--categories", type=str, nargs="*",
        help="Prompt categories to include (default: all)"
    )
    parser.add_argument(
        "--prompt-file", type=str,
        help="Load prompts from file instead of templates"
    )
    parser.add_argument(
        "--skill-file", type=str,
        help="Path to a skill definition file (JSON/YAML). Claude will synthesize "
             "targeted prompts from the skill definitions."
    )
    parser.add_argument(
        "--prompts-per-skill", type=int, default=20,
        help="Number of prompts to synthesize per skill (default: 20)"
    )
    parser.add_argument(
        "--synthesis-model", type=str, default=None,
        help="Model to use for prompt synthesis (default: same as --model)"
    )
    parser.add_argument(
        "--reformat", type=str,
        help="Path to existing traces JSON to reformat (skips collection)"
    )
    parser.add_argument(
        "--rate-limit", type=int, default=50,
        help="Rate limit in requests per minute (default: 50)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Maximum concurrent API requests (default: 10)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this run (default: auto-generated)"
    )

    # Compliance and consent
    parser.add_argument(
        "--compliance", action="store_true",
        help="Enable compliance mode: conservative concurrency (5), strict rate "
             "limits (30 RPM). Recommended for all production runs."
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive consent prompt (for CI/automated environments "
             "where compliance has already been confirmed)"
    )

    # Rejection sampling & verification
    parser.add_argument(
        "--samples-per-prompt", type=int, default=1,
        help="Number of traces to collect per prompt for rejection sampling. "
             "Set to 8-16 for full pipeline with verification (default: 1)"
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Model for LLM-as-judge verification (default: same as --model)"
    )
    parser.add_argument(
        "--min-judge-score", type=float, default=6.0,
        help="Minimum LLM-as-judge score for non-verifiable tasks (default: 6.0)"
    )
    parser.add_argument(
        "--difficulty-min", type=float, default=0.05,
        help="Minimum difficulty to keep (filters out too-easy prompts, default: 0.05)"
    )
    parser.add_argument(
        "--difficulty-max", type=float, default=0.95,
        help="Maximum difficulty to keep (filters out too-hard prompts, default: 0.95)"
    )

    # Curriculum ordering (Light-R1 style)
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Sort training data easy-to-hard (Light-R1 curriculum learning)"
    )

    # Training flags
    parser.add_argument(
        "--train", action="store_true",
        help="After exporting the dataset, finetune a student model using Unsloth"
    )
    parser.add_argument(
        "--base-model", type=str, default="unsloth/Qwen3-14B",
        help="Student model to finetune (default: unsloth/Qwen3-14B)"
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
        "--train-epochs", type=float, default=3.0,
        help="Training epochs (default: 3.0)"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantized training"
    )

    # REDI contrastive Stage 2
    parser.add_argument(
        "--dpo-beta", type=float, default=0.1,
        help="DPO beta parameter for Stage 2 contrastive training (default: 0.1)"
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

    # Consent prompt (skip for --reformat since it doesn't query a provider)
    if not args.reformat and not args.yes:
        if not _prompt_consent():
            print("\nConsent not given. Exiting.")
            sys.exit(0)

    # Handle reformat mode
    if args.reformat:
        run_from_existing_traces(
            traces_path=args.reformat,
            output_dir=args.output_dir,
            format_mode=args.format,
            export_format=args.export,
        )
        return

    # Build config
    if args.config:
        config = PipelineConfig.from_file(args.config)
    else:
        # Determine prompt source
        if args.skill_file:
            prompt_source = "skills"
        elif args.prompt_file:
            prompt_source = args.prompt_file
        else:
            prompt_source = "templates"

        model_short = args.model.split("/")[-1].split("-")[0] if "/" in args.model else args.model.split("-")[1] if "-" in args.model else args.model
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
            model=args.model,
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


if __name__ == "__main__":
    main()
