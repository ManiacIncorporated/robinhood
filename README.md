<p align="center">
  <h1 align="center">robinhood</h1>
  <p align="center">
    <strong>Reasoning trace collection &amp; dataset formatting for your own teacher models.</strong>
  </p>
  <p align="center">
    Open-source toolkit for collecting reasoning traces from LLM teachers, building high-quality finetuning datasets, and training student models with Unsloth.
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#how-it-works">How It Works</a> &middot;
    <a href="#skill-files">Skill Files</a> &middot;
    <a href="#training">Training</a> &middot;
    <a href="#what-the-research-says-how-real-distillation-datasets-are-curated">Research</a> &middot;
    <a href="#compliance">Compliance</a> &middot;
    <a href="#api-reference">API Reference</a>
  </p>
</p>

---

## What is this?

**robinhood** is a research-grade toolkit for reasoning trace distillation — collecting chain-of-thought traces from a teacher model and converting them into high-quality finetuning datasets for a smaller student model. It implements techniques from DeepSeek R1, MiniMax M2, OpenThoughts, Light-R1, and REDI as a clean, extensible Python library.

The full pipeline:

1. **Generate** diverse, targeted prompts (from templates or your own [skill files](#skill-files))
2. **Collect** reasoning traces from teacher models with extended thinking enabled
3. **Verify** traces with dual pipelines (automated for code/math, LLM-as-judge for the rest)
4. **Format** the verified traces into SFT-ready datasets (4 format modes)
5. **Train** a student model in two stages: curriculum SFT then contrastive DPO

Supports **Anthropic**, **OpenAI**, and **OpenRouter** APIs when permitted by their terms of service. Users are responsible for ensuring compliance with provider terms before collecting traces.

## Quick Start

### Install

```bash
pip install -e .                          # from source
pip install robinhood-distill             # from PyPI (when published)
pip install -e ".[train]"                 # include training deps (unsloth, trl)
pip install -e ".[all]"                   # everything
```

After install the `robinhood` command is available globally:

```bash
robinhood --version
robinhood collect --help
robinhood train   --help
robinhood reformat --help
```

### Collect traces + train a model (full pipeline)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

robinhood collect \
    --model claude-sonnet-4-20250514 \
    --samples-per-category 100 \
    --thinking-budget 16000 \
    --samples-per-prompt 8 \
    --curriculum \
    --train \
    --base-model unsloth/Qwen3-14B \
    --compliance
```

This collects 8 traces per prompt, verifies each one (automated for math/code, LLM-as-judge for everything else), picks the best, scores difficulty from pass rates, exports curriculum-ordered SFT data + DPO pairs from rejected traces, then trains in two stages: curriculum SFT followed by REDI contrastive refinement.

### Simple mode (no rejection sampling)

```bash
robinhood collect \
    --model claude-sonnet-4-20250514 \
    --samples-per-category 100 \
    --train \
    --base-model unsloth/Qwen3-14B
```

### Or step by step in Python

```python
import robinhood

# 1. Generate prompts
prompts = robinhood.generate_prompts(samples_per_category=50)

# 2. Collect reasoning traces
collector = robinhood.TraceCollector(
    config=robinhood.CollectionConfig(
        model="claude-sonnet-4-20250514",
        thinking=robinhood.ThinkingConfig(budget_tokens=10000),
    )
)
traces = collector.collect_traces_sync(prompts, save_path="traces.json")

# 3. Format into a training dataset
formatter = robinhood.DatasetFormatter(
    config=robinhood.FormatterConfig(format_mode="thinking_and_output")
)
dataset = formatter.format_traces([t.to_dict() for t in traces])
formatter.export_for_platform(dataset, output_dir="./dataset")

# 4. Train a student model
trainer = robinhood.DistillTrainer(robinhood.TrainConfig(
    base_model="unsloth/Qwen3-14B",
    dataset_path="./dataset/train.json",
))
trainer.train()
```

## How It Works

### Full closed-loop pipeline (v0.2)

When `--samples-per-prompt` is set to > 1, robinhood runs the full Chinese distillation pipeline matching DeepSeek R1, MiniMax M2, Light-R1, and REDI:

```
                            ┌────────────────────┐
                            │   Frontier Model    │
                            │ (Claude/GPT-4/etc)  │
                            └─────────┬──────────┘
                                      │ N completions per prompt
                                      ▼
┌──────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│    Prompts   │───>│  Multi-Answer         │───>│  Dual Verification   │
│              │    │  Trace Collector      │    │  (DeepSeek / MiniMax)│
│ - Templates  │    │  (OpenThoughts-style) │    │                      │
│ - Skill files│    │                      │    │  Math/Code: automated│
│ - Custom     │    │  8-16 traces/prompt  │    │  Other: LLM-as-judge │
└──────────────┘    └──────────────────────┘    └──────────┬───────────┘
                                                           │
                                            ┌──────────────┴───────────────┐
                                            │                              │
                                            ▼                              ▼
                                   ┌────────────────┐            ┌─────────────────┐
                                   │ Best trace per  │            │ Rejected traces  │
                                   │ prompt (selected)│            │ (saved for DPO)  │
                                   │ + difficulty     │            │                 │
                                   │   score          │            │                 │
                                   └───────┬─────────┘            └────────┬────────┘
                                           │                               │
                                           ▼                               │
                                  ┌─────────────────┐                      │
                                  │   Formatter      │                      │
                                  │                  │                      │
                                  │ Curriculum order │                      │
                                  │ (easy → hard)    │                      │
                                  └───────┬──────────┘                      │
                                          │                                │
                             ┌────────────┴───────────┐                    │
                             │                        │                    │
                             ▼                        ▼                    ▼
                    ┌────────────────┐      ┌─────────────────┐   ┌──────────────┐
                    │ Stage 1: SFT   │─────>│ Stage 2: DPO    │   │ DPO Pairs    │
                    │ (Light-R1      │      │ (REDI           │<──│ chosen +     │
                    │  curriculum)   │      │  contrastive)   │   │ rejected     │
                    └────────────────┘      └────────┬────────┘   └──────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │ Your Model   │
                                            │ (LoRA adapter)│
                                            └──────────────┘
```

### Standard pipeline (legacy)

With `--samples-per-prompt 1` (the default), robinhood runs the simpler single-sample pipeline:

```
Prompts ──> Trace Collector ──> Quality Filter ──> Formatter ──> Trainer ──> Your Model
```

### The reasoning trace is the secret sauce

When Claude thinks through a problem with extended thinking, it produces a chain of reasoning *before* the final answer. By training a smaller model on both the reasoning **and** the answer, the student learns to **think like the teacher** — not just mimic its outputs.

```
Teacher (Claude with extended thinking):

  <think>
  The user is asking about the integral of sin(x)cos(x).
  I can use the substitution u = sin(x), du = cos(x)dx.
  Then the integral becomes ∫ u du = u²/2 = sin²(x)/2 + C.
  Let me verify: d/dx[sin²(x)/2] = 2sin(x)cos(x)/2 = sin(x)cos(x). ✓
  </think>

  The integral of sin(x)cos(x)dx = sin²(x)/2 + C.

Student model learns to reproduce BOTH the reasoning AND the answer.
```

## Multi-Provider Support

robinhood works with any major LLM provider. The provider is auto-detected from the model name or set explicitly.

```bash
# Anthropic (native extended thinking)
robinhood collect --model claude-sonnet-4-20250514

# OpenAI
robinhood collect --provider openai --model gpt-4o --api-key sk-...

# OpenRouter (access hundreds of models)
robinhood collect --model deepseek/deepseek-r1
robinhood collect --model anthropic/claude-sonnet-4
```

| Provider | Env Var | Extended Thinking |
|----------|---------|-------------------|
| Anthropic | `ANTHROPIC_API_KEY` | Native thinking blocks |
| OpenAI | `OPENAI_API_KEY` | `<think>` tag prompting |
| OpenRouter | `OPENROUTER_API_KEY` | `<think>` tag prompting |

## Skill Files

Instead of generic prompts, define **exactly what you want the model to be good at** in a skill file. robinhood uses Claude to synthesize diverse, targeted prompts for each skill.

```json
{
    "name": "Medical Chart Summarizer",
    "description": "Summarize patient charts into structured clinical notes",
    "system_prompt": "You are a clinical documentation specialist...",
    "skills": [
        {
            "name": "diagnosis_extraction",
            "description": "Extract diagnoses from unstructured chart text",
            "difficulty_levels": ["straightforward", "ambiguous", "adversarial"],
            "example_inputs": ["Patient presents with chest pain..."],
            "constraints": ["Only include explicitly stated diagnoses"]
        }
    ]
}
```

```bash
robinhood collect \
    --skill-file my_skills.json \
    --prompts-per-skill 50 \
    --model claude-sonnet-4-20250514
```

Each skill gets prompts generated at every difficulty level, covering the full capability surface. See `robinhood/examples/` for complete examples:
- `medical_charting.json` — clinical documentation
- `code_review.json` — code review and bug detection
- `legal_analysis.yaml` — contract analysis (YAML format)

## Dataset Format Modes

| Mode | Target Format | Best For |
|------|--------------|----------|
| `thinking_and_output` | `<think>reasoning</think> answer` | Maximum capability transfer |
| `output_only` | `answer` | Simple, cheap to train |
| `reasoning_augmented` | `<STRATEGY>hint</STRATEGY> answer` | Balanced cost/quality |
| `multi_turn` | Reasoning as separate assistant turn | Models without `<think>` support |

```bash
# Collect with full reasoning
robinhood collect --format thinking_and_output

# Re-format existing traces to output-only (no re-collection needed)
robinhood reformat traces.json --format output_only
```

## Training

### Two-stage training (full pipeline)

When the full pipeline generates both SFT data and DPO pairs, training runs in two stages:

```bash
robinhood collect \
    --model claude-sonnet-4-20250514 \
    --samples-per-category 200 \
    --samples-per-prompt 8 \
    --curriculum \
    --train \
    --base-model unsloth/Qwen3-14B \
    --lora-rank 32 \
    --train-epochs 3
```

### Standalone two-stage training

```bash
# Stage 1: Curriculum SFT
robinhood train \
    --base-model unsloth/Qwen3-14B \
    --dataset ./robinhood_output/dataset/train.json \
    --curriculum \
    --lora-rank 32 \
    --epochs 3 \
    --lr 3e-4

# Stage 1 + Stage 2: Curriculum SFT → REDI contrastive
robinhood train \
    --base-model unsloth/Qwen3-14B \
    --dataset ./robinhood_output/dataset/train.json \
    --dpo-dataset ./robinhood_output/dataset/dpo_pairs.jsonl \
    --curriculum \
    --lora-rank 32 \
    --epochs 3 \
    --dpo-epochs 1 \
    --dpo-beta 0.1
```

### From Python

```python
from robinhood import DistillTrainer, TrainConfig, LoraConfig

trainer = DistillTrainer(TrainConfig(
    base_model="unsloth/Qwen3-14B",
    dataset_path="./train.json",
    lora=LoraConfig(rank=32, alpha=64),
    num_epochs=3.0,
    load_in_4bit=True,
    train_on_responses_only=True,
    curriculum_order=True,
    dpo_dataset_path="./dpo_pairs.jsonl",
))
adapter_path = trainer.train()
```

**Training features:**
- **Curriculum ordering** — samples sorted easy-to-hard by difficulty (Light-R1)
- **Two-stage training** — SFT then REDI contrastive DPO using rejected traces
- **Unsloth** for 2x faster LoRA finetuning
- **4-bit QLoRA** by default for large models
- **Response-only masking** — only trains on the assistant output, not input tokens
- **Wandb** logging support (`--report-to wandb`)
- **Checkpoint saving** at configurable intervals

## Scaling

| Scale | Prompts | Est. Cost (Sonnet) | Est. Time (50 RPM) |
|-------|---------|---------------------|---------------------|
| Test | 50 | ~$5 | ~1 min |
| Small | 1,000 | ~$100 | ~20 min |
| Medium | 10,000 | ~$1,000 | ~3.3 hours |
| Large | 100,000 | ~$10,000 | ~33 hours |

**Cost optimization:** Use Haiku for simple tasks, Sonnet/Opus for complex reasoning. Lower thinking budgets for straightforward prompts. Use prompt caching where available. Always respect provider rate limits — use `--compliance` mode for conservative defaults.

## Full CLI Reference

```bash
# Full pipeline with rejection sampling + verification + training
robinhood collect \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --samples-per-category 100 \
    --thinking-budget 16000 \
    --samples-per-prompt 8 \
    --curriculum \
    --format thinking_and_output \
    --skill-file skills.json \
    --prompts-per-skill 50 \
    --export platform \
    --output-dir ./output \
    --train \
    --base-model unsloth/Qwen3-14B \
    --lora-rank 32 \
    --dpo-beta 0.1 \
    --dpo-epochs 1 \
    --compliance

# Simple pipeline (single trace per prompt, no verification)
robinhood collect \
    --model claude-sonnet-4-20250514 \
    --samples-per-category 100 \
    --train \
    --base-model unsloth/Qwen3-14B

# Standalone two-stage trainer
robinhood train \
    --base-model unsloth/Qwen3-14B \
    --dataset ./train.json \
    --val-dataset ./val.json \
    --dpo-dataset ./dpo_pairs.jsonl \
    --curriculum \
    --output-dir ./trained \
    --lora-rank 32 \
    --epochs 3 \
    --dpo-epochs 1 \
    --dpo-beta 0.1 \
    --report-to wandb

# Reformat existing traces without re-collecting
robinhood reformat traces.json --format output_only --output-dir ./new
```

## API Reference

| Module | Key Exports |
|--------|-------------|
| `robinhood` | `TraceCollector`, `DatasetFormatter`, `TraceVerifier`, `RobinhoodPipeline`, `DistillTrainer` |
| `robinhood.providers` | `LLMClient`, `LLMResponse`, `detect_provider` |
| `robinhood.trace_collector` | `TraceCollector`, `CollectionConfig`, `ThinkingConfig`, `CollectedTrace` |
| `robinhood.verification` | `TraceVerifier`, `VerificationConfig`, `VerifiedTrace` |
| `robinhood.dataset_formatter` | `DatasetFormatter`, `FormatterConfig` |
| `robinhood.prompt_sources` | `generate_prompts`, `generate_prompts_from_file`, `PROMPT_TEMPLATES` |
| `robinhood.skills` | `SkillSet`, `Skill`, `SkillPromptSynthesizer`, `generate_prompts_from_skills` |
| `robinhood.pipeline` | `RobinhoodPipeline`, `PipelineConfig`, `run_from_existing_traces` |
| `robinhood.trainer` | `DistillTrainer`, `TrainConfig`, `LoraConfig` |

## Project Structure

```
robinhood/
  __init__.py          # Public API surface
  providers.py         # Multi-provider LLM client (Anthropic/OpenAI/OpenRouter)
  trace_collector.py   # Async trace collection with multi-answer sampling
  verification.py      # Dual verification: deterministic + LLM-as-judge
  dataset_formatter.py # 4 format modes + curriculum ordering + DPO pair export
  prompt_sources.py    # Built-in prompt templates (7 categories)
  skills.py            # Skill file loading + LLM-powered prompt synthesis
  pipeline.py          # Full closed-loop orchestration + CLI
  trainer.py           # Two-stage training: curriculum SFT + REDI DPO
  examples/
    medical_charting.json
    code_review.json
    legal_analysis.yaml
```

## What The Research Says: How Real Distillation Datasets Are Curated

We compiled findings from DeepSeek R1, MiniMax M2, OpenThoughts, LIMO, Light-R1, s1, DLCoT, and REDI. Full details in [`robinhood/RESEARCH.md`](robinhood/RESEARCH.md). The highlights:

### Quality beats quantity by a wide margin

- **LIMO** achieved 57.1% on AIME with just **817 examples** (vs 6.5% with previous SFT). 1% of the data, 10x the performance.
- **s1** used **1,000 curated examples** to beat o1-preview on competition math.
- **DeepSeek R1** used 800K traces, but each one was **verified for correctness** (test cases for code, reference answers for math, reward models for everything else).

### MiniMax's two-pipeline approach

MiniMax splits data into **Verifiable** (math, code — automated checking) and **Non-Verifiable** (reasoning, science — LLM-as-judge) pipelines. Key findings:
- Harder queries are more valuable per training token
- Math/code data improves *all* tasks, even creative writing
- Format diversity prevents benchmark overfitting
- Rules + LLM-as-judge for data cleaning is essential

### OpenThoughts: 1000 ablation experiments

The most rigorous study of what matters. Surprising results:
- **Sample 16 answers per question** from the teacher, keep the best — 16x more data *and* higher quality
- **Source concentration beats diversity** — 1-2 great question sources > 16 mediocre ones
- **Better benchmark scores don't make a better teacher** — QwQ-32B outperformed DeepSeek-R1 as a teacher
- Answer filtering beyond correctness checks provided **no significant improvement**

### Curriculum learning works (Light-R1)

Training easy-to-hard instead of random order significantly improves results. Light-R1-14B beat models 5x its size by using three progressive stages: SFT -> DPO -> RL, each with increasing difficulty.

### Don't throw away wrong answers (REDI)

Standard rejection sampling discards incorrect traces. REDI showed that keeping them for a contrastive learning stage after SFT lets a 1.5B model trained on 131K examples match 800K-example models.

### Practical playbook (all implemented in robinhood v0.2)

| Stage | What to do | robinhood feature |
|-------|-----------|-------------------|
| **Collection** | Sample 8-16x answers per prompt | `--samples-per-prompt 8` |
| **Verification** | Math/code: automated; other: LLM-as-judge | `TraceVerifier` dual pipelines |
| **Difficulty** | Filter to intermediate difficulty, score from pass rate | `--difficulty-min 0.05 --difficulty-max 0.95` |
| **Formatting** | Curriculum order (easy → hard) | `--curriculum` |
| **Training** | SFT then contrastive DPO with rejected traces | `--dpo-dataset` / auto from pipeline |
| **Teacher** | Test multiple teachers | `--model` supports Anthropic/OpenAI/OpenRouter |

## Compliance

robinhood is a research tool for reasoning trace collection and dataset curation. It supports multiple LLM providers but **does not grant you the right to use any provider's outputs for training**.

Before using robinhood, you must:

1. **Review your provider's terms of service** to confirm that using model outputs for training is permitted for your use case
2. **Respect rate limits** — use `--compliance` mode (or `compliance_mode=True` in `PipelineConfig`) for conservative concurrency defaults that stay well within published limits
3. **Use a single authorized account** — robinhood does not support or encourage multi-account usage, rate-limit circumvention, or any form of access-restriction bypass
4. **Evaluate safety independently** — student models trained via distillation may not inherit the teacher's safety training; evaluate your student model's safety properties before deployment

The CLI will prompt you to confirm compliance on first use. Pass `--yes` to skip in automated environments where you have already confirmed.

| Provider | Output-for-training policy | Check here |
|----------|---------------------------|------------|
| Anthropic | Review current usage policy | [anthropic.com/policies](https://www.anthropic.com/policies) |
| OpenAI | Review current usage policy | [openai.com/policies](https://openai.com/policies) |
| OpenRouter | Varies by underlying model | [openrouter.ai/terms](https://openrouter.ai/terms) |

## License

MIT
