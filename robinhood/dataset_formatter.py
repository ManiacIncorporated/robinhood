"""
robinhood.dataset_formatter — Dataset Formatter

Converts collected reasoning traces into training-ready datasets for SFT.
Supports multiple output formats used in distillation research:

1. **thinking_and_output**: Train the student to reproduce both the reasoning
   trace and the final output: <think>...</think> answer
2. **output_only**: Use Claude's output as high-quality labels, discarding the
   reasoning trace (traditional distillation).
3. **reasoning_augmented**: Prepend a condensed version of the reasoning as a
   strategy hint in the training target (RGT-style).
4. **multi_turn**: Format as multi-turn conversations where the thinking is an
   explicit assistant turn before the final answer.

Compatible with the platform's existing SFT pipeline and HuggingFace datasets.
"""

import json
import os
import hashlib
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field


FormatMode = Literal[
    "thinking_and_output",
    "output_only",
    "reasoning_augmented",
    "multi_turn",
]


@dataclass
class FormatterConfig:
    """Configuration for dataset formatting."""
    format_mode: FormatMode = "thinking_and_output"
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    strategy_start_token: str = "<STRATEGY>"
    strategy_end_token: str = "</STRATEGY>"
    max_thinking_tokens: Optional[int] = None
    include_system_prompt: bool = True
    include_raw_metadata: bool = False
    train_split_ratio: float = 0.9
    shuffle_seed: int = 42


class DatasetFormatter:
    """
    Formats collected reasoning traces into SFT training datasets.

    Takes CollectedTrace dicts (from TraceCollector) and produces datasets
    in the platform's expected format:
        {
            "input": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
            "output": {"choices": [{"message": {"content": ..., "role": "assistant"}}]}
        }
    """

    def __init__(self, config: FormatterConfig = None):
        self.config = config or FormatterConfig()

    def format_traces(
        self,
        traces: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format a list of trace dicts into SFT training samples.

        Args:
            traces: List of trace dicts (from ClaudeTraceCollector.to_dict())
            output_path: If provided, save formatted dataset to this path.

        Returns:
            List of formatted training samples.
        """
        formatter_fn = {
            "thinking_and_output": self._format_thinking_and_output,
            "output_only": self._format_output_only,
            "reasoning_augmented": self._format_reasoning_augmented,
            "multi_turn": self._format_multi_turn,
        }[self.config.format_mode]

        formatted = []
        for trace in traces:
            try:
                sample = formatter_fn(trace)
                if sample:
                    formatted.append(sample)
            except Exception as e:
                print(f"[FORMATTER] Error formatting trace {trace.get('trace_id', '?')}: {e}")

        print(
            f"[FORMATTER] Formatted {len(formatted)}/{len(traces)} traces "
            f"using mode '{self.config.format_mode}'"
        )

        if output_path:
            self._save_dataset(formatted, output_path)

        return formatted

    def _build_input_messages(self, trace: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the input message list from a trace."""
        messages = []
        if self.config.include_system_prompt and trace.get("system_prompt"):
            messages.append({"role": "system", "content": trace["system_prompt"]})
        messages.append({"role": "user", "content": trace["user_message"]})
        return messages

    def _build_output(self, content: str) -> Dict[str, Any]:
        """Build the output dict in the platform's expected format."""
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "role": "assistant",
                    }
                }
            ]
        }

    def _truncate_thinking(self, thinking: str) -> str:
        """Optionally truncate thinking text to max tokens (approximated by chars / 4)."""
        if self.config.max_thinking_tokens and thinking:
            max_chars = self.config.max_thinking_tokens * 4
            if len(thinking) > max_chars:
                thinking = thinking[:max_chars] + "..."
        return thinking

    def _format_thinking_and_output(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format: Train student to produce both reasoning and output.
        Target: <think>reasoning</think>output
        
        This is the primary distillation format - the student learns to reason
        like Claude by reproducing the full chain of thought.
        """
        thinking = self._truncate_thinking(trace.get("thinking_text", ""))
        output = trace.get("output_text", "")
        if not output:
            return None

        ts = self.config.think_start_token
        te = self.config.think_end_token
        target = f"{ts}\n{thinking}\n{te}\n{output}" if thinking else output

        sample = {
            "input": self._build_input_messages(trace),
            "output": self._build_output(target),
            "metadata": {
                "format_mode": "thinking_and_output",
                "source_model": trace.get("model", "unknown"),
                "category": trace.get("prompt_category", "general"),
                "trace_id": trace.get("trace_id", ""),
                "prompt_id": trace.get("prompt_id", ""),
                "thinking_chars": len(thinking),
                "output_chars": len(output),
                "difficulty": trace.get("difficulty", 0.5),
                "verification_score": trace.get("verification_score", 0.0),
                "verification_method": trace.get("verification_method", ""),
            },
        }
        return sample

    def _format_output_only(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format: Traditional distillation, only the final output.
        Target: output (no reasoning)
        
        The simplest form of distillation - use Claude's high-quality outputs
        as training labels. The reasoning improves output quality even though
        it's not included in the training target.
        """
        output = trace.get("output_text", "")
        if not output:
            return None

        sample = {
            "input": self._build_input_messages(trace),
            "output": self._build_output(output),
            "metadata": {
                "format_mode": "output_only",
                "source_model": trace.get("model", "unknown"),
                "category": trace.get("prompt_category", "general"),
                "trace_id": trace.get("trace_id", ""),
                "prompt_id": trace.get("prompt_id", ""),
                "difficulty": trace.get("difficulty", 0.5),
                "verification_score": trace.get("verification_score", 0.0),
                "verification_method": trace.get("verification_method", ""),
            },
        }
        return sample

    def _format_reasoning_augmented(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format: RGT-style with strategy distillation prefix.
        Target: <STRATEGY>condensed_reasoning</STRATEGY>output
        
        Instead of the full thinking trace, extract the key strategic insight
        and prepend it as a compact hint. Compatible with the platform's
        existing RGT (Rationale-Guided Training) approach.
        """
        thinking = trace.get("thinking_text", "")
        output = trace.get("output_text", "")
        if not output:
            return None

        strategy = self._extract_strategy(thinking, output)

        ss = self.config.strategy_start_token
        se = self.config.strategy_end_token
        target = f"{ss}{strategy}{se}{output}" if strategy else output

        sample = {
            "input": self._build_input_messages(trace),
            "output": self._build_output(target),
            "metadata": {
                "format_mode": "reasoning_augmented",
                "source_model": trace.get("model", "unknown"),
                "category": trace.get("prompt_category", "general"),
                "trace_id": trace.get("trace_id", ""),
                "prompt_id": trace.get("prompt_id", ""),
                "strategy_length": len(strategy),
                "difficulty": trace.get("difficulty", 0.5),
                "verification_score": trace.get("verification_score", 0.0),
                "verification_method": trace.get("verification_method", ""),
            },
        }
        return sample

    def _format_multi_turn(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format: Multi-turn where thinking is an explicit assistant turn.
        
        Input messages include an assistant "thinking" turn followed by a user
        "continue" turn, training the model to produce the final output after
        being shown reasoning.
        """
        thinking = self._truncate_thinking(trace.get("thinking_text", ""))
        output = trace.get("output_text", "")
        if not output:
            return None

        messages = self._build_input_messages(trace)
        if thinking:
            messages.append({
                "role": "assistant",
                "content": f"[Internal reasoning]\n{thinking}",
            })
            messages.append({
                "role": "user",
                "content": "Please provide your final answer based on your reasoning.",
            })

        sample = {
            "input": messages,
            "output": self._build_output(output),
            "metadata": {
                "format_mode": "multi_turn",
                "source_model": trace.get("model", "unknown"),
                "category": trace.get("prompt_category", "general"),
                "trace_id": trace.get("trace_id", ""),
                "prompt_id": trace.get("prompt_id", ""),
                "difficulty": trace.get("difficulty", 0.5),
                "verification_score": trace.get("verification_score", 0.0),
                "verification_method": trace.get("verification_method", ""),
            },
        }
        return sample

    def _extract_strategy(self, thinking: str, output: str) -> str:
        """
        Extract a condensed strategy from the full thinking trace.

        Heuristic: take the last ~200 chars of thinking which typically
        contains the key decision/conclusion. For production use, this
        should be replaced with an LLM-based strategy distillation call
        (see reasoning_preprocessor.py distill_strategy).
        """
        if not thinking:
            return ""
        lines = thinking.strip().split("\n")
        conclusion_lines = []
        total_chars = 0
        for line in reversed(lines):
            if total_chars + len(line) > 500:
                break
            conclusion_lines.insert(0, line)
            total_chars += len(line)
        return "\n".join(conclusion_lines).strip()

    def create_train_val_split(
        self,
        samples: List[Dict[str, Any]],
        curriculum_order: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split formatted samples into train and validation sets.

        When *curriculum_order* is True (Light-R1 style), the training split
        is sorted by difficulty ascending (easy first) instead of shuffled.
        The validation split is always shuffled.
        """
        import random
        rng = random.Random(self.config.shuffle_seed)

        indices = list(range(len(samples)))
        rng.shuffle(indices)

        split_idx = int(len(samples) * self.config.train_split_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        if curriculum_order:
            train_samples.sort(
                key=lambda s: s.get("metadata", {}).get("difficulty", 0.5)
            )
            print(
                f"[FORMATTER] Curriculum ordering applied — "
                f"training samples sorted easy-to-hard"
            )

        return {
            "train": train_samples,
            "validation": val_samples,
        }

    def export_for_platform(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        curriculum_order: bool = False,
    ) -> Dict[str, str]:
        """
        Export formatted dataset in the platform's expected format.
        
        Creates:
            - {output_dir}/train.json: Training split
            - {output_dir}/val.json: Validation split
            - {output_dir}/metadata.json: Dataset metadata and statistics

        When *curriculum_order* is True, the training split is sorted
        easy-to-hard by difficulty (Light-R1 style).
        """
        os.makedirs(output_dir, exist_ok=True)

        splits = self.create_train_val_split(samples, curriculum_order=curriculum_order)

        paths = {}
        for split_name, split_data in splits.items():
            filename = "train.json" if split_name == "train" else "val.json"
            path = os.path.join(output_dir, filename)
            self._save_dataset(split_data, path)
            paths[split_name] = path

        metadata = self._compute_metadata(samples, splits)
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        paths["metadata"] = meta_path

        print(f"[FORMATTER] Exported dataset to {output_dir}:")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Validation: {len(splits['validation'])} samples")

        return paths

    def export_huggingface(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
    ) -> str:
        """
        Export in HuggingFace chat format (messages + completion).

        Each sample becomes:
            {"messages": [...input messages...], "completion": "...target..."}
        """
        os.makedirs(output_dir, exist_ok=True)

        hf_samples = []
        for sample in samples:
            messages = sample.get("input", [])
            output_content = ""
            try:
                output_content = sample["output"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                continue

            hf_sample = {
                "messages": messages + [{"role": "assistant", "content": output_content}],
            }
            hf_samples.append(hf_sample)

        path = os.path.join(output_dir, "dataset.jsonl")
        with open(path, "w") as f:
            for s in hf_samples:
                f.write(json.dumps(s) + "\n")

        print(f"[FORMATTER] Exported {len(hf_samples)} samples to {path} (HuggingFace JSONL)")
        return path

    def export_dpo_pairs(
        self,
        selected: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        output_dir: str,
    ) -> str:
        """
        Export chosen/rejected pairs for REDI-style contrastive training.

        Pairs each selected (verified-correct) trace with the lowest-scoring
        rejected trace from the same prompt, producing a dataset suitable for
        DPO / SimPO / contrastive refinement after the initial SFT stage.

        Returns:
            Path to the saved DPO pairs JSONL file.
        """
        os.makedirs(output_dir, exist_ok=True)

        rejected_by_prompt: Dict[str, List[Dict[str, Any]]] = {}
        for r in rejected:
            pid = r.get("prompt_id") or r.get("metadata", {}).get("prompt_id", "")
            if pid:
                rejected_by_prompt.setdefault(pid, []).append(r)

        pairs = []
        for sel in selected:
            pid = sel.get("prompt_id") or sel.get("metadata", {}).get("prompt_id", "")
            prompt_rejects = rejected_by_prompt.get(pid, [])
            if not prompt_rejects:
                continue

            worst = min(
                prompt_rejects,
                key=lambda r: r.get("verification_score", r.get("metadata", {}).get("verification_score", 0)),
            )

            input_messages = self._build_input_messages(sel)

            chosen_output = self._build_target_content(sel)
            rejected_output = self._build_target_content(worst)

            if chosen_output and rejected_output and chosen_output != rejected_output:
                pairs.append({
                    "prompt": input_messages,
                    "chosen": chosen_output,
                    "rejected": rejected_output,
                    "metadata": {
                        "prompt_id": pid,
                        "chosen_score": sel.get("verification_score", 0),
                        "rejected_score": worst.get("verification_score", 0),
                        "difficulty": sel.get("difficulty", 0.5),
                    },
                })

        path = os.path.join(output_dir, "dpo_pairs.jsonl")
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        print(
            f"[FORMATTER] Exported {len(pairs)} DPO pairs to {path} "
            f"(from {len(selected)} selected, {len(rejected)} rejected)"
        )
        return path

    def _build_target_content(self, trace: Dict[str, Any]) -> str:
        """Build the target string from a trace dict (using current format mode)."""
        thinking = trace.get("thinking_text", "")
        output = trace.get("output_text", "")
        if not output:
            return ""

        if self.config.format_mode == "thinking_and_output":
            ts = self.config.think_start_token
            te = self.config.think_end_token
            return f"{ts}\n{thinking}\n{te}\n{output}" if thinking else output
        elif self.config.format_mode == "output_only":
            return output
        elif self.config.format_mode == "reasoning_augmented":
            strategy = self._extract_strategy(thinking, output)
            ss = self.config.strategy_start_token
            se = self.config.strategy_end_token
            return f"{ss}{strategy}{se}{output}" if strategy else output
        else:
            return output

    def _save_dataset(self, samples: List[Dict[str, Any]], path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"[FORMATTER] Saved {len(samples)} samples to {path}")

    def _compute_metadata(
        self,
        all_samples: List[Dict[str, Any]],
        splits: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        categories = {}
        models = {}
        for s in all_samples:
            meta = s.get("metadata", {})
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            model = meta.get("source_model", "unknown")
            models[model] = models.get(model, 0) + 1

        return {
            "total_samples": len(all_samples),
            "train_samples": len(splits.get("train", [])),
            "validation_samples": len(splits.get("validation", [])),
            "format_mode": self.config.format_mode,
            "categories": categories,
            "source_models": models,
            "config": {
                "format_mode": self.config.format_mode,
                "think_start_token": self.config.think_start_token,
                "think_end_token": self.config.think_end_token,
                "max_thinking_tokens": self.config.max_thinking_tokens,
                "include_system_prompt": self.config.include_system_prompt,
                "train_split_ratio": self.config.train_split_ratio,
            },
            "created_at": datetime.utcnow().isoformat() if _has_datetime() else None,
        }


def _has_datetime() -> bool:
    try:
        from datetime import datetime
        return True
    except ImportError:
        return False


from datetime import datetime
