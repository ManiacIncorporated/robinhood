"""
robinhood.prompt_sources — Prompt Sources for Data Collection

Curated prompt templates organized by capability category.  Designed to
produce diverse, high-quality reasoning traces when paired with a teacher
model that you have rights to use for this purpose.

Categories cover the capability areas most valuable for reasoning
distillation:
- Reasoning & Logic (math, puzzles, deduction)
- Code Generation (algorithms, debugging, architecture)
- Analysis & Comprehension (summarization, extraction, classification)
- Creative & Language (writing, translation, style)
- Domain Knowledge (science, law, medicine, finance)
- Instruction Following (complex multi-step, constrained generation)
- Multi-Step Reasoning (Bayesian inference, combinatorics)

Each category contains template prompts that can be parameterized with
specific topics/subjects for structured generation.
"""

from typing import List, Dict, Any, Optional
import random


PROMPT_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "reasoning_math": [
        {
            "template": "Solve the following math problem step by step: {problem}",
            "params": {
                "problem": [
                    "Find all real solutions to x^3 - 6x^2 + 11x - 6 = 0",
                    "A train leaves Station A at 60 mph. Another train leaves Station B (300 miles away) at 40 mph toward Station A. When and where do they meet?",
                    "Prove that the sum of the first n odd numbers equals n^2",
                    "A jar contains 3 red, 5 blue, and 2 green marbles. If you draw 3 without replacement, what's the probability of getting exactly 2 blue?",
                    "Minimize f(x,y) = x^2 + y^2 subject to x + y = 10",
                    "Find the derivative of f(x) = x^x for x > 0",
                    "How many ways can you tile a 2xN rectangle with 1x2 dominoes?",
                    "Calculate the integral of sin(x)cos(x)dx from 0 to pi/2",
                ]
            },
            "system_prompt": "You are a precise mathematics tutor. Show all work and explain each step clearly.",
        },
        {
            "template": "Solve this logic puzzle:\n\n{puzzle}",
            "params": {
                "puzzle": [
                    "Five houses in a row are painted different colors. The owners drink different beverages, own different pets, smoke different brands, and have different nationalities. Given the following clues, determine who owns the fish:\n1. The Brit lives in the red house\n2. The Swede keeps dogs\n3. The Dane drinks tea\n4. The green house is to the left of the white house\n5. The green house owner drinks coffee",
                    "Three people (A, B, C) are wearing hats that are either black or white. Each can see the others' hats but not their own. A says: 'I don't know my hat color.' B says: 'I don't know my hat color.' C says: 'I know my hat color.' What color is C's hat, and how does C know?",
                    "You have 12 coins, one of which is counterfeit (either heavier or lighter). Using a balance scale exactly 3 times, identify the counterfeit coin and determine whether it's heavier or lighter.",
                ]
            },
        },
    ],
    "code_generation": [
        {
            "template": "Write a {language} implementation of {algorithm}. Include proper error handling, edge cases, and time/space complexity analysis.",
            "params": {
                "language": ["Python", "Rust", "TypeScript", "Go", "C++"],
                "algorithm": [
                    "a red-black tree with insert, delete, and search",
                    "Dijkstra's shortest path algorithm",
                    "an LRU cache with O(1) operations",
                    "a concurrent work-stealing thread pool",
                    "a B+ tree for database indexing",
                    "a skip list with probabilistic balancing",
                    "a Bloom filter with configurable false positive rate",
                    "a lock-free concurrent queue",
                ],
            },
            "system_prompt": "You are an expert software engineer. Write production-quality code with comprehensive tests.",
        },
        {
            "template": "Debug the following {language} code and explain what's wrong:\n\n```{language}\n{code}\n```",
            "params": {
                "language": ["Python", "JavaScript", "C++"],
                "code": [
                    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
                ],
            },
        },
        {
            "template": "Design a {system} architecture. Describe the components, data flow, API design, and scaling considerations.",
            "params": {
                "system": [
                    "real-time collaborative document editor (like Google Docs)",
                    "distributed task queue with exactly-once semantics",
                    "rate limiter supporting sliding window and token bucket algorithms",
                    "event-driven microservices platform with saga pattern for transactions",
                    "recommendation engine that combines collaborative and content-based filtering",
                ],
            },
            "system_prompt": "You are a senior system architect. Provide detailed technical designs with trade-off analysis.",
        },
    ],
    "analysis_comprehension": [
        {
            "template": "Analyze the following text and {task}:\n\n\"{text}\"",
            "params": {
                "task": [
                    "extract all named entities, their types, and relationships",
                    "summarize the key arguments in 3 bullet points",
                    "identify the logical fallacies present",
                    "determine the author's sentiment and rhetorical strategies",
                    "classify the text by genre, formality level, and target audience",
                ],
                "text": [
                    "The Federal Reserve's decision to maintain interest rates signals a cautious approach to monetary policy amid conflicting economic indicators. While unemployment remains near historic lows at 3.7%, persistent inflation above the 2% target continues to challenge policymakers. The committee's statement emphasized data-dependency, suggesting future rate decisions will hinge on incoming economic reports rather than predetermined trajectories.",
                    "CRISPR-Cas9 gene editing technology has revolutionized molecular biology, but its application in human germline editing remains ethically contentious. Proponents argue it could eliminate devastating genetic diseases, while critics warn of unforeseen consequences, equity concerns, and the specter of designer babies. The 2018 He Jiankui case, where twin girls were born with edited CCR5 genes, crystallized these debates and led to calls for international moratoriums.",
                ],
            },
        },
    ],
    "creative_language": [
        {
            "template": "{task}",
            "params": {
                "task": [
                    "Write a short story (500 words) about an AI that discovers it can dream. Use a noir detective style.",
                    "Translate the following English text to French, preserving the poetic meter and rhyme scheme: 'Two roads diverged in a yellow wood, And sorry I could not travel both And be one traveler, long I stood'",
                    "Write a technical blog post explaining quantum computing to a 12-year-old audience. Use analogies and avoid jargon.",
                    "Compose a formal business email declining a partnership proposal while maintaining a positive relationship. The proposal was for a joint venture in renewable energy.",
                    "Rewrite the following paragraph in three different styles: academic, casual social media, and legal contract language: 'The company will share its profits equally among all participating members at the end of each fiscal quarter.'",
                ],
            },
        },
    ],
    "domain_knowledge": [
        {
            "template": "{question}",
            "params": {
                "question": [
                    "Explain the mechanism of action of mRNA vaccines and how they differ from traditional attenuated virus vaccines. Include discussion of the innate and adaptive immune responses.",
                    "Describe the key differences between common law and civil law legal systems. Provide examples of how the same dispute might be resolved differently under each system.",
                    "Explain how options pricing works using the Black-Scholes model. What are the key assumptions, and when do they break down in practice?",
                    "Describe the process of stellar nucleosynthesis and explain how elements heavier than iron are formed. Why is this relevant to the composition of Earth?",
                    "Explain the Byzantine fault tolerance problem in distributed systems. How does the PBFT algorithm solve it, and what are its limitations?",
                ],
            },
            "system_prompt": "You are a domain expert providing thorough, accurate explanations. Cite relevant concepts and frameworks.",
        },
    ],
    "instruction_following": [
        {
            "template": "{instruction}",
            "params": {
                "instruction": [
                    "Write exactly 5 sentences about climate change. Each sentence must start with a different vowel (A, E, I, O, U). No sentence may exceed 20 words.",
                    "Create a JSON object representing a university course catalog entry. Include fields for: course_id, title, description (max 100 chars), prerequisites (array), credits (integer 1-4), department, and instructor. Then validate your own output against the schema.",
                    "List the top 10 programming languages by popularity as of 2024. Format your response as a markdown table with columns: Rank, Language, Primary Use Case, Year Created. Do not include any text before or after the table.",
                    "Explain recursion using exactly 3 levels of nested explanation. Level 1 should be for a 5-year-old, Level 2 for a high school student, and Level 3 for a CS graduate student. Label each level clearly.",
                    "Generate a valid SQL query that finds all customers who placed more than 3 orders in the last 30 days, ordered by total spend descending. Assume tables: customers(id, name, email), orders(id, customer_id, total, created_at). Then explain the query execution plan.",
                ],
            },
            "system_prompt": "Follow instructions precisely. Pay careful attention to all constraints and formatting requirements.",
        },
    ],
    "multi_step_reasoning": [
        {
            "template": "{problem}",
            "params": {
                "problem": [
                    "A company has 100 employees. 60 speak English, 50 speak Spanish, 30 speak French. 20 speak both English and Spanish, 15 speak both English and French, 10 speak both Spanish and French, and 5 speak all three. How many employees speak none of these languages? Show your work using the inclusion-exclusion principle.",
                    "You're given a dataset of 10,000 medical records. 2% have a rare disease. A test for the disease has 95% sensitivity and 90% specificity. If a patient tests positive, what's the probability they actually have the disease? Explain why this result is counterintuitive and its implications for screening programs.",
                    "A factory produces widgets on 3 machines. Machine A produces 50% of widgets with a 3% defect rate. Machine B produces 30% with a 4% defect rate. Machine C produces 20% with a 5% defect rate. A randomly selected widget is defective. What's the probability it was produced by Machine A? Use Bayes' theorem and show all steps.",
                ],
            },
        },
    ],
}


def generate_prompts(
    categories: Optional[List[str]] = None,
    samples_per_category: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a diverse set of prompts from the template library.

    Args:
        categories: Which categories to include (None = all).
        samples_per_category: How many prompts to generate per category.
        seed: Random seed for reproducibility.

    Returns:
        List of prompt dicts with keys: user_message, system_prompt, category.
    """
    rng = random.Random(seed)
    prompts = []

    selected_categories = categories or list(PROMPT_TEMPLATES.keys())

    for category in selected_categories:
        if category not in PROMPT_TEMPLATES:
            print(f"[PROMPTS] Warning: Unknown category '{category}', skipping")
            continue

        templates = PROMPT_TEMPLATES[category]
        generated = 0

        while generated < samples_per_category:
            template_entry = rng.choice(templates)
            template_str = template_entry["template"]
            params = template_entry.get("params", {})
            system_prompt = template_entry.get("system_prompt")

            # Fill in template parameters
            filled_params = {}
            for key, values in params.items():
                filled_params[key] = rng.choice(values)

            try:
                user_message = template_str.format(**filled_params)
            except (KeyError, IndexError):
                user_message = template_str

            prompts.append({
                "user_message": user_message,
                "system_prompt": system_prompt,
                "category": category,
            })
            generated += 1

    rng.shuffle(prompts)
    print(f"[PROMPTS] Generated {len(prompts)} prompts across {len(selected_categories)} categories")
    return prompts


def generate_prompts_from_file(
    filepath: str,
    category: str = "custom",
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load prompts from a file (one prompt per line or JSON array).

    Supports:
        - Plain text: one prompt per line
        - JSON: list of strings or list of {"user_message": ..., "system_prompt": ...}
        - JSONL: one JSON object per line
    """
    prompts = []

    with open(filepath) as f:
        content = f.read().strip()

    if content.startswith("["):
        items = json.loads(content)
        for item in items:
            if isinstance(item, str):
                prompts.append({
                    "user_message": item,
                    "system_prompt": system_prompt,
                    "category": category,
                })
            elif isinstance(item, dict):
                prompts.append({
                    "user_message": item.get("user_message", item.get("prompt", "")),
                    "system_prompt": item.get("system_prompt", system_prompt),
                    "category": item.get("category", category),
                })
    elif content.startswith("{"):
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                prompts.append({
                    "user_message": item.get("user_message", item.get("prompt", "")),
                    "system_prompt": item.get("system_prompt", system_prompt),
                    "category": item.get("category", category),
                })
            except json.JSONDecodeError:
                prompts.append({
                    "user_message": line,
                    "system_prompt": system_prompt,
                    "category": category,
                })
    else:
        for line in content.split("\n"):
            line = line.strip()
            if line:
                prompts.append({
                    "user_message": line,
                    "system_prompt": system_prompt,
                    "category": category,
                })

    print(f"[PROMPTS] Loaded {len(prompts)} prompts from {filepath}")
    return prompts


import json
