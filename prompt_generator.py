#!/usr/bin/env python3
"""
Prompt Generator: Creates elite-level prompts using AI meta-prompts.

Uses GPT-5/Claude with research context and best practices to generate
sophisticated prompts that incorporate domain expertise.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class PromptGenerator:
    def __init__(self, nonprofit_ai_path: str = "../nonprofit.ai"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.nonprofit_ai_path = Path(nonprofit_ai_path)
        self.output_dir = self.nonprofit_ai_path / "src" / "content" / "prompts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load meta-prompt template
        with open("meta_prompt_template.txt", "r") as f:
            self.meta_prompt_template = f.read()

        # Load research context
        if Path("nonprofit_research.md").exists():
            with open("nonprofit_research.md", "r") as f:
                self.research_context = f.read()
        else:
            self.research_context = ""

    def generate_prompt(self, prompt_spec: Dict) -> str:
        """
        Generate an elite-level prompt using AI.

        Args:
            prompt_spec: Dict with topic, category, context, best_practices, etc.

        Returns:
            Generated prompt text
        """
        # Fill in meta-prompt template
        target_user = ', '.join(prompt_spec.get('role', [])) if isinstance(prompt_spec.get('role'), list) else prompt_spec.get('role', 'Nonprofit Professional')

        meta_prompt = self.meta_prompt_template.format(
            TOPIC=prompt_spec['title'],
            CATEGORY=prompt_spec['category'],
            SUBCATEGORY=prompt_spec.get('subcategory', ''),
            TARGET_USER=target_user,
            DIFFICULTY=prompt_spec['difficulty'],
            USER_CONTEXT=prompt_spec.get('user_context', ''),
            BEST_PRACTICES=prompt_spec.get('best_practices', ''),
            PAIN_POINTS=prompt_spec.get('pain_points', '')
        )

        print(f"Generating prompt for: {prompt_spec['title']}")
        print(f"  Using meta-prompt ({len(meta_prompt)} chars)...")

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-5",  # Use GPT-5 for generation
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.7
            )

            generated_prompt = response.choices[0].message.content
            tokens = response.usage.total_tokens
            cost = (response.usage.prompt_tokens * 3 / 1_000_000) + \
                   (response.usage.completion_tokens * 15 / 1_000_000)

            print(f"  ✓ Generated ({tokens} tokens, ${cost:.4f})")

            # Extract the actual prompt from markdown code blocks if present
            # The AI might wrap it in ```
            match = re.search(r'```(?:markdown)?\n([\s\S]*?)\n```', generated_prompt)
            if match:
                generated_prompt = match.group(1)

            return generated_prompt

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def create_markdown_file(self, prompt_spec: Dict, generated_prompt: str):
        """
        Create markdown file with frontmatter and generated prompt.

        Args:
            prompt_spec: Prompt specification dict
            generated_prompt: The AI-generated prompt content
        """
        # Build frontmatter
        frontmatter = f"""---
title: "{prompt_spec['title']}"
description: "{prompt_spec['description']}"
category: "{prompt_spec['category']}"
subcategory: "{prompt_spec.get('subcategory', '')}"
tags: {json.dumps(prompt_spec.get('tags', []))}
role: {json.dumps(prompt_spec.get('role', []))}
difficulty: "{prompt_spec['difficulty']}"
updated: {datetime.now().strftime('%Y-%m-%d')}
---

"""

        # Combine frontmatter + generated content
        full_content = frontmatter + generated_prompt

        # Save to file
        filename = f"{prompt_spec['id']}.md"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(full_content)

        print(f"  ✓ Saved to {filepath}")

        return filepath

    def generate_from_inventory(self, inventory_file: str = "prompt_inventory.json"):
        """
        Generate all prompts from inventory file.

        Args:
            inventory_file: Path to JSON inventory file
        """
        with open(inventory_file, 'r') as f:
            inventory = json.load(f)

        prompts_to_generate = inventory.get('prompts', [])
        print(f"Found {len(prompts_to_generate)} prompts in inventory")

        total_cost = 0
        successful = 0
        failed = 0

        for i, prompt_spec in enumerate(prompts_to_generate, 1):
            print(f"\n[{i}/{len(prompts_to_generate)}] ", end="")

            generated = self.generate_prompt(prompt_spec)

            if generated:
                self.create_markdown_file(prompt_spec, generated)
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Estimated total cost: ${total_cost:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elite prompts using AI")
    parser.add_argument(
        "--nonprofit-ai-path",
        default="../nonprofit.ai",
        help="Path to nonprofit.ai repository"
    )
    parser.add_argument(
        "--inventory",
        default="prompt_inventory.json",
        help="Path to prompt inventory JSON file"
    )

    args = parser.parse_args()

    generator = PromptGenerator(args.nonprofit_ai_path)
    generator.generate_from_inventory(args.inventory)
