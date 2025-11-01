#!/usr/bin/env python3
"""
Elite Prompt Builder: AI-powered prompt generation with critique and refinement.

Uses GPT-5 to generate prompts, Claude to critique them, and refinement loop
to ensure quality before saving.
"""

import json
import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import argparse

load_dotenv()

# Model Configuration - change these to experiment with different combinations
MODEL_CONFIG = {
    "generator": "openai/gpt-5",  # Who generates the initial prompt
    "critic": "anthropic/claude-sonnet-4.5",  # Who critiques
    "refiner": "openai/gpt-5"  # Who refines based on critique (usually same as generator)
}


class ElitePromptBuilder:
    def __init__(self, nonprofit_ai_path: str = "../nonprofit.ai"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.nonprofit_ai_path = Path(nonprofit_ai_path)
        self.output_dir = self.nonprofit_ai_path / "src" / "content" / "prompts"

        # Load meta-prompts
        with open("generation_meta_prompt.txt", "r") as f:
            self.generation_template = f.read()

        with open("critique_meta_prompt.txt", "r") as f:
            self.critique_template = f.read()

    def _clean_output(self, text: str) -> str:
        """Remove common wrapper patterns from model output."""
        # Try to extract from code blocks first
        match = re.search(r'```(?:\w+)?\n([\s\S]*?)\n```', text)
        if match:
            text = match.group(1)

        # Remove common preambles
        text = re.sub(r'^(?:Here\'s|Here is|Below is).*?[:\n]+', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^(?:The|This) (?:prompt|refined prompt).*?[:\n]+', '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Trim whitespace
        text = text.strip()

        return text

    def generate_prompt(self, spec: dict) -> str:
        """Generate initial prompt using GPT-5."""
        print(f"\n{'='*60}")
        print(f"Generating: {spec['title']}")
        print(f"{'='*60}")

        # Fill generation meta-prompt
        meta_prompt = self.generation_template.format(
            TOPIC=spec['title'],
            CATEGORY=spec['category'],
            SUBCATEGORY=spec.get('subcategory', ''),
            TARGET_USER=', '.join(spec.get('role', [])),
            DIFFICULTY=spec['difficulty'],
            PAIN_POINTS=spec.get('pain_points', ''),
            BEST_PRACTICES=spec.get('best_practices', ''),
            WHY_IT_MATTERS=spec.get('why_it_matters', '')
        )

        print("Step 1: Calling GPT-5 to generate prompt...")

        try:
            response = self.client.chat.completions.create(
                model=MODEL_CONFIG['generator'],
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.7
            )

            draft = response.choices[0].message.content
            cost = (response.usage.prompt_tokens * 3 + response.usage.completion_tokens * 15) / 1_000_000

            print(f"  ✓ Generated ({response.usage.total_tokens} tokens, ${cost:.4f})")

            # Clean up the output
            draft = self._clean_output(draft)

            return draft

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def critique_prompt(self, spec: dict, draft_prompt: str) -> dict:
        """Have Claude critique the draft prompt."""
        print("Step 2: Calling Claude to critique...")

        critique_prompt = self.critique_template.format(
            TOPIC=spec['title'],
            TARGET_USER=', '.join(spec.get('role', [])),
            CATEGORY=spec['category'],
            DRAFT_PROMPT=draft_prompt
        )

        try:
            response = self.client.chat.completions.create(
                model=MODEL_CONFIG['critic'],
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.3
            )

            critique_text = response.choices[0].message.content
            cost = (response.usage.prompt_tokens * 3 + response.usage.completion_tokens * 15) / 1_000_000

            # Parse the critique
            score_match = re.search(r'\*\*Overall Score:\*\*\s*([\d.]+)/10', critique_text)
            score = float(score_match.group(1)) if score_match else 0

            print(f"  ✓ Critique complete (Score: {score}/10, ${cost:.4f})")

            # Extract rewritten version if present
            rewritten_match = re.search(r'\*\*Rewritten Version.*?\*\*.*?\n([\s\S]*?)(?:\*\*|$)', critique_text)
            rewritten = rewritten_match.group(1).strip() if rewritten_match else None

            return {
                'score': score,
                'full_critique': critique_text,
                'rewritten': rewritten
            }

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {'score': 0, 'full_critique': str(e), 'rewritten': None}

    def refine_with_critique(self, draft: str, critique_text: str) -> str:
        """Have GPT-5 refine the prompt based on Claude's critique."""
        print("Step 3: Calling GPT-5 to refine based on critique...")

        refinement_prompt = f"""You previously generated this AI prompt for nonprofit professionals:

{draft}

An expert reviewer provided this critique:

{critique_text}

Please refine the prompt to address the critique. Incorporate the specific improvements suggested while maintaining the prompt's core purpose.

**CRITICAL OUTPUT FORMAT:**
- Output ONLY the refined prompt text itself, nothing else
- Do NOT include any preamble, explanation, or commentary
- Do NOT include phrases like "Here's the refined version:" or "Here is..."
- Do NOT wrap in markdown code blocks (no ```)
- Just output the raw prompt text that a user would copy and paste into ChatGPT/Claude"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_CONFIG['refiner'],
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.5
            )

            refined = response.choices[0].message.content
            cost = (response.usage.prompt_tokens * 3 + response.usage.completion_tokens * 15) / 1_000_000

            # Clean up the output
            refined = self._clean_output(refined)

            print(f"  ✓ Refined ({response.usage.total_tokens} tokens, ${cost:.4f})")
            return refined

        except Exception as e:
            print(f"  ✗ Refinement failed: {e}")
            return draft

    def build_elite_prompt(self, spec: dict) -> dict:
        """Full process: generate, critique, refine."""
        # Generate draft
        draft = self.generate_prompt(spec)
        if not draft:
            return None

        # Critique draft
        critique = self.critique_prompt(spec, draft)

        # Refine if score < 8.0
        if critique['score'] >= 8.0:
            print(f"  → ACCEPTED (score {critique['score']}/10)")
            final_prompt = draft
            final_score = critique['score']
        else:
            print(f"  → REFINING (initial score {critique['score']}/10)")
            refined = self.refine_with_critique(draft, critique['full_critique'])

            # Critique the refined version
            refined_critique = self.critique_prompt(spec, refined)
            final_score = refined_critique['score']

            if refined_critique['score'] > critique['score']:
                print(f"  ✓ IMPROVED (score {critique['score']} → {refined_critique['score']}/10)")
                final_prompt = refined
            else:
                print(f"  ✗ No improvement (keeping original at {critique['score']}/10)")
                final_prompt = draft

        return {
            'final_prompt': final_prompt,
            'initial_score': critique['score'],
            'final_score': final_score,
            'critique': critique['full_critique']
        }

    def save_prompt(self, spec: dict, result: dict):
        """Save elite prompt to markdown file."""
        # Build markdown
        frontmatter = f"""---
title: "{spec['title']}"
description: "{spec['description']}"
category: "{spec['category']}"
subcategory: "{spec.get('subcategory', '')}"
tags: {json.dumps(spec.get('tags', []))}
role: {json.dumps(spec.get('role', []))}
difficulty: "{spec['difficulty']}"
updated: {datetime.now().strftime('%Y-%m-%d')}
---

## The Prompt

```
{result['final_prompt']}
```

## How to Customize

1. Replace all [BRACKETED] fields with your specific information
2. Adjust tone and length as needed for your audience
3. Review and personalize before using

## Pro Tips

1. Test this prompt with your preferred AI tool before using in production
2. Always review AI output for accuracy and appropriateness
3. Customize outputs to match your organization's voice and brand

## Related Prompts

(See other prompts in the {spec['category']} category)
"""

        filepath = self.output_dir / f"{spec['id']}.md"
        with open(filepath, 'w') as f:
            f.write(frontmatter)

        print(f"  ✓ Saved to {filepath.name}")

        return filepath

    def process_inventory(self, inventory_path: str, prompt_ids: list = None):
        """Process prompts from inventory."""
        with open(inventory_path, 'r') as f:
            inventory = json.load(f)

        prompts = inventory.get('prompts', [])

        # Filter if specific IDs provided
        if prompt_ids:
            prompts = [p for p in prompts if p['id'] in prompt_ids]

        print(f"Processing {len(prompts)} prompts...")

        results = []
        total_cost = 0
        skipped = 0

        for i, spec in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")

            # Check if prompt already exists (checkpoint/resume capability)
            filepath = self.output_dir / f"{spec['id']}.md"
            if filepath.exists():
                print(f"  ⏭️  SKIPPING (already exists): {spec['title']}")
                skipped += 1
                continue

            result = self.build_elite_prompt(spec)
            if result:
                self.save_prompt(spec, result)
                results.append({
                    'id': spec['id'],
                    'title': spec['title'],
                    'score': result['initial_score']
                })

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total in inventory: {len(prompts)}")
        print(f"Skipped (already exist): {skipped}")
        print(f"Processed: {len(results)}")
        if results:
            print(f"Average Score: {sum(r['score'] for r in results) / len(results):.2f}/10")

            # Flag low scorers
            low_scorers = [r for r in results if r['score'] < 7.0]
            if low_scorers:
                print(f"\n⚠️  Low-scoring prompts (review needed):")
                for r in low_scorers:
                    print(f"  - {r['title']}: {r['score']}/10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build elite prompts with AI generation and critique")
    parser.add_argument("--inventory", default="prompt_inventory_weak.json", help="Inventory file")
    parser.add_argument("--prompts", nargs='+', help="Specific prompt IDs to process")
    parser.add_argument("--nonprofit-ai-path", default="../nonprofit.ai", help="Path to nonprofit.ai repo")

    args = parser.parse_args()

    builder = ElitePromptBuilder(args.nonprofit_ai_path)
    builder.process_inventory(args.inventory, args.prompts)
