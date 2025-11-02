#!/usr/bin/env python3
"""
Prompt Testing Harness: Main orchestrator.

Coordinates scenario generation, prompt execution, output evaluation, and export.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from scenario_generator import ScenarioGenerator
from prompt_executor import PromptExecutor
from output_evaluator import OutputEvaluator


class PromptHarness:
    def __init__(self, nonprofit_ai_path: str):
        """
        Initialize the harness.

        Args:
            nonprofit_ai_path: Path to nonprofit.ai repository
        """
        self.nonprofit_ai_path = Path(nonprofit_ai_path)
        self.prompts_dir = self.nonprofit_ai_path / "src" / "content" / "prompts"
        self.outputs_dir = self.nonprofit_ai_path / "data" / "outputs"

        # Ensure output directory exists
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.scenario_gen = ScenarioGenerator()
        self.executor = PromptExecutor()
        self.evaluator = OutputEvaluator()

    def extract_prompt_and_metadata(self, markdown_path: Path) -> Dict:
        """Extract prompt text and frontmatter from markdown file."""
        with open(markdown_path, 'r') as f:
            content = f.read()

        # Extract frontmatter
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        frontmatter = {}
        if frontmatter_match:
            # Parse YAML-like frontmatter (simplified)
            fm_text = frontmatter_match.group(1)
            for line in fm_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')

        # Extract prompt from first code block
        prompt_match = re.search(r'```\n([\s\S]*?)\n```', content)
        prompt_text = prompt_match.group(1) if prompt_match else None

        return {
            "frontmatter": frontmatter,
            "prompt_text": prompt_text
        }

    def process_prompt(self, prompt_slug: str, num_scenarios: int = 3) -> Dict:
        """
        Process a single prompt through the full harness.

        Args:
            prompt_slug: Prompt filename without .md extension
            num_scenarios: Number of scenarios to generate

        Returns:
            Complete results dict
        """
        markdown_path = self.prompts_dir / f"{prompt_slug}.md"

        if not markdown_path.exists():
            print(f"Error: Prompt not found: {markdown_path}")
            return None

        # Check if output already exists (skip logic)
        output_file = self.outputs_dir / f"{prompt_slug}.json"
        if output_file.exists():
            print(f"\n{'='*60}")
            print(f"Skipping: {prompt_slug} (output already exists)")
            print(f"{'='*60}\n")
            return None

        print(f"\n{'='*60}")
        print(f"Processing: {prompt_slug}")
        print(f"{'='*60}\n")

        # Step 1: Extract prompt and metadata
        print("Step 1: Extracting prompt and metadata...")
        data = self.extract_prompt_and_metadata(markdown_path)
        prompt_text = data['prompt_text']
        category = data['frontmatter'].get('category', 'unknown')

        if not prompt_text:
            print("Error: Could not extract prompt text")
            return None

        print(f"✓ Extracted {len(prompt_text)} character prompt")
        print(f"✓ Category: {category}")

        # Step 2: Generate scenarios
        print(f"\nStep 2: Generating {num_scenarios} scenarios...")
        scenarios = self.scenario_gen.generate_scenarios(
            prompt_text,
            category,
            num_scenarios
        )

        if not scenarios:
            print("Error: Failed to generate scenarios")
            return None

        print(f"✓ Generated {len(scenarios)} scenarios")
        for scenario in scenarios:
            print(f"  - {scenario['name']}: {scenario['description']}")

        # Step 3: Execute prompts for each scenario across all models
        print(f"\nStep 3: Executing prompts ({len(scenarios)} scenarios × 3 models = {len(scenarios) * 3} calls)...")
        all_outputs = []
        total_cost = 0

        for scenario in scenarios:
            print(f"\n  Scenario: {scenario['name']}")
            filled_prompt = self.scenario_gen.fill_prompt(prompt_text, scenario)

            outputs = self.executor.execute_all_models(filled_prompt, scenario['id'])
            all_outputs.extend(outputs)

            for output in outputs:
                if output['success']:
                    total_cost += output.get('cost_usd', 0)

        print(f"\n✓ Total cost: ${total_cost:.4f}")

        # Step 4: Evaluate outputs (in parallel)
        print(f"\nStep 4: Evaluating {len(all_outputs)} outputs in parallel...")

        def evaluate_single_output(output):
            """Helper function for parallel evaluation"""
            if not output['success']:
                return output

            scenario = next((s for s in scenarios if s['id'] == output['scenario_id']), None)
            if not scenario:
                return output

            evaluation = self.evaluator.evaluate_output(
                prompt_text,
                scenario,
                output['content'],
                category
            )
            output['evaluation'] = evaluation
            return output

        # Evaluate all outputs in parallel
        with ThreadPoolExecutor(max_workers=9) as executor:
            future_to_output = {
                executor.submit(evaluate_single_output, output): output
                for output in all_outputs if output['success']
            }

            for future in as_completed(future_to_output):
                output = future_to_output[future]
                result = future.result()
                if result.get('evaluation'):
                    scenario = next((s for s in scenarios if s['id'] == result['scenario_id']), None)
                    print(f"    ✓ {result['model_display_name']} ({scenario['name'] if scenario else '?'}): {result['evaluation'].get('overall_score', 'N/A')}/10")

        # Step 5: Export results
        print(f"\nStep 5: Exporting results...")
        results = {
            "prompt_id": prompt_slug,
            "prompt_text": prompt_text,
            "category": category,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "scenarios": scenarios,
            "outputs": all_outputs,
            "summary": {
                "total_scenarios": len(scenarios),
                "total_outputs": len(all_outputs),
                "successful_outputs": sum(1 for o in all_outputs if o['success']),
                "failed_outputs": sum(1 for o in all_outputs if not o['success']),
                "total_cost_usd": round(total_cost, 4),
                "average_score": round(
                    sum(o.get('evaluation', {}).get('overall_score', 0)
                        for o in all_outputs if o['success']) / max(len([o for o in all_outputs if o['success']]), 1),
                    2
                )
            }
        }

        # Save to JSON
        output_file = self.outputs_dir / f"{prompt_slug}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved to {output_file}")
        print(f"\nSummary:")
        print(f"  Scenarios: {results['summary']['total_scenarios']}")
        print(f"  Outputs: {results['summary']['successful_outputs']}/{results['summary']['total_outputs']}")
        print(f"  Average Score: {results['summary']['average_score']}/10")
        print(f"  Total Cost: ${results['summary']['total_cost_usd']}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Prompt Testing Harness - Generate and validate prompt outputs"
    )
    parser.add_argument(
        "--nonprofit-ai-path",
        default="../nonprofit.ai",
        help="Path to nonprofit.ai repository"
    )
    parser.add_argument(
        "--prompt",
        help="Process a specific prompt by slug"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all prompts"
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=3,
        help="Number of scenarios to generate per prompt (default: 3)"
    )

    args = parser.parse_args()

    harness = PromptHarness(args.nonprofit_ai_path)

    if args.prompt:
        harness.process_prompt(args.prompt, args.scenarios)
    elif args.all:
        # Process all prompts
        prompts = list(harness.prompts_dir.glob("*.md"))
        print(f"Found {len(prompts)} prompts to process")

        for prompt_file in prompts:
            harness.process_prompt(prompt_file.stem, args.scenarios)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
