#!/usr/bin/env python3
"""
Prompt Executor: Runs filled prompts through multiple AI models via OpenRouter.

Takes a prompt with filled parameters and executes it across multiple AI models,
capturing outputs, timestamps, token usage, and costs.
"""

import time
from typing import Dict, List
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Model configuration
MODELS = [
    {
        "key": "gpt5",
        "model_id": "openai/gpt-5",
        "display_name": "ChatGPT (GPT-5)"
    },
    {
        "key": "claude",
        "model_id": "anthropic/claude-sonnet-4.5",
        "display_name": "Claude (Sonnet 4.5)"
    },
    {
        "key": "gemini",
        "model_id": "google/gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash"
    }
]


class PromptExecutor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def execute_prompt(
        self,
        prompt_text: str,
        scenario_id: str,
        model_config: Dict
    ) -> Dict:
        """
        Execute a prompt through a specific model.

        Args:
            prompt_text: The filled prompt (no [BRACKETS])
            scenario_id: ID of the scenario being tested
            model_config: Model configuration dict

        Returns:
            Dict with output, metadata, timing, and cost info
        """
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model_config['model_id'],
                messages=[
                    {"role": "user", "content": prompt_text}
                ]
            )

            end_time = time.time()

            output = response.choices[0].message.content
            usage = response.usage

            return {
                "scenario_id": scenario_id,
                "model": model_config['key'],
                "model_display_name": model_config['display_name'],
                "content": output,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_seconds": round(end_time - start_time, 2),
                "tokens": {
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens
                },
                "cost_usd": self._estimate_cost(model_config['key'], usage),
                "success": True
            }

        except Exception as e:
            return {
                "scenario_id": scenario_id,
                "model": model_config['key'],
                "model_display_name": model_config['display_name'],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": False
            }

    def _estimate_cost(self, model_key: str, usage) -> float:
        """
        Estimate cost based on model and token usage.
        Prices are approximate as of Nov 2025.
        """
        pricing = {
            "gpt5": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
            "claude": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
            "gemini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000}
        }

        prices = pricing.get(model_key, {"input": 0, "output": 0})
        input_cost = usage.prompt_tokens * prices["input"]
        output_cost = usage.completion_tokens * prices["output"]

        return round(input_cost + output_cost, 6)

    def execute_all_models(
        self,
        prompt_text: str,
        scenario_id: str
    ) -> List[Dict]:
        """
        Execute a prompt through all configured models (in parallel).

        Args:
            prompt_text: The filled prompt
            scenario_id: ID of the scenario

        Returns:
            List of outputs from all models
        """
        print(f"  Executing all models in parallel...", flush=True)

        outputs = []

        # Execute all models in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(self.execute_prompt, prompt_text, scenario_id, model_config): model_config
                for model_config in MODELS
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_config = future_to_model[future]
                result = future.result()

                if result['success']:
                    print(f"    ✓ {model_config['display_name']}: {result['tokens']['total']} tokens, ${result['cost_usd']:.4f}")
                else:
                    print(f"    ✗ {model_config['display_name']}: {result.get('error', 'Unknown')}")

                outputs.append(result)

        return outputs


if __name__ == "__main__":
    # Example usage
    executor = PromptExecutor()

    filled_prompt = """Write a warm, appreciative thank-you letter for a first-time donor who gave
$50 to Riverside Food Pantry. The letter should:

- Express genuine gratitude for their support
- Use a warm tone
- Reference this campaign or appeal: Spring Hunger Relief Drive
- Be signed by Maria Garcia, Executive Director"""

    print("Executing prompt across all models...")
    outputs = executor.execute_all_models(filled_prompt, "scenario_1")

    print(f"\nGenerated {len(outputs)} outputs")
    for output in outputs:
        if output['success']:
            print(f"- {output['model_display_name']}: {len(output['content'])} chars")
