#!/usr/bin/env python3
"""
Scenario Generator: Extracts parameters from prompts and generates realistic scenarios.

Analyzes prompt templates to identify [BRACKETED] parameters, then uses AI to
generate 3 realistic nonprofit scenarios with concrete values for each parameter.
"""

import re
from typing import List, Dict, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

class ScenarioGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def extract_parameters(self, prompt_text: str) -> List[str]:
        """
        Extract all [BRACKETED] parameters from prompt text.
        Returns list of unique parameters in order of appearance.
        """
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, prompt_text)
        # Preserve order but remove duplicates
        seen = set()
        unique_params = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_params.append(match)
        return unique_params

    def generate_scenarios(
        self,
        prompt_text: str,
        prompt_category: str,
        num_scenarios: int = 3
    ) -> List[Dict]:
        """
        Generate realistic nonprofit scenarios for a given prompt.

        Args:
            prompt_text: The template prompt with [PARAMETERS]
            prompt_category: Category (fundraising, communications, etc.)
            num_scenarios: Number of scenarios to generate (default: 3)

        Returns:
            List of scenarios, each with filled parameter values
        """
        parameters = self.extract_parameters(prompt_text)

        if not parameters:
            return []

        # Build AI prompt to generate scenarios
        generation_prompt = f"""You are helping create realistic nonprofit scenarios for testing AI prompts.

Given this prompt template:
\"\"\"
{prompt_text}
\"\"\"

Category: {prompt_category}

The prompt contains these parameters that need values: {', '.join(parameters)}

Generate {num_scenarios} realistic, diverse nonprofit scenarios. Each scenario should represent different:
- Organization sizes (small community org, mid-size professional org, large established org)
- Contexts (urban/rural, different causes, different donor segments)
- Realistic names and details that feel authentic to the nonprofit sector

For each scenario, provide concrete values for ALL parameters. Make them feel real and varied.

Respond ONLY with valid JSON in this exact format:
{{
  "scenarios": [
    {{
      "id": "scenario_1",
      "name": "Small Community Org",
      "description": "Brief description of org type/context",
      "values": {{
        "PARAMETER1": "concrete value",
        "PARAMETER2": "concrete value"
      }}
    }}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-5",
                messages=[
                    {"role": "user", "content": generation_prompt}
                ],
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            # Extract JSON from response (might be wrapped in markdown)
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)

            result = json.loads(response_text)
            return result.get('scenarios', [])

        except Exception as e:
            print(f"Error generating scenarios: {e}")
            return []

    def fill_prompt(self, prompt_text: str, scenario: Dict) -> str:
        """
        Fill a prompt template with scenario values.

        Args:
            prompt_text: Template with [PARAMETERS]
            scenario: Dict with parameter values

        Returns:
            Filled prompt text
        """
        filled = prompt_text
        values = scenario.get('values', {})

        for param, value in values.items():
            # Replace [PARAM] with value
            filled = filled.replace(f'[{param}]', value)

        return filled


if __name__ == "__main__":
    # Example usage
    generator = ScenarioGenerator()

    example_prompt = """Write a warm, appreciative thank-you letter for a first-time donor who gave
$[AMOUNT] to [ORGANIZATION NAME]. The letter should:

- Express genuine gratitude for their support
- Use a [FORMAL/WARM/CASUAL] tone
- Reference this campaign or appeal: [CAMPAIGN NAME]
- Be signed by [SIGNER NAME and TITLE]"""

    print("Extracting parameters...")
    params = generator.extract_parameters(example_prompt)
    print(f"Found parameters: {params}")

    print("\nGenerating scenarios...")
    scenarios = generator.generate_scenarios(example_prompt, "fundraising")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Values: {json.dumps(scenario['values'], indent=2)}")
