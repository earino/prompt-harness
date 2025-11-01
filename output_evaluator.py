#!/usr/bin/env python3
"""
Output Evaluator: Uses AI to score the quality of generated outputs.

Evaluates AI-generated outputs for tone, completeness, usefulness, and accuracy.
Provides 0-10 score and reasoning for each output.
"""

from typing import Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()


class OutputEvaluator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def evaluate_output(
        self,
        original_prompt: str,
        scenario_context: Dict,
        model_output: str,
        category: str
    ) -> Dict:
        """
        Evaluate the quality of an AI-generated output.

        Args:
            original_prompt: The original template prompt
            scenario_context: The scenario that was used (with values)
            model_output: The AI's generated response
            category: Prompt category (fundraising, communications, etc.)

        Returns:
            Dict with score (0-10), reasoning, and evaluation criteria
        """
        evaluation_prompt = f"""You are an expert evaluator of nonprofit communications and AI-generated content.

Evaluate this AI-generated output for quality and usefulness.

**Original Prompt Template:**
{original_prompt}

**Scenario Used:**
{json.dumps(scenario_context, indent=2)}

**AI Generated Output:**
{model_output}

**Category:** {category}

Evaluate the output on these criteria (each 0-10):
1. **Tone Appropriateness**: Does it match the requested tone and nonprofit sector norms?
2. **Completeness**: Does it fulfill all requirements from the prompt?
3. **Usefulness**: Would a nonprofit professional actually use this?
4. **Accuracy**: Are there any errors, inappropriate language, or red flags?
5. **Authenticity**: Does it sound genuine or robotic/generic?

Provide:
- Overall score (0-10, average of criteria)
- Brief reasoning (2-3 sentences)
- Specific strengths
- Specific weaknesses (if any)

Respond ONLY with valid JSON:
{{
  "overall_score": 8.5,
  "criteria_scores": {{
    "tone": 9,
    "completeness": 8,
    "usefulness": 9,
    "accuracy": 8,
    "authenticity": 8
  }},
  "reasoning": "Brief explanation",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1"] or []
}}"""

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-sonnet-4.5",  # Use Claude for evaluation
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3  # Lower temp for consistent evaluation
            )

            response_text = response.choices[0].message.content
            # Extract JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)

            evaluation = json.loads(response_text)
            evaluation['evaluator_model'] = 'claude-sonnet-4.5'
            return evaluation

        except Exception as e:
            print(f"Error evaluating output: {e}")
            return {
                "overall_score": 0,
                "error": str(e),
                "evaluator_model": "claude-sonnet-4.5"
            }


if __name__ == "__main__":
    # Example usage
    evaluator = OutputEvaluator()

    example_prompt = "Write a thank-you letter for a $50 first-time donor..."
    example_scenario = {
        "name": "Small Community Org",
        "values": {
            "AMOUNT": "$50",
            "ORGANIZATION NAME": "Riverside Food Pantry"
        }
    }
    example_output = "Dear Friend,\n\nThank you for your $50 gift to Riverside Food Pantry..."

    print("Evaluating output...")
    score = evaluator.evaluate_output(
        example_prompt,
        example_scenario,
        example_output,
        "fundraising"
    )

    print(f"\nScore: {score.get('overall_score', 'N/A')}/10")
    print(f"Reasoning: {score.get('reasoning', 'N/A')}")
