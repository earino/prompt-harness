# Prompt Testing Harness

Automated framework for generating, testing, and validating AI prompts with real scenarios and multi-model outputs.

## Overview

This harness generates realistic scenarios for prompts, executes them across multiple AI models (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Flash), evaluates output quality, and exports structured data for display.

## Architecture

1. **Scenario Generator**: Extracts [PARAMETERS] from prompts and uses AI to generate 3 realistic nonprofit scenarios
2. **Prompt Executor**: Fills scenarios into prompts and calls OpenRouter API for multiple models
3. **Output Evaluator**: Uses Claude to score outputs (0-10) on tone, completeness, usefulness, accuracy, authenticity
4. **Main Harness**: Orchestrates the pipeline and exports JSON results

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Usage

```bash
# Process a single prompt
python harness.py --nonprofit-ai-path ../nonprofit.ai --prompt thank-you-first-time-donor

# Process all prompts
python harness.py --nonprofit-ai-path ../nonprofit.ai --all

# Generate more scenarios per prompt
python harness.py --nonprofit-ai-path ../nonprofit.ai --prompt example --scenarios 5
```

## Output Format

Results are saved to `../nonprofit.ai/data/outputs/{prompt_id}.json`:

```json
{
  "prompt_id": "thank-you-first-time-donor",
  "prompt_text": "...",
  "category": "fundraising",
  "generated_at": "2025-11-01T14:30:00Z",
  "scenarios": [
    {
      "id": "scenario_1",
      "name": "Small Community Org",
      "description": "...",
      "values": { "AMOUNT": "$50", ... }
    }
  ],
  "outputs": [
    {
      "scenario_id": "scenario_1",
      "model": "gpt5",
      "model_display_name": "ChatGPT (GPT-5)",
      "content": "Generated output...",
      "timestamp": "2025-11-01T14:30:15Z",
      "tokens": { "input": 450, "output": 320, "total": 770 },
      "cost_usd": 0.006,
      "evaluation": {
        "overall_score": 8.5,
        "criteria_scores": { ... },
        "reasoning": "...",
        "strengths": [...],
        "weaknesses": [...]
      }
    }
  ],
  "summary": {
    "total_cost_usd": 0.054,
    "average_score": 8.3
  }
}
```

## Model Configuration

Models are defined in each component file. To add/remove models:

Edit `prompt_executor.py` MODELS list:
```python
MODELS = [
    {"key": "gpt5", "model_id": "openai/gpt-5", "display_name": "ChatGPT (GPT-5)"},
    # Add more models here
]
```

## How It Works

1. Read prompt markdown from nonprofit.ai repo
2. Extract [PARAMETERS] using regex
3. Use GPT-5 to generate 3 realistic scenarios with filled values
4. For each scenario: Execute prompt through 3 models
5. Use Claude to evaluate each output (score 0-10)
6. Export everything as JSON
7. Nonprofit.ai site reads JSON and displays with dropdown + tabs UI

## Cost Estimates

Per prompt (3 scenarios × 3 models = 9 outputs + evaluations):
- Scenario generation: ~$0.01
- Prompt execution: ~$0.03-0.05
- Evaluation: ~$0.02-0.03
- **Total per prompt: ~$0.06-0.09**
- **80 prompts: ~$5-7 total**

## Reusability

This harness can be used for any prompt domain by:
1. Pointing to different prompt source
2. Adjusting scenario generation prompts
3. Configuring evaluation criteria

The architecture separates content generation (this harness) from content display (nonprofit.ai site).
