#!/usr/bin/env python3
"""
Pilot test: Compare Claude vs Gemini as evaluators.

Tests a subset of outputs to see if dual-evaluation provides signal.
"""

import json
import random
import glob
from pathlib import Path
from output_evaluator import OutputEvaluator
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from statistics import mean, stdev
from scipy.stats import pearsonr

load_dotenv()

# Pick 15 random prompts
json_files = glob.glob('../nonprofit.ai/data/outputs/*.json')
sample_files = random.sample(json_files, min(15, len(json_files)))

print(f"Testing {len(sample_files)} prompts with Gemini evaluator...")
print("=" * 60)

comparisons = []

for json_file in sample_files:
    with open(json_file) as f:
        data = json.load(f)
    
    prompt_id = data['prompt_id']
    prompt_text = data['prompt_text']
    category = data['category']
    
    print(f"\nPrompt: {prompt_id}")
    
    # Pick up to 3 successful outputs from this prompt
    successful_outputs = [o for o in data['outputs'] if o.get('success') and o.get('evaluation')]
    sample_outputs = random.sample(successful_outputs, min(3, len(successful_outputs)))
    
    for output in sample_outputs:
        claude_score = output['evaluation']['overall_score']
        model_name = output['model_display_name']
        
        # Find the scenario
        scenario = next((s for s in data['scenarios'] if s['id'] == output['scenario_id']), None)
        if not scenario:
            continue
        
        # Re-evaluate with Gemini
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        evaluation_prompt = f"""You are an expert evaluator of nonprofit communications and AI-generated content.

Evaluate this AI-generated output for quality and usefulness.

**Original Prompt Template:**
{prompt_text}

**Scenario Used:**
{json.dumps(scenario, indent=2)}

**AI Generated Output:**
{output['content']}

**Category:** {category}

Rate the output on these criteria (0-10 scale):
- **Tone** (1-10): Appropriate for nonprofit audience, warm/professional as needed
- **Completeness** (1-10): Addresses all requirements in the prompt
- **Usefulness** (1-10): Practical value for a nonprofit professional
- **Accuracy** (1-10): Factually sound, no obvious errors
- **Authenticity** (1-10): Feels genuine, not generic or robotic

Respond ONLY with valid JSON in this exact format:
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
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
            
            gemini_eval = json.loads(response_text)
            gemini_score = gemini_eval['overall_score']
            
            diff = gemini_score - claude_score
            comparisons.append({
                'prompt_id': prompt_id,
                'model': output['model'],
                'claude_score': claude_score,
                'gemini_score': gemini_score,
                'difference': diff
            })
            
            print(f"  {model_name}: Claude={claude_score}, Gemini={gemini_score}, Diff={diff:+.1f}")
            
        except Exception as e:
            print(f"  {model_name}: ERROR - {e}")

# Statistical analysis
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

if len(comparisons) >= 10:
    claude_scores = [c['claude_score'] for c in comparisons]
    gemini_scores = [c['gemini_score'] for c in comparisons]
    differences = [c['difference'] for c in comparisons]
    
    correlation, p_value = pearsonr(claude_scores, gemini_scores)
    mean_diff = mean(differences)
    std_diff = stdev(differences) if len(differences) > 1 else 0
    
    print(f"\nSample size: {len(comparisons)} outputs")
    print(f"\nClaude avg: {mean(claude_scores):.2f}")
    print(f"Gemini avg: {mean(gemini_scores):.2f}")
    print(f"\nCorrelation: {correlation:.3f} (p={p_value:.4f})")
    print(f"Mean difference: {mean_diff:+.2f} (Gemini - Claude)")
    print(f"Std deviation: {std_diff:.2f}")
    
    print("\n" + "-" * 60)
    
    if correlation > 0.95:
        print("VERDICT: High agreement (>0.95). Bias is minimal.")
        print("  → Single evaluator (Claude) is probably fine")
    elif correlation > 0.85:
        print("VERDICT: Moderate agreement (0.85-0.95). Some differences.")
        print("  → Dual evaluation could add value")
    else:
        print("VERDICT: Low agreement (<0.85). Significant differences!")
        print("  → Dual evaluation is strongly recommended")
    
    print("\n" + "=" * 60)
    print(f"Estimated cost for full re-evaluation:")
    print(f"  722 outputs × $0.0014 ≈ ${722 * 0.0014:.2f}")
else:
    print("Not enough data for statistical analysis")

