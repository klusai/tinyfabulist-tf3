import argparse
import json
import os
import re
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")


def evaluate_grammar_with_llm(
    text: str,
    api_key: str,
    model: str = "openai/gpt-5-mini",
) -> Dict:
    """Evaluate Romanian text for grammatical mistakes using OpenRouter LLM."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Evaluate the following Romanian text for grammatical mistakes.

Text:
{text}

Analyze the text and:
1. Identify all grammatical mistakes (agreement, conjugation, declension, etc.)
2. For each mistake, indicate:
   - Position in text (word/phrase)
   - Type of mistake
   - Suggested correction
3. Calculate a score from 0-100 for grammatical quality:
   - 100 = perfect, no mistakes
   - 90-99 = very good, minor mistakes
   - 70-89 = good, few mistakes
   - 50-69 = acceptable, many mistakes
   - 0-49 = poor, very many mistakes

Respond in JSON format with the following structure:
{{
  "score": <score 0-100>,
  "total_mistakes": <total number of mistakes>,
  "mistakes": [
    {{
      "position": "<position in text>",
      "type": "<type of mistake>",
      "original": "<incorrect text>",
      "correction": "<suggested correction>",
      "explanation": "<short explanation>"
    }}
  ],
  "summary": "<general summary of grammatical quality>"
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Romanian grammar. You analyze texts and identify grammatical mistakes with precision. Always respond in valid JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Lower temperature for more consistent results
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
        # Look for JSON in the response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # Fallback: try to parse the whole response as JSON
            result = json.loads(content)
            return result

    except Exception as e:
        return {
            "error": str(e),
            "score": 0,
            "total_mistakes": 0,
            "mistakes": [],
            "summary": f"Error during evaluation: {e}",
        }


def evaluate_file(
    file_path: str,
    api_key: str,
    model: str = "anthropic/claude-3.5-sonnet",
    batch_size: Optional[int] = None,
) -> Dict:
    """Evaluate a file containing Romanian text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Combine all lines into text
    full_text = "".join(lines).strip()

    if not full_text:
        return {
            "score": 0,
            "total_mistakes": 0,
            "mistakes": [],
            "summary": "File is empty",
        }

    # If batch_size is specified, process in batches
    if batch_size and len(lines) > batch_size:
        all_mistakes = []
        total_score = 0
        num_batches = 0

        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i : i + batch_size]
            batch_text = "".join(batch_lines).strip()

            if batch_text:
                result = evaluate_grammar_with_llm(batch_text, api_key, model)
                if "error" not in result:
                    all_mistakes.extend(result.get("mistakes", []))
                    total_score += result.get("score", 0)
                    num_batches += 1

        avg_score = total_score / num_batches if num_batches > 0 else 0

        return {
            "score": round(avg_score, 2),
            "total_mistakes": len(all_mistakes),
            "mistakes": all_mistakes,
            "summary": f"Evaluated in {num_batches} batches. Average score: {avg_score:.2f}",
        }
    else:
        # Process entire file at once
        return evaluate_grammar_with_llm(full_text, api_key, model)


def llm_grammar_score(texts: List[str], api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet") -> float:
    """
    Compute grammar score for a list of Romanian texts using LLM.
    Returns a score from 0-100 (higher is better).
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")
    
    # Combine all texts
    full_text = "\n".join(texts).strip()
    
    if not full_text:
        return 0.0
    
    # Evaluate with LLM
    result = evaluate_grammar_with_llm(full_text, api_key, model)
    
    if "error" in result:
        return 0.0
    
    # Return score (0-100)
    score = result.get("score", 0)
    return float(score)


def print_results(results: Dict, verbose: bool = False):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print("REZULTATE EVALUARE GRAMATICALĂ")
    print("=" * 80)

    if "error" in results:
        print(f"\n❌ EROARE: {results['error']}")
        return

    score = results.get("score", 0)
    total_mistakes = results.get("total_mistakes", 0)
    mistakes = results.get("mistakes", [])
    summary = results.get("summary", "")

    # Print score
    print(f"\n📊 SCOR GRAMATICAL: {score:.2f}/100")
    if score >= 90:
        print("   ✅ Calitate excelentă")
    elif score >= 70:
        print("   ✅ Calitate bună")
    elif score >= 50:
        print("   ⚠️  Calitate acceptabilă")
    else:
        print("   ❌ Calitate slabă")

    # Print summary
    if summary:
        print(f"\n📝 REZUMAT: {summary}")

    # Print mistakes
    print(f"\n🔍 TOTAL GREȘELI IDENTIFICATE: {total_mistakes}")

    if mistakes and verbose:
        print("\n" + "-" * 80)
        print("DETALII GREȘELI:")
        print("-" * 80)
        for i, mistake in enumerate(mistakes, 1):
            print(f"\n{i}. Poziție: {mistake.get('position', 'N/A')}")
            print(f"   Tip: {mistake.get('type', 'N/A')}")
            print(f"   Original: {mistake.get('original', 'N/A')}")
            print(f"   Corecție: {mistake.get('correction', 'N/A')}")
            if mistake.get("explanation"):
                print(f"   Explicație: {mistake.get('explanation')}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Romanian text for grammatical mistakes using LLM via OpenRouter"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the Romanian text file to evaluate",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="Model to use (default: anthropic/claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process file in batches of N lines (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed mistake information",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var or use --api-key"
        )

    # Evaluate file
    results = evaluate_file(args.file, api_key, args.model, args.batch_size)

    # Print results
    print_results(results, args.verbose)

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Rezultatele au fost salvate în: {args.output}")

    # Return score for programmatic use
    return results.get("score", 0)


if __name__ == "__main__":
    main()
