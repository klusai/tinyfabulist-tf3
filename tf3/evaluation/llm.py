import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

MODEL = "google/gemini-2.5-flash"


def evaluate_line_with_llm(
    text: str,
    model: str = MODEL,
) -> Dict:
    """Evaluate a single line of Romanian text for fluency, coherence, and mistakes."""

    prompt = f"""Evaluate the following Romanian text line for fluency, coherence, and grammatical mistakes.

Text:
{text}

Analyze the text and:
1. **Fluency** (0-100): How natural and fluent the text sounds in Romanian. Consider:
   - Natural word order and phrasing
   - Appropriate use of Romanian expressions
   - Smooth flow and readability
   - 100 = perfectly fluent, natural Romanian
   - 80-99 = very fluent with minor awkwardness
   - 60-79 = mostly fluent but some awkward phrasing
   - 40-59 = somewhat fluent but noticeable issues
   - 0-39 = not fluent, very awkward

2. **Coherence** (0-100): How well the text makes sense and maintains logical flow. Consider:
   - Logical structure and meaning
   - Clear connections between ideas
   - Consistency in narrative/argument
   - 100 = perfectly coherent and clear
   - 80-99 = very coherent with minor issues
   - 60-79 = mostly coherent but some confusion
   - 40-59 = somewhat coherent but unclear parts
   - 0-39 = incoherent, confusing

3. **Mistakes**: Count and identify all grammatical mistakes (agreement, conjugation, declension, word choice, etc.)

Respond in JSON format with the following structure:
{{
  "fluency": <score 0-100>,
  "coherence": <score 0-100>,
  "total_mistakes": <exact count of mistakes>,
  "mistakes": [
    {{
      "position": "<position in text>",
      "type": "<type of mistake>",
      "original": "<incorrect text>",
      "correction": "<suggested correction>",
      "explanation": "<short explanation>"
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Romanian language. You evaluate texts for fluency, coherence, and grammatical accuracy with precision. Always respond in valid JSON format only, no additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Lower temperature for more consistent results
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
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
            "fluency": 0,
            "coherence": 0,
            "total_mistakes": 0,
            "mistakes": [],
        }


def evaluate_file(
    file_path: str,
    model: str = MODEL,
    batch_size: Optional[int] = None,
    max_workers: int = 10,
) -> Dict:
    print(f"Evaluating file: {file_path} with model: {model}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return {
            "fluency": 0.0,
            "coherence": 0.0,
            "total_mistakes": 0,
            "average_score": 0.0,
            "summary": "File is empty",
        }

    # Use the detailed evaluation function
    result = llm_grammar_score_detailed(lines, model, max_workers=max_workers)
    result["summary"] = f"Evaluated {len(lines)} lines"
    return result


def llm_grammar_score_detailed(texts: List[str], model: str = MODEL, max_workers: int = 10) -> Dict:
    """
    Compute detailed grammar scores for a list of Romanian texts using LLM.
    Evaluates each line for fluency and coherence in parallel, then averages the scores.
    Returns a dict with fluency, coherence, total_mistakes, and average_score.
    
    Args:
        texts: List of text strings to evaluate
        model: Model name to use for evaluation
        max_workers: Maximum number of parallel workers (default: 10)
    """
    if not texts:
        return {
            "fluency": 0.0,
            "coherence": 0.0,
            "total_mistakes": 0,
            "average_score": 0.0,
        }
    
    # Filter out empty texts
    non_empty_texts = [text.strip() for text in texts if text.strip()]
    if not non_empty_texts:
        return {
            "fluency": 0.0,
            "coherence": 0.0,
            "total_mistakes": 0,
            "average_score": 0.0,
        }
    
    total_lines = len(non_empty_texts)
    print(f"Evaluating {total_lines} lines for fluency, coherence, and mistakes (parallel, {max_workers} workers)...")
    
    # Thread-safe progress tracking
    progress_lock = Lock()
    completed = 0
    
    # Store results in order
    results_list = [None] * total_lines
    
    def evaluate_with_progress(idx: int, text: str):
        """Evaluate a single line and update progress."""
        nonlocal completed
        result = evaluate_line_with_llm(text, model)
        
        with progress_lock:
            completed += 1
            results_list[idx] = result
            
            if "error" not in result:
                fluency = result.get("fluency", 0)
                coherence = result.get("coherence", 0)
                mistakes = result.get("total_mistakes", 0)
                print(f"  [{completed}/{total_lines}] ✓ Line {idx+1}: Fluency: {fluency:.1f}, Coherence: {coherence:.1f}, Mistakes: {mistakes}")
            else:
                print(f"  [{completed}/{total_lines}] ✗ Line {idx+1}: Error: {result.get('error', 'Unknown error')}")
        
        return result
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(evaluate_with_progress, idx, text): idx 
                   for idx, text in enumerate(non_empty_texts)}
        
        # Wait for all to complete (results are already printed in the function)
        for future in as_completed(futures):
            future.result()  # Wait for completion and handle any exceptions
    
    print(f"\nCompleted evaluation of {total_lines} lines.")
    
    # Aggregate results
    fluency_scores = []
    coherence_scores = []
    total_mistakes = 0
    
    for result in results_list:
        if result and "error" not in result:
            fluency = result.get("fluency", 0)
            coherence = result.get("coherence", 0)
            mistakes = result.get("total_mistakes", 0)
            
            fluency_scores.append(float(fluency))
            coherence_scores.append(float(coherence))
            total_mistakes += int(mistakes)
        else:
            # If error, assign 0 scores
            fluency_scores.append(0.0)
            coherence_scores.append(0.0)
    
    # Calculate averages
    avg_fluency = sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.0
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    # Return average of fluency and coherence
    final_score = (avg_fluency + avg_coherence) / 2.0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Average Fluency: {avg_fluency:.2f}/100")
    print(f"  Average Coherence: {avg_coherence:.2f}/100")
    print(f"  Average Score: {final_score:.2f}/100")
    print(f"  Total Mistakes: {total_mistakes}")
    print(f"{'='*60}\n")
    
    return {
        "fluency": float(avg_fluency),
        "coherence": float(avg_coherence),
        "total_mistakes": total_mistakes,
        "average_score": float(final_score),
    }


def llm_grammar_score(texts: List[str], model: str = MODEL) -> float:
    """
    Compute grammar score for a list of Romanian texts using LLM.
    Evaluates each line for fluency and coherence, then averages the scores.
    Returns a score from 0-100 (higher is better) - average of fluency and coherence.
    """
    result = llm_grammar_score_detailed(texts, model)
    return result["average_score"]


def print_results(results: Dict, verbose: bool = False):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print("GRAMMATICAL EVALUATION RESULTS")
    print("=" * 80)

    if "error" in results:
        print(f"\n❌ ERROR: {results['error']}")
        return

    # Handle both old and new format
    if "fluency" in results and "coherence" in results:
        # New format with fluency and coherence
        fluency = results.get("fluency", 0.0)
        coherence = results.get("coherence", 0.0)
        avg_score = results.get("average_score", (fluency + coherence) / 2.0)
        total_mistakes = results.get("total_mistakes", 0)
        summary = results.get("summary", "")
        
        print(f"\n📊 Average Score: {avg_score:.2f}/100")
        print(f"   📈 Fluency: {fluency:.2f}/100")
        print(f"   📈 Coherence: {coherence:.2f}/100")
        print(f"   🔍 Total Mistakes: {total_mistakes}")
    else:
        # Old format (backward compatibility)
        average_score = results.get("average_score", 0)
        total_mistakes = results.get("total_mistakes", 0)
        mistakes = results.get("mistakes", [])
        summary = results.get("summary", "")

        print(f"\n📊 Average Score: {average_score:.2f}/100")

        # Print mistakes
        print(f"\n🔍 Total Mistakes: {total_mistakes}")

        if mistakes and verbose:
            print("\n" + "-" * 80)
            print("Mistakes Details:")
            print("-" * 80)
            for i, mistake in enumerate(mistakes, 1):
                print(f"\n{i}. Position: {mistake.get('position', 'N/A')}")
                print(f"   Type: {mistake.get('type', 'N/A')}")
                print(f"   Original: {mistake.get('original', 'N/A')}")
                print(f"   Correction: {mistake.get('correction', 'N/A')}")
                if mistake.get("explanation"):
                    print(f"   Explanation: {mistake.get('explanation')}")

    # Print summary
    if summary:
        print(f"\n📝 Summary: {summary}")

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
        "--model",
        type=str,
        default=MODEL,
        help="Model to use (default: " + MODEL + ")",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process file in batches of N lines (optional)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=25,
        help="Maximum number of parallel workers for evaluation (default: 25)",
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

    # Check API key is set
    if not API_KEY:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
        )

    # Evaluate file
    results = evaluate_file(args.file, args.model, args.batch_size, max_workers=args.max_workers)

    # Print results
    print_results(results, args.verbose)

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

    # Return score for programmatic use
    return results.get("score", 0)


if __name__ == "__main__":
    main()
