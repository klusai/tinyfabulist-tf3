import argparse
from typing import List

import language_tool_python as lt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="crevetele mărunt la soare. Într-o zi, un pește mare, cu dinți ascuțiți, a ajuns în mlaștină.",
    )
    return parser.parse_args()


def language_tool_score(texts: List[str]) -> float:
    """
    Compute grammar score using LanguageTool for Romanian.
    Returns a score from 0-1 (higher is better).
    Score = 1 - (mistakes / total_words)
    """
    tool = lt.LanguageTool("ro-RO")  # Romanian

    total_mistakes = 0
    total_words = 0

    for text in texts:
        if not text.strip():
            continue

        # Count words (simple approximation)
        words = text.split()
        total_words += len(words)

        # Check for mistakes
        matches = tool.check(text)
        total_mistakes += len(matches)

    if total_words == 0:
        return 1.0  # No text = perfect score

    # Score: 1 - (mistakes / words), clamped to [0, 1]
    mistake_rate = total_mistakes / total_words
    score = max(0.0, 1.0 - mistake_rate)

    return score


if __name__ == "__main__":
    args = parse_args()
    score = language_tool_score([args.text])
    print(f"LanguageTool Grammar Score: {score:.4f}")
