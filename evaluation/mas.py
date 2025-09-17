"""
Morphological Agreement Score
"""

import argparse

import stanza
stanza.download("ro")
nlp = stanza.Pipeline("ro")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    text = args.text

    doc = nlp(text)

    correct = 0
    total = len(doc.sentences)

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == "VERB" and word.deprel == "ROOT":
                correct += 1
            total += 1

    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"MAS: {correct / total}")

if __name__ == "__main__":
    main()