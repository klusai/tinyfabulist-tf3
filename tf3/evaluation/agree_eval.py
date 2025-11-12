import argparse
import os
from typing import List

import stanza
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Stanza (Romanian)
def build_nlp():
    stanza.download("ro", processors="tokenize,pos,lemma,depparse", verbose=False)
    return stanza.Pipeline(
        "ro",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=False,
        verbose=False,
    )


def agree_feats(feats1, feats2, keys, word1=None, word2=None):
    """Return True if all keys match in both feats dicts"""
    for k in keys:
        if (
            feats1.get(k) != feats2.get(k)
            and feats1.get(k) != None
            and feats2.get(k) != None
        ):
            return False, f"{k}, {feats1.get(k)} != {feats2.get(k)}"
    return True, None


def parse_feats(feats_str):
    return dict(f.split("=") for f in feats_str.split("|")) if feats_str else {}


def check_agreement_feats(word, head, feats):
    wf = parse_feats(word.feats)
    hf = parse_feats(head.feats)
    ok, msg = agree_feats(wf, hf, feats, word.text, head.text)
    if not ok:
        return f"{word.text} and {head.text} do not agree on {msg}"
    return None


def check_agreement(sentence):
    errors = []

    for word in sentence.words:
        head = sentence.words[word.head - 1] if word.head > 0 else None
        if not head:
            continue

        # adjective–noun
        if word.deprel == "amod" and head.upos == "NOUN":
            msg = check_agreement_feats(word, head, ["Gender", "Number"])
            if msg:
                errors.append((word.text, head.text, "Adj–Noun", msg))

        # determiner–noun
        if word.deprel == "det" and head.upos == "NOUN":
            msg = check_agreement_feats(word, head, ["Gender", "Number"])
            if msg:
                errors.append((word.text, head.text, "Det–Noun", msg))

        # subject–verb
        if word.deprel == "nsubj" and head.upos == "VERB":
            # finite verb
            if parse_feats(head.feats).get("VerbForm") == "Fin":
                msg = check_agreement_feats(word, head, ["Number"])
                if msg:
                    errors.append((word.text, head.text, "Subj–Verb", msg))
            # participle verb: check auxiliary instead
            elif parse_feats(head.feats).get("VerbForm") == "Part":
                for aux in [
                    w for w in sentence.words if w.head == head.id and w.deprel == "aux"
                ]:
                    af = parse_feats(aux.feats)
                    msg = check_agreement_feats(word, aux, ["Number"])
                    if msg:
                        errors.append((word.text, aux.text, "Subj–Aux", msg))

        if word.deprel == "acl" and head.upos == "NOUN":
            if parse_feats(word.feats).get("VerbForm") == "Part":
                msg = check_agreement_feats(word, head, ["Gender", "Number"])
                if msg:
                    errors.append((word.text, head.text, "Participle–Noun", msg))

    return errors


def compute_agree_stats(texts: List[str], file: str = None):
    nlp = build_nlp()

    dataset_stats = {"total_sentences": 0, "total_mistakes": 0, "agree": 0.0}

    if texts:
        for text in texts:
            # remove first and last sentence
            text = text.split(".")[1:-1]
            text = "\n".join(text)

            doc = nlp(text)
            for sent in doc.sentences:
                dataset_stats["total_sentences"] += 1

                mistakes = len(check_agreement(sent))
                dataset_stats["total_mistakes"] += mistakes
    elif file:
        with open(file, "r", encoding="utf-8") as f:
            for text in f:
                # remove first and last sentence
                text = text.split(".")[1:-1]
                text = "\n".join(text)

                doc = nlp(text)

                for sent in doc.sentences:
                    dataset_stats["total_sentences"] += 1
                    mistakes = len(check_agreement(sent))
                    dataset_stats["total_mistakes"] += mistakes

    dataset_stats["agree"] = (
        dataset_stats["total_mistakes"] / dataset_stats["total_sentences"]
    )

    return dataset_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=str, help="Single input text")
    parser.add_argument(
        "--file", type=str, help="Path to a file with one text per line (UTF-8)"
    )
    args = parser.parse_args()

    compute_agree_stats(args.texts, args.file)
