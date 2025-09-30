import argparse
from collections import Counter
import math
from typing import Dict, List

import stanza
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-ner-model/checkpoint-3192")
    parser.add_argument(
        "--text",
        type=str,
        default="koala somnoroasă  îi plăcea să exploreze și să se joace cu prietenii ei. Dar într-o zi, în timp ce se încălzea la soare, a întâlnit un iepure blând care i-a spus: „Bună, micuțo! De ce ești atât de tristă?” Koala a răspuns: „Mi-aș dori să pot fi la fel de fericită ca iepurele. Aș vrea să pot să alerg ca ea.” Iepurele a zâmbit și a spus: „Dar fiecare este bun la ceva. Ești bună la ceva?” Koala s-a gândit o clipă și a spus: „Da, mi-ar plăcea!” Iepurele a zâmbit și a spus: „Atunci de ce vrei să fii ca mine?” Koala a zâmbit înapoi, simțindu-se fericită și încrezătoare. În timp ce se jucau împreună, a apărut o vulpe șireată. Vulpea a spus: „Hei, prieteni! Vă voi ajuta să găsiți cele mai gustoase fructe de pădure din pădure. Dar vă rog să mă credeți!” Iepurele și koala ",
    )
    return parser.parse_args()


def group_entities_based_on_base_form(entities: List[Dict]):
    nlp = stanza.Pipeline(
        "ro",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=False,
        verbose=False,
    )

    entity_base_forms = {}

    for entity in entities:
        entity_base_form = nlp(entity["word"])[0].lemma
        if entity_base_form not in entity_base_forms:
            entity_base_forms[entity_base_form] = []
        entity_base_forms[entity_base_form].append(entity)

    return entity_base_forms


def entity_coherence_score(texts, model):
    """
    High score when entity mentions are balanced (uniform).
    Low score when dominated by a single entity.
    """

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForTokenClassification.from_pretrained(model)

    ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="first"
    )

    nlp = stanza.Pipeline(
        "ro", processors="tokenize,pos,lemma", verbose=False
    )

    def lemmatize_token(token: str) -> str:
        doc = nlp(token)
        for sent in doc.sentences:
            for w in sent.words:
                return w.lemma.lower()
        return token.lower()

    lemma_counts = Counter()

    for text in texts:
        entities = ner_pipeline(text)
        for ent in entities:
            lemma = lemmatize_token(ent["word"])
            lemma_counts[lemma] += 1

    total = sum(lemma_counts.values())
    if total == 0:
        return 1.0  # no entities = trivially "balanced"

    # Shannon entropy
    probs = [c / total for c in lemma_counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)

    # Normalize: entropy / max_entropy (log2 of #entities)
    max_entropy = math.log(len(probs), 2) if len(probs) > 1 else 1
    score = entropy / max_entropy
    return score


if __name__ == "__main__":
    args = parse_args()
    score = entity_coherence_score([args.text], args.model)
    print(f"Entity Coherence Score: {score:3f}")
