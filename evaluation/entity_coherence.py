import argparse
from collections import Counter
from typing import Dict, List

import stanza
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-ner-model/checkpoint-3192")
    parser.add_argument(
        "--text",
        type=str,
        default="Un iepure blând a sugerat: „Haideți să lucrăm împreună ca să construim o casă nouă pentru iepure, iar ea va fi atât de fericită încât va putea să se joace cu noi.” Dar iepurele a refuzat, spunând: „Nu, o voi face singur. Nu am nevoie de ajutorul nimănui.” Iepurele a încercat să o convingă, dar ea nu a vrut să asculte. Pe măsură ce treceau zilele, iepurele a observat că celelalte animale au început să se certe și să se lupte. Iepurele și-a dat seama că, fără ajutorul lor, nu mai avea cu cine să se joace. Într-o zi, iepurele a avut o idee. „Haideți să construim o casă nouă pentru toți, iar eu voi ajuta la construirea unei noi case pentru o familie de păsări.” Celelalte animale au fost de acord, iar împreună au ridicat o casă frumoasă și primitoare. Iepurele a învățat că, atunci când îi ajuți pe ceilalți, și ei te vor ajuta. Din acea zi, iepurele a fost cunoscut drept cel mai bun iepure din pădure. Iar ori de câte ori animalele aveau nevoie de ajutor, știau că se pot baza pe iepurele bun care îi ajutase. Morala poveștii este: Ajutorul la timp aduce loialitate de durată. Într-o poiană însorită, plină de flori colorate, trăia o mică buburuză pe nume Inimă Bună. Îi plăcea să zboare de la o floare la alta, adunând nectar pentru prietenii ei. Într-o zi, în timp ce sorbea dintr-o floare galbenă strălucitoare, Inimă Bună a întâlnit un fluture frumos numit Aripi Blânde. S-au îndrăgostit, iar Inimă Bună a crezut că ar fi cel mai bun prieten al ei. Dar, pe măsură ce trecea timpul, Inimă Bună a început să se simtă tristă și singură. Îi era dor de Aripi Blânde și voia să fie cu ea, dar nu știa cum. A început să se gândească: „De ce nu pot fi prietenă cu Aripi Blânde? E atât de frumoasă și de puternică.” Inimă Bună a început să se poarte urât cu Aripi Blânde, încercând să o facă să se simtă rău. Dar Aripi Blânde nu a renunțat. A continuat să zboare și să se joace cu Inimă Bună, iar curând au devenit cei mai buni prieteni. Într-o zi, în timp ce Inimă Bună zbura, a văzut un grup de fluturi care se chinuiau să găsească nectar în poiană. A",
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
    Score how much entities are repeated across a set of texts.
    High score = entities are reused frequently (coherent).
    Low score = entities mostly appear once (incoherent).
    """

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForTokenClassification.from_pretrained(model)

    # Create inference pipeline
    ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="first"
    )

    # Build a Romanian lemmatizer once
    nlp = stanza.Pipeline(
        "ro",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=False,
        verbose=False,
    )

    def lemmatize_token(token: str) -> str:
        doc = nlp(token)
        for sent in doc.sentences:
            for w in sent.words:
                return w.lemma.lower()
        return token.lower()

    lemma_counts = Counter()
    total_mentions = 0

    for text in texts:
        entities = ner_pipeline(text)
        for ent in entities:
            lemma = lemmatize_token(ent["word"])  # robust to different surface forms
            lemma_counts[lemma] += 1
            total_mentions += 1

    if total_mentions == 0:
        return 1.0  # no entities = trivially coherent

    print(lemma_counts)

    # Reward repetitions: only counts beyond the first mention of each lemma
    repeated_mentions = sum(lemma_counts.values()) - len(lemma_counts)

    return repeated_mentions / total_mentions


if __name__ == "__main__":
    args = parse_args()
    score = entity_coherence_score([args.text], args.model)
    print(f"Entity Coherence Score: {score:.3f}")
