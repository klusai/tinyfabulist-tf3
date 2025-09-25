from collections import defaultdict

import spacy

nlp = spacy.load("ro_core_news_sm")  # Romanian model

from collections import Counter


def entity_recurrence_score(texts):
    """
    Score how much entities are repeated across a set of texts.
    High score = entities are reused frequently (coherent).
    Low score = entities mostly appear once (incoherent).
    """
    entity_counts = Counter()
    total_mentions = 0

    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            entity_counts[ent.text] += 1
            total_mentions += 1

    if total_mentions == 0:
        return 1.0  # no entities = trivially coherent

    # reward repetitions: only counts beyond the first matter
    repeated_mentions = sum(entity_counts.values()) - len(entity_counts)

    print(entity_counts, repeated_mentions, total_mentions)

    return repeated_mentions / total_mentions


# Example
texts = [
    "vulpea isteață  se ascundea în umbre, încercând să se ascundă după un tufiș. Într-o zi, o furtună puternică a lovit ruinele, iar casa ursului a fost distrusă. Ursul, speriat și singur, a strigat după ajutor. Ursul, văzându-și prietenul în necaz, a sărit în ajutor. S-a cățărat cu grijă pe o colină din apropiere și a găsit o peșteră primitoare. Înăuntrul peșterii, ursul a văzut un grup de păsări prins într-o peșteră din apropiere. Ursul ar fi putut să le mănânce, dar și-a amintit cuvintele înțelepte ale mamei sale: „Un adevărat prieten este cel care ne face puternici.” Ursul a hotărât să-și folosească puterea pentru a ridica cu grijă peștera, iar împreună au așteptat să treacă furtuna. Când a ieșit soarele, ursul a spus: „Nu credeam că pot să fac asta. Dar acum văd că ești bună și de ajutor.” Ursul a zâmbit, iar cei doi au devenit prieteni. Din acea zi, ursul a învățat că"
]

score = entity_recurrence_score(texts)
print(f"Entity Coherence Score: {score:.3f}")
