import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default="bert-ner-model/checkpoint-3192"
    )
    parser.add_argument(
        "--base-model", type=str, default="bert-base-multilingual-cased"
    )
    parser.add_argument("--aggregation-strategy", type=str, default="first")
    parser.add_argument(
        "--text",
        type=str,
        default="puma agilă și îi conducea de obicei pe ceilalți pe colinele abrupte ale deșertului. Într-o zi, o vulpe șireată a început să le șoptească celorlalți: „De ce urmăm mereu aceleași cărări? De ce să urmăm doar poteci vechi și cărări înguste? Așa am putea descoperi ceva nou și palpitant!” Puma devotată a ascultat cu atenție, dar cuvintele vulpii i-au stârnit curiozitatea. Puma a hotărât să exploreze ambele poteci, dar de fiecare dată când încerca, se rătăcea. Vulpea a încercat să o urmeze, dar s-a rătăcit în labirintul de coridoare al labirintului. Într-o zi, puma a dat peste o peșteră ascunsă. Înăuntru a găsit suluri străvechi care șopteau secretele străvechi ale ținutului. Puma a citit despre importanța de a asculta și de a învăța din greșeli. Puma s-a întors în peșteră, nerăbdătoare să învețe și să-și împărtășească cunoștințele. A descoperit că fiecare creatură pe care o întâlnea era o exploratoare curioasă și aventuroasă, care învăța și se juca. Puma a realizat",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure we load a FAST tokenizer (needs tokenizer.json)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=args.aggregation_strategy,
        ignore_labels=["O"],
    )

    for entity in ner(args.text):
        print(entity["entity_group"], entity["word"], entity["score"])
