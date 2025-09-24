from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_dir = "bert-ner-model/checkpoint-3192"
base_model = "bert-base-multilingual-cased"  

# Ensure we load a FAST tokenizer (needs tokenizer.json)
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model     = AutoModelForTokenClassification.from_pretrained(model_dir)

ner = pipeline(
    task="token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",  
    ignore_labels=["O"]
)

text = "puma agilă și îi conducea de obicei pe ceilalți pe colinele abrupte ale deșertului. Într-o zi, o vulpe șireată a început să le șoptească celorlalți: „De ce urmăm mereu aceleași cărări? De ce să urmăm doar poteci vechi și cărări înguste? Așa am putea descoperi ceva nou și palpitant!” Puma devotată a ascultat cu atenție, dar cuvintele vulpii i-au stârnit curiozitatea. Puma a hotărât să exploreze ambele poteci, dar de fiecare dată când încerca, se rătăcea. Vulpea a încercat să o urmeze, dar s-a rătăcit în labirintul de coridoare al labirintului. Într-o zi, puma a dat peste o peșteră ascunsă. Înăuntru a găsit suluri străvechi care șopteau secretele străvechi ale ținutului. Puma a citit despre importanța de a asculta și de a învăța din greșeli. Puma s-a întors în peșteră, nerăbdătoare să învețe și să-și împărtășească cunoștințele. A descoperit că fiecare creatură pe care o întâlnea era o exploratoare curioasă și aventuroasă, care învăța și se juca. Puma a realizat"

for entity in ner(text):
    print(entity["entity_group"], entity["word"], entity["score"])