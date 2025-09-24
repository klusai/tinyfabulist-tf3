from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_dir = "/home/andrei/Documents/Work/tf3/bert-ner-model/checkpoint-6384"
base_model = "bert-base-multilingual-cased"   # <-- or whatever you fine-tuned from

# Ensure we load a FAST tokenizer (needs tokenizer.json)
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model     = AutoModelForTokenClassification.from_pretrained(model_dir)

ner = pipeline(
    task="token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",   # <-- Option A pairing
    ignore_labels=["O"]
)

text = "albina harnică  se simțea tristă și singură. A încercat să vorbească cu albina, dar aceasta doar a bâzâit și a zburat departe. Albina a hotărât să ia lucrurile în propriile aripi. A zburat până la stupul albinei și a spus: „Te rog, dă-mi înapoi floarea mea. Este specială pentru mine.” Albina, surprinsă de tonul calm al albinei, a răspuns: „De ce să te ajut? Eu sunt cea care a fost aici tot timpul.” Albina a explicat: „Nu am vrut să te rănesc. Am vrut doar să fiu la conducere. Am crezut că ar fi distractiv să fim prietene.” Albina s-a gândit o clipă și apoi a spus: „Îmi pare rău și mie. Am fost prea mândră ca să cer ajutor.” Albina a iertat-o pe albină și a spus: „Hai să lucrăm împreună ca să construim o casă nouă pentru amândouă.” Albina și albina au devenit cele mai bune prietene, iar."

for entity in ner(text):
    print(entity["entity_group"], entity["word"], entity["score"])
