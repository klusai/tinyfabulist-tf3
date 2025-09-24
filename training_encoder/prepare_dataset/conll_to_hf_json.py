import datasets

labels = ["O", "B-ENTITY", "I-ENTITY", "B-LOCATION", "I-LOCATION"]
label2id = {l: i for i, l in enumerate(labels)}

def read_conll(path):
    sentences = []
    tags = []
    with open(path, encoding="utf-8") as f:
        tokens, ner_tags = [], []
        for line in f:
            line = line.strip()
            if not line:  # end of sentence
                if tokens:
                    sentences.append(tokens)
                    tags.append(ner_tags)
                    tokens, ner_tags = [], []
            else:
                word, tag = line.split()
                tokens.append(word)
                ner_tags.append(label2id[tag])
        if tokens:  # last sentence
            sentences.append(tokens)
            tags.append(ner_tags)
    return datasets.Dataset.from_dict({"tokens": sentences, "ner_tags": tags})

dataset = read_conll("training_encoder/ner_dataset.conll")
dataset.save_to_disk("training_encoder/ner_dataset")