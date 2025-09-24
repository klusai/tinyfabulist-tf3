from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import datasets

labels = ["O", "B-ENTITY", "I-ENTITY", "B-LOCATION", "I-LOCATION"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],  # assumes your dataset has 'tokens' column
        truncation=True,
        is_split_into_words=True
    )
    
    labels_aligned = []
    for i, label in enumerate(example["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)  # ignored in loss
            elif word_id != previous_word: #only first token of the word is tagged, rest are -100
                aligned.append(label[word_id])
            else:
                aligned.append(-100)  # mask subword continuations
            previous_word = word_id
        labels_aligned.append(aligned)

    tokenized["labels"] = labels_aligned
    return tokenized

dataset = datasets.load_from_disk("training_encoder/ner_dataset")

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# remove text columns the model can't handle
tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])

print(dataset[0])  # before tokenization
print(tokenized_datasets[0])  # after tokenization

tokenized_datasets.save_to_disk("training_encoder/ner_dataset_tokenized/")