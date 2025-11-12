"""
Get labels for the NER dataset

- O: other
- B-ENTITY: beginning of an entity
- I-ENTITY: inside an entity (continuation of the entity)
- B-LOCATION: beginning of a location
- I-LOCATION: inside a location (continuation of the location)
"""


def get_labels():
    labels = ["O", "B-ENTITY", "I-ENTITY", "B-LOCATION", "I-LOCATION"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label
