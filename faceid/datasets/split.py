import random
from copy import deepcopy


def split_dataset_by_identity(
    dataset,
    train_ratio=0.8,
    seed=42,
):
    """
    Split FaceDataset by identities instead of images.
    """

    random.seed(seed)

    labels = list(dataset.person_to_images.keys())

    random.shuffle(labels)

    split = int(len(labels) * train_ratio)

    train_labels = set(labels[:split])
    val_labels = set(labels[split:])

    train_dataset = deepcopy(dataset)
    val_dataset = deepcopy(dataset)

    train_dataset.image_paths = []
    train_dataset.labels = []
    train_dataset.samples = []
    train_dataset.person_to_images = {}
    train_dataset.person_to_label = {}
    train_dataset.label_to_person = {}

    val_dataset.image_paths = []
    val_dataset.labels = []
    val_dataset.samples = []
    val_dataset.person_to_images = {}
    val_dataset.person_to_label = {}
    val_dataset.label_to_person = {}

    for label in train_labels:

        train_dataset.person_to_images[label] = dataset.person_to_images[label]

        person = dataset.label_to_person[label]

        train_dataset.label_to_person[label] = person
        train_dataset.person_to_label[person] = label

        for image in dataset.person_to_images[label]:

            train_dataset.image_paths.append(image)
            train_dataset.labels.append(label)
            train_dataset.samples.append((image, label))

    for label in val_labels:

        val_dataset.person_to_images[label] = dataset.person_to_images[label]

        person = dataset.label_to_person[label]

        val_dataset.label_to_person[label] = person
        val_dataset.person_to_label[person] = label

        for image in dataset.person_to_images[label]:

            val_dataset.image_paths.append(image)
            val_dataset.labels.append(label)
            val_dataset.samples.append((image, label))

    return train_dataset, val_dataset