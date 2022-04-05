import sklearn.linear_model
import fire
import numpy as np
import yaml
from pathlib import Path
import pandas as pd

# Constants
label_idx = 0  # IDs look like "and/common_voice..." this is the index of the label


def create_train_sets(
    selected_data, allowed_training_embeddings, training_classes, train_dataset
):
    train_x = []  # embeddings
    train_y = []  # numerical index of training_classes
    allowed_nontargets = set(allowed_training_embeddings["nontargets"])
    for selected_id in selected_data["nontargets"]:
        if not selected_id in allowed_nontargets:
            print("invaid embedding ID: ", selected_id)
            raise ValueError(
                "Selected Embedding ID not present in allowed training data"
            )

        label = Path(selected_id).parts[label_idx]
        if label in training_classes:
            print("invaid ID for nontarget: ", selected_id)
            raise ValueError(
                "ID provided for nontarget class has a label that matches a target"
            )
        parquet_file = pd.read_parquet(train_dataset / (label + ".parquet"))
        idx = parquet_file["clip_id"].tolist().index(selected_id)
        train_x.append(parquet_file["mswc_embedding_vector"][idx])
        train_y.append(training_classes.index("nontargets"))

    for label in selected_data["targets"].keys():
        if not label in training_classes:
            raise ValueError("Provided label doesnt match classes in eval set")
        allowed_targets = set(allowed_training_embeddings["targets"][label])
        for selected_id in selected_data["targets"][label]:
            if not selected_id in allowed_targets:
                print("invaid embedding ID: ", selected_id)
                raise ValueError(
                    "Selected Embedding ID not present in allowed training data"
                )
            if not label == Path(selected_id).parts[label_idx]:
                print("invaid label for embedding ID: ", selected_id)
                raise ValueError("Provided label doesnt match label in ID")

            parquet_file = pd.read_parquet(train_dataset / (label + ".parquet"))
            idx = parquet_file["clip_id"].tolist().index(selected_id)
            train_x.append(parquet_file["mswc_embedding_vector"][idx])
            train_y.append(training_classes.index(label))
    return train_x, train_y


def create_eval_sets(eval_data, eval_dataset, training_classes):
    eval_x = []
    eval_y = []
    for id in eval_data["nontargets"]:
        label = Path(id).parts[label_idx]
        parquet_file = pd.read_parquet(eval_dataset / (label + ".parquet"))
        idx = parquet_file["clip_id"].tolist().index(id)
        eval_x.append(parquet_file["mswc_embedding_vector"][idx])
        eval_y.append(training_classes.index("nontargets"))
    for label in eval_data["targets"].keys():
        parquet_file = pd.read_parquet(eval_dataset / (label + ".parquet"))
        for id in eval_data["targets"][label]:
            idx = parquet_file["clip_id"].tolist().index(id)
            eval_x.append(parquet_file["mswc_embedding_vector"][idx])
            eval_y.append(training_classes.index(label))
    return eval_x, eval_y


def main(
    eval_embeddings_dir="embeddings/en",  # embeddings dir point to the same parquet file for testing and online eval
    train_embeddings_dir="embeddings/en",
    allowed_training_set="allowed_training_set.yaml",
    eval_file="eval.yaml",
    train_file="workdir/train.yaml",
    config_file="dataperf_speech_config.yaml",
):

    with open(config_file, "r") as fh:
        config = yaml.safe_load(fh)
    train_set_size_limit = config["train_set_size_limit"]
    random_seed = config["random_seed"]

    with open(allowed_training_set, "r") as fh:
        allowed_training_embeddings = yaml.safe_load(
            fh
        )  # dict {"targets": {"dog":[list]}, "nontargets": [list]}
    training_classes = list(
        allowed_training_embeddings["targets"].keys()
    )  # specifies the label indicies for training
    assert "nontarget" not in training_classes, "nontarget is a reserved label"
    training_classes = [
        "nontargets"
    ] + training_classes  # adds the nontargets class to the list

    train_dataset = Path(train_embeddings_dir)
    eval_dataset = Path(eval_embeddings_dir)

    with open(train_file, "r") as fh:
        selected_data = yaml.safe_load(
            fh
        )  # dict {"targets": {"dog":[list]}, "nontargets": [list]}

    train_x, train_y = create_train_sets(
        selected_data, allowed_training_embeddings, training_classes, train_dataset
    )

    if len(train_y) > train_set_size_limit:
        print("Selected training set size: ", len(train_y))
        print("Training set size limit: ", train_set_size_limit)
        raise ValueError("Selected training set is too large")

    Percept = sklearn.linear_model.Perceptron(random_state=random_seed).fit(
        train_x, train_y
    )
    svm = sklearn.svm.SVC(random_state=random_seed, decision_function_shape="ovr").fit(
        train_x, train_y
    )

    # eval
    with open(eval_file, "r") as fh:
        eval_data = yaml.safe_load(
            fh
        )  # dict {"targets": {"dog":[list]}, "nontargets": [list]}
    eval_x, eval_y = create_eval_sets(eval_data, eval_dataset, training_classes)

    print("Perceptron score", Percept.score(eval_x, eval_y))
    print("svm score", svm.score(eval_x, eval_y))


if __name__ == "__main__":
    fire.Fire(main)
