from pathlib import Path

import fire
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import tqdm
import yaml
import json

from selection.load_samples import load_samples

from sklearn.metrics import f1_score


def validate_selected_ids(selected_ids, allowed_training_ids, train_set_size_limit):
    groundtruth_targets = set(allowed_training_ids["targets"].keys())
    for target in selected_ids["targets"].keys():
        assert target in groundtruth_targets, f"target {target} not in allowed set"

    groundtruth_target_ids = {
        target: set(samples)
        for target, samples in allowed_training_ids["targets"].items()
    }
    for target, samples in selected_ids["targets"].items():
        assert set(samples).issubset(
            groundtruth_target_ids[target]
        ), f"{target} contains an ID not in allowed set"

    groundtruth_nontarget_ids = set(allowed_training_ids["nontargets"])
    assert set(selected_ids["nontargets"]).issubset(
        groundtruth_nontarget_ids
    ), f"nontargets contain an ID not in allowed set"

    n_training_samples = sum(
        [len(samples) for samples in selected_ids["targets"].values()]
    ) + len(selected_ids["nontargets"])
    assert (
        n_training_samples <= train_set_size_limit
    ), f"{n_training_samples} samples exceeds limit of {train_set_size_limit}"


def create_dataset(embeddings):
    """
    Creates an sklearn-compatible dataset from an embedding dict
    """
    target_to_classid = {
        target: ix + 1 for ix, target in enumerate(sorted(embeddings["targets"].keys()))
    }
    target_to_classid["nontarget"] = 0

    target_samples = np.array(
        [
            sample["feature_vector"]
            for target, samples in embeddings["targets"].items()
            for sample in samples
        ]
    )

    target_labels = np.array(
        [
            target_to_classid[target]
            for (target, samples) in embeddings["targets"].items()
            for sample in samples
        ]
    )

    nontarget_samples = np.array(
        [sample["feature_vector"] for sample in embeddings["nontargets"]]
    )
    nontarget_labels = np.zeros(nontarget_samples.shape[0])

    Xs = np.vstack([target_samples, nontarget_samples])
    ys = np.concatenate([target_labels, nontarget_labels])
    return Xs, ys


def main(
    language,
    train_size: int,
    eval_embeddings_dir=None,  # embeddings dir point to the same parquet file for testing and online eval
    train_embeddings_dir=None,
    allowed_training_set=None,
    eval_file=None,
    train_file=None,
    config_file="workspace/dataperf_speech_config.yaml",
):

    if language not in ['en', 'id', 'pt']:
        raise ValueError(f"language {language} not supported. Supported languages are: en, id, pt")
    
    dynabench_sizes = [25, 60]
    if train_size not in dynabench_sizes:
        print(f"Warning: train_size {train_size} does not match one of the leaderboards on dynabench.\
                The dynabench leaderboads support train size options: {dynabench_sizes}\
                You can submit a train set that's smaller than one of the options, but it will \
                be compared against submissions of size up to and including the leaderboard max size.")

    if eval_embeddings_dir is None:
        eval_embeddings_dir = f"workspace/data/dataperf_{language}_data/eval_embeddings"

    if train_embeddings_dir is None:
        train_embeddings_dir = f"workspace/data/dataperf_{language}_data/train_embeddings"

    if allowed_training_set is None:
        allowed_training_set = f"workspace/data/dataperf_{language}_data/allowed_training_set.yaml"

    if eval_file is None:
        eval_file = f"workspace/data/dataperf_{language}_data/eval.yaml"

    if train_file is None:
        train_file = f"workspace/{language}_{train_size}_train.json"

    config = yaml.safe_load(Path(config_file).read_text())
    eval_random_seeds = config["eval_random_seeds"]

    allowed_training_ids = yaml.safe_load(Path(allowed_training_set).read_text())
    selected_ids = json.loads(Path(train_file).read_text())

    print("validating selected IDs")
    validate_selected_ids(selected_ids, allowed_training_ids, train_size)

    print("loading selected training data")
    selected_embeddings = load_samples(
        sample_ids=selected_ids, embeddings_dir=train_embeddings_dir
    )
    print("loading eval data")
    eval_ids = yaml.safe_load(Path(eval_file).read_text())
    eval_embeddings = load_samples(
        sample_ids=eval_ids, embeddings_dir=eval_embeddings_dir
    )

    train_x, train_y = create_dataset(selected_embeddings)

    #average the scores over multiple random seeds
    scores = []
    for random_seed in eval_random_seeds:
        clf = sklearn.ensemble.VotingClassifier(
            estimators=[
                ("svm", sklearn.svm.SVC(probability=True, random_state=random_seed)),
                ("lr", sklearn.linear_model.LogisticRegression(random_state=random_seed)),
            ],
            voting="soft",
            weights=None,
        )
        clf.fit(train_x, train_y)

        # eval
        eval_x, eval_y = create_dataset(eval_embeddings)

        pred_y = clf.predict(eval_x)
        scores.append(f1_score(eval_y, pred_y, average="macro"))

    average_score = np.mean(scores)
    print("Score: ", average_score)


if __name__ == "__main__":
    fire.Fire(main)
