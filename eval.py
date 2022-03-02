import sklearn.linear_model
import fire
import numpy as np
import yaml
from selection.serialization import deserialize


def main(eval_file, input_samples, train_file="train.yaml"):

    eval_data = deserialize.deserialize_from_pb(eval_file)
    eval_target = eval_data["target_mswc_vectors"]
    eval_nontarget = eval_data["nontarget_mswc_vectors"]
    eval_x = np.vstack([eval_target, eval_nontarget])
    eval_y = np.concatenate(
        [np.ones(eval_target.shape[0]), np.zeros(eval_nontarget.shape[0])]
    )

    sample_data = deserialize.deserialize_from_pb(input_samples)
    target_vectors = sample_data["target_mswc_vectors"]
    target_ids = sample_data["target_ids"]
    nontarget_vectors = sample_data["nontarget_mswc_vectors"]
    nontarget_ids = sample_data["nontarget_ids"]

    # TODO(mmaz) ensure these are a proper subset
    with open(train_file, "r") as fh:
        train_data = yaml.safe_load(fh)

    train_target_ids = set(train_data["target_ids"])
    train_nontarget_ids = set(train_data["nontarget_ids"])
    train_target_vectors = []
    for ix, target_id in enumerate(target_ids):
        if target_id in train_target_ids:
            train_target_vectors.append(target_vectors[ix])
    train_nontarget_vectors = []
    for ix, nontarget_id in enumerate(nontarget_ids):
        if nontarget_id in train_nontarget_ids:
            train_nontarget_vectors.append(nontarget_vectors[ix])

    train_x = np.vstack([train_target_vectors, train_nontarget_vectors])
    train_y = np.concatenate(
        [np.ones(len(train_target_vectors)), np.zeros(len(train_nontarget_vectors))]
    )

    clf = sklearn.linear_model.LogisticRegression(random_state=0).fit(train_x, train_y)
    print("eval score", clf.score(eval_x, eval_y))


if __name__ == "__main__":
    fire.Fire(main)
