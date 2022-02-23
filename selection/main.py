import fire
import numpy as np
from selection.serialization import deserialize


def main(train_file):
    train_data = deserialize.deserialize_from_pb(train_file)
    target_vectors = train_data["target_mswc_vectors"]
    target_ids = train_data["target_ids"]
    nontarget_vectors = train_data["nontarget_mswc_vectors"]
    nontarget_ids = train_data["nontarget_ids"]

    # inspect some of the training data
    print(target_vectors.shape)
    print(nontarget_vectors.shape)
    print(len(target_ids), target_ids[0])
    print(len(nontarget_ids), nontarget_ids[0])

    rng = np.random.RandomState(0)

    sel_target_idxs = rng.choice(len(target_ids), 10, replace=False)
    sel_nontarget_idxs = rng.choice(len(nontarget_ids), 10, replace=False)
   
    # conform to expected inputs for a logistic regression classifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    train_x = np.vstack([target_vectors[sel_target_idxs], nontarget_vectors[sel_nontarget_idxs]])
    train_y = np.concatenate([np.ones(len(sel_target_idxs)), np.zeros(len(sel_nontarget_idxs))])

    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)


if __name__ == "__main__":
    fire.Fire(main)