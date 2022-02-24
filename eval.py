from yaml import CLoader as Loader, CDumper as Dumper
import sklearn.linear_model
import fire
import numpy as np
from selection.serialization import deserialize


def main(eval_file, train_file="train.npz"):
    eval_data = deserialize.deserialize_from_pb(eval_file)
    eval_target = eval_data["target_mswc_vectors"]
    eval_nontarget = eval_data["nontarget_mswc_vectors"]
    eval_x = np.vstack([eval_target, eval_nontarget])
    eval_y = np.concatenate(
        [np.ones(eval_target.shape[0]), np.zeros(eval_nontarget.shape[0])]
    )

    # TODO(mmaz) ensure these are a proper subset
    train = np.load(train_file)
    train_x = train["train_x"]
    train_y = train["train_y"]

    clf = sklearn.linear_model.LogisticRegression(random_state=0).fit(train_x, train_y)
    print("eval score", clf.score(eval_x, eval_y))


if __name__ == "__main__":
    fire.Fire(main)
