import numpy as np
# include additional dependencies as needed


class TrainingSetSelection:
    def __init__(
        self, target_vectors, target_ids, nontarget_vectors, nontarget_ids
    ) -> None:
        self.target_vectors = target_vectors
        self.target_ids = target_ids
        self.nontarget_ectors = nontarget_vectors
        self.nontarget_ids = nontarget_ids

    def select(self):
        """"
        Replace this with your custom training set selection algorithm

        Returns: 
            train_x, train_y
                conform to expected inputs for a logistic regression classifier
                https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """
        
        # inspect some of the training data
        print(self.target_vectors.shape)
        print(self.nontarget_vectors.shape)
        print(len(self.target_ids), self.target_ids[0])
        print(len(self.nontarget_ids), self.nontarget_ids[0])

        rng = np.random.RandomState(0)

        sel_target_idxs = rng.choice(len(self.target_ids), 10, replace=False)
        sel_nontarget_idxs = rng.choice(len(self.nontarget_ids), 10, replace=False)


        train_x = np.vstack(
            [
                self.target_vectors[sel_target_idxs],
                self.nontarget_vectors[sel_nontarget_idxs],
            ]
        )
        train_y = np.concatenate(
            [np.ones(len(sel_target_idxs)), np.zeros(len(sel_nontarget_idxs))]
        )

        return train_x, train_y
