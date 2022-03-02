from typing import List
import numpy as np
import dataclasses


@dataclasses.dataclass
class TrainingSet:
    target_ids: List[str]
    nontarget_ids: List[str]


# include additional dependencies as needed


class TrainingSetSelection:
    def __init__(
        self,
        target_vectors: np.ndarray,
        target_ids: List[str],
        nontarget_vectors: np.ndarray,
        nontarget_ids: List[str],
    ) -> None:
        """
        Arguments:
            target_vectors:
            target_ids:
            nontarget_vectors:
            nontarget_ids:
        """
        self.target_vectors = target_vectors
        self.target_ids = target_ids
        self.nontarget_mswc_vectors = nontarget_vectors
        self.nontarget_ids = nontarget_ids

    def select(self):
        """"
        Replace this with your custom training set selection algorithm

        Returns: 
            TrainingSet
        """

        # inspect some of the training data
        print(self.target_vectors.shape)
        print(self.nontarget_mswc_vectors.shape)
        print(len(self.target_ids), self.target_ids[0])
        print(len(self.nontarget_ids), self.nontarget_ids[0])

        rng = np.random.RandomState(0)

        sel_target_idxs = rng.choice(len(self.target_ids), 10, replace=False)
        sel_nontarget_idxs = rng.choice(len(self.nontarget_ids), 10, replace=False)

        training_set = TrainingSet(
            target_ids=[self.target_ids[ix] for ix in sel_target_idxs],
            nontarget_ids=[self.nontarget_ids[ix] for ix in sel_nontarget_idxs],
        )

        return training_set
