from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import tqdm

# include additional dependencies as needed:
import random

@dataclass
class TrainingSet:
    """
    dict {"targets": {"dog":[list of IDs], ...}, "nontargets": [list of IDs]}
    """

    targets: Dict[str, List[str]]
    nontargets: List[str]


class TrainingSetSelection:
    def __init__(self, allowed_embeddings, config, audio_flag=False) -> None:
        """
        Args:
            allowed_embeddings: dict {"targets": {"dog":[{'ID':string,'feature_vector':np.array,'audio':np.array}, ...], ...}, "nontargets": [list]}

            train_set_size: int (total number of samples to select)

            audio_flag: bool (if audio is included in the allowed_embeddings)

        """

        self.embeddings = allowed_embeddings
        # {"targets": {"dog":[{'ID':string,'feature_vector':np.array,'audio':np.array}, ...], ...},
        #  "nontargets": [{'ID':string,'feature_vector':np.array,'audio':np.array}, ...]}
        self.train_set_size = config["train_set_size_limit"]
        self.random_seed = config["random_seed"]
        self.audio_flag = audio_flag

    def select(self):
        """"
        Returns: 
            TrainingSet
        """

        #random selection baseline
        #each class is evenly balanced
        random.seed(0) #fixed seed for consistency

        num_classes = len(self.embeddings['targets'].keys()) + 1 #num targets + one non-target class

        per_class_size = self.train_set_size // num_classes

        selected_targets = {k: [] for k in self.embeddings["targets"].keys()}
        for target in selected_targets.keys():
            selected_targets[target] = [sample['ID'] for sample in random.sample(self.embeddings['targets'][target], per_class_size)]

        selected_nontargets = [sample['ID'] for sample in random.sample(self.embeddings['nontargets'], per_class_size)]

        

        training_set = TrainingSet(
            targets=selected_targets, nontargets=selected_nontargets,
        )

        return training_set
