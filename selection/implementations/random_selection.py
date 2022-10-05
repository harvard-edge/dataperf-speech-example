from typing import Dict, List
from dataclasses import dataclass

from selection.selection import TrainingSetSelection, TrainingSet

# include additional dependencies as needed:
import random

class RandomSelection(TrainingSetSelection):
    def __init__(self, **kwargs) -> None:
        super(RandomSelection, self).__init__(**kwargs)


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
