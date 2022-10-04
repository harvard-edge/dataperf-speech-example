from typing import Dict, List
from dataclasses import dataclass


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
        """ "
        Returns:
            TrainingSet
        """
        raise NotImplementedError
