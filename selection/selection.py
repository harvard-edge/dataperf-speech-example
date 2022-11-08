from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class TrainingSet:
    """
    dict {"targets": {"dog":[list of IDs], ...}, "nontargets": [list of IDs]}
    """

    targets: Dict[str, List[str]]
    nontargets: List[str]


class TrainingSetSelection:
    def __init__(
        self,
        allowed_embeddings: Dict[str, Any],
        config: Dict[str, Any],
        audio_flag: bool = False,
    ) -> None:
        """
        Args:
            allowed_embeddings: dict {"targets": {"dog":[{'ID':string,'feature_vector':np.array,'audio':np.array}, ...], ...}, "nontargets": [list]}

            config: see dataperf_speech_config.yaml

            train_set_size: int (total number of samples to select)

            audio_flag: bool (if audio is included in the allowed_embeddings)

        """
        self.embeddings = allowed_embeddings
        # {"targets": {"dog":[{'ID':string,'feature_vector':np.array,'audio':np.array}, ...], ...},
        #  "nontargets": [{'ID':string,'feature_vector':np.array,'audio':np.array}, ...]}
        self.config = config
        self.train_set_size = config["train_set_size_limit"]
        self.random_seed = config["random_seed"]
        self.audio_flag = audio_flag

    def select(self):
        """ "
        Returns:
            TrainingSet
        """
        raise NotImplementedError
