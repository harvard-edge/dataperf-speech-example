from typing import Optional
from pathlib import Path
import os
import fire
import yaml
import json
import pandas as pd
import numpy as np
import tqdm
import importlib

from selection.load_samples import load_samples
from selection.selection import TrainingSetSelection


def main(
    language: str,
    train_size: int,
    allowed_training_set: os.PathLike = None,
    train_embeddings_dir: os.PathLike = None,
    audio_dir: Optional[os.PathLike] = None,
    config_file: os.PathLike = "workspace/dataperf_speech_config.yaml",
    outdir: os.PathLike = "workspace",
):
    """
    Entrypoint for the training set selection algorithm. Challenge participants
    should *NOT MODIFY* main.py, and should instead modify selection.py (adding
    additional modules and dependencies is also fine, but the selection algorithm
    should be able to run offline without network access).

    :param language: language of the training set selection, either "en" (English),
      "id" (Indonesian), or "pt" (Portuguese)

    :param train_size: number of training samples to select, see 
      dataperf_speech_config.yaml for allowed options

    :param allowed_training_set: path to a yaml file containing the allowed clip
      IDs for training set selection, organized as a dictionary of potential target
      samples and a list of potential nontarget samples.

    :param train_embeddings_dir: directory containing the training feature
      vectors, i.e., embeddings, stored as parquet files.

    :param audio_dir: optional, a directory containing audio files for MSWC
      samples, encoded as 16KHz wavs. Your selection algorithm can solely consider
      the embeddings, but if you also wish to use audio, this parameter must be
      specified as a directory to the MSWC 16KHz wav samples.

    :param config_file: path to a yaml file containing the configuration for the
      experiment, such as random seeds and the maximum number of training samples.
      You can extend this config file if needed.

    :param outdir: output directory to save the selected training set as a yaml file

    """
    if language not in ['en', 'id', 'pt']:
        raise ValueError(f"language {language} not supported. Supported languages are: en, id, pt")

    if train_embeddings_dir is None:
        train_embeddings_dir = f"workspace/data/dataperf_{language}_data/train_embeddings"

    if allowed_training_set is None:
        allowed_training_set = f"workspace/data/dataperf_{language}_data/allowed_training_set.yaml"

    assert Path(
        outdir
    ).is_dir(), (
        f"{outdir} does not exist, please specify --outdir as a command line argument"
    )

    # dict {"targets": {"dog":[list]}, "nontargets": [list]}
    allowed_training_ids = yaml.safe_load(Path(allowed_training_set).read_text())

    allowed_training_embeddings = load_samples(
        sample_ids=allowed_training_ids,
        embeddings_dir=train_embeddings_dir,
        audio_dir=audio_dir,
    )

    audio_flag = False
    if audio_dir is not None:
        audio_flag = True

    config = yaml.safe_load(Path(config_file).read_text())
    module = importlib.import_module(config["selection_algorithm_module"])
    class_ = getattr(module, config["selection_algorithm_class"])
    assert issubclass(
        class_, TrainingSetSelection
    ), f"loaded class {config['selection_algorithm_module']}{config['selection_algorithm_class']} is not a subclass of TrainingSetSelection"

    # TODO(mmaz) support class arguments

    selection = class_(
        allowed_embeddings=allowed_training_embeddings,
        train_size=train_size,
        config=config,
        audio_flag=audio_flag,
    )

    train = selection.select()

    n_selected = sum([len(sample_ids) for sample_ids in train.targets.values()]) + len(
        train.nontargets
    )

    assert (
        n_selected <= train_size
    ), f"selected {n_selected} samples, expected {train_size}"

    output = Path(outdir) / f"{language}_{train_size}_train.json"
    output.write_text(
        json.dumps(
            dict(targets=train.targets, nontargets=train.nontargets),
            sort_keys=True,
            indent=4,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
