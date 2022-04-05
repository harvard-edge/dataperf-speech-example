from typing import Optional
from pathlib import Path
import os
import fire
import yaml
import pandas as pd
import numpy as np
import tqdm

from selection.selection import TrainingSetSelection


def main(
    allowed_training_set: os.PathLike,
    train_embeddings_dir: os.PathLike = "/embeddings/en",
    audio_dir: Optional[os.PathLike] = None,
    config_file: os.PathLike = "config_files/dataperf_speech_config.yaml",
    outdir: os.PathLike = "/workdir",
):
    """
    Entrypoint for the training set selection algorithm. Challenge participants 
    should *NOT MODIFY* main.py, and should instead modify selection.py (adding 
    additional modules and dependencies is also fine, but the selection algorithm
    should be able to run offline without network access). 

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

    # TODO(mmaz) need an override mechanism, see https://github.com/harvard-edge/dataperf-speech-example/issues/3
    with open(config_file, "r") as fh:
        config = yaml.safe_load(fh)

    embedding_dataset = Path(train_embeddings_dir)

    with open(allowed_training_set, "r") as fh:
        allowed_training_embeddings = yaml.safe_load(
            fh
        )  # dict {"targets": {"dog":[list]}, "nontargets": [list]}

    train_embeddings = dict(targets={}, nontargets=[])
    # {"targets": {"dog":[{'ID':string,'feature_vector':np.array,'audio':np.array}, ...], ...}, "nontargets": [list]}
    for target, id_list in tqdm.tqdm(allowed_training_embeddings["targets"].items(), desc="Loading targets"):
        train_embeddings["targets"][target] = []
        target_parquet = pd.read_parquet(embedding_dataset / (target + ".parquet"))
        allowed_ids_mask = target_parquet["clip_id"].isin(id_list)
        for row in target_parquet[allowed_ids_mask].itertuples():
            train_embeddings["targets"][target].append(
                dict(ID=row.clip_id, feature_vector=row.mswc_embedding_vector)
            )

    for id in tqdm.tqdm(allowed_training_embeddings["nontargets"], desc="Loading nontargets"):
        label = Path(id).parts[0]  # "cat/common_voice_id_12345.wav"
        parquet_file = pd.read_parquet(embedding_dataset / (label + ".parquet"))
        row = parquet_file.loc[parquet_file["clip_id"] == id].iloc[0]
        train_embeddings["nontargets"].append(
            dict(ID=row.clip_id, feature_vector=row.mswc_embedding_vector)
        )

    audio_flag = False
    if audio_dir is not None:
        # TODO Test with wav files
        from scipy.io import wavfile

        audio_flag = True
        for target, sample_list in train_embeddings["targets"].items():
            for sample in sample_list:
                _, audio = wavfile.read(audio_dir + "/" + sample["ID"])
                sample["audio"] = audio
        for sample in train_embeddings["nontargets"]:
            _, audio = wavfile.read(audio_dir + "/" + sample["ID"])
            sample["audio"] = audio

    selection = TrainingSetSelection(
        train_embeddings=train_embeddings, config=config, audio_flag=audio_flag,
    )

    train = selection.select()

    assert Path(
        outdir
    ).is_dir(), (
        f"{outdir} does not exist, please specify --outdir as a command line argument"
    )

    n_selected = sum([len(sample_ids) for sample_ids in train.targets.values()]) + len(
        train.nontargets
    )
    assert (
        n_selected < config["train_set_size_limit"]
    ), f"{n_selected} samples selected, but the limit is {config['train_set_size_limit']}"

    output = Path(outdir) / "train.yaml"
    with open(output, "w") as fh:
        yaml.dump(
            dict(targets=train.targets, nontargets=train.nontargets),
            fh,
            default_flow_style=None,
            sort_keys=False,
        )


if __name__ == "__main__":
    fire.Fire(main)
