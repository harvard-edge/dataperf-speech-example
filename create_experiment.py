from dataclasses import dataclass
from typing import List
from pathlib import Path
import json
import gzip
import os
import csv

import yaml
import pandas as pd
import fire
import tqdm
import numpy as np

"""
for i, file in enumerate(target_files):
    target_df = pd.read_parquet(file)
    sampled_df = target_df["clip_id"].sample(
        n=num_per_target[i], random_state=random_seed
    )
    targets[file.stem] = sampled_df.tolist()


while len(nontargets) < num_nontarget:
    file = random.choice(nontarget_files)
    nontarget_df = df = pd.read_parquet(file)
    if nontarget_df.shape[0] < (num_per_word_nontarget * 10):
        continue
    sampled_df = nontarget_df["clip_id"].sample(
        n=num_per_word_nontarget, random_state=random_seed
    )
    nontargets.extend(sampled_df.tolist())


"""


@dataclass
class GenerationParams:
    minimum_samples_for_nontarget_words: int = 200  # nontarget samples do not follow MSWC train/dev/test splits
    num_nontarget_training_words: int = 100
    num_nontarget_eval_words: int = 100  # distinct words the model has not been trained on as nontarget
    seed_nontarget_selection: int = 0


def select_nontarget_samples(meta, language_isocode, target_words):
    """
    
    Returns: dict
        train_eval_nontarget will be equally dividided between training and eval samples
        unknown_nontarget will be used purely for evaluation, as words unseen by the classifier
    """
    print("generating nontarget samples")
    candidate_nontarget_words = []
    for w, c in meta[language_isocode]["wordcounts"].items():
        if (
            c > GenerationParams.minimum_samples_for_nontarget_words
            and w not in target_words
        ):
            candidate_nontarget_words.append(w)

    # repeatability via (1) fixed iter order in metadata.json.gz and (2) fixed seed
    unknown_rng = np.random.RandomState(GenerationParams.seed_nontarget_selection)
    selected_unknowns = unknown_rng.choice(
        candidate_nontarget_words,
        GenerationParams.num_nontarget_training_words
        + GenerationParams.num_nontarget_eval_words,
        replace=False,
    )
    selected_samples = []
    for word in selected_unknowns:
        selected_opus = unknown_rng.choice(
            meta[language_isocode]["filenames"][word],
            GenerationParams.minimum_samples_for_nontarget_words,
            replace=False,
        )
        wavs = [str(Path(word) / (Path(o).stem + ".wav")) for o in selected_opus]
        selected_samples.append((word, wavs))
    train_eval_nontarget = selected_samples[
        : GenerationParams.num_nontarget_training_words
    ]
    unknown_nontarget = selected_samples[
        GenerationParams.num_nontarget_training_words :
    ]
    return dict(
        train_eval_nontarget=train_eval_nontarget, unknown_nontarget=unknown_nontarget
    )


def main(
    path_to_metadata: os.PathLike,
    language_isocode: str,
    path_to_splits_csv: os.PathLike,
    path_to_embeddings: os.PathLike,
    target_words: List[str],
    outdir: os.PathLike,
):
    """
    :param path_to_metadata: MSWC metadata in .json.gz format, can be downloaded 
      from https://storage.googleapis.com/public-datasets-mswc/metadata.json.gz
    
    :param language_isocode: language code, e.g. "en"

    :param path_to_splits_csv: train/test/dev splits CSV file (for example, 
      splits_en/en_splits.csv). can be downloaded per language at 
      https://mlcommons.org/words for example, english is available at: 
      https://storage.googleapis.com/public-datasets-mswc/splits/en.tar.gz
    
    :param path_to_embeddings: path to directory of MSWC feature vectors stored
      as parquet files for a target language, can be downloaded from [TODO(MMAZ)]
    
    :param target_words: comma-delimited list of target words to classify, for
      example, "fifty,episode,restaurant,route,job"

    :param outdir: output directory to save the training and eval sets
    """

    outdir = Path(outdir)
    assert outdir.is_dir(), f"{outdir} is not a directory"
    train_embeddings = outdir / "train_embeddings"
    eval_embeddings = outdir / "eval_embeddings"
    # assert not train_embeddings.is_dir(), f"{train_embeddings} already exists"
    # assert not eval_embeddings.is_dir(), f"{eval_embeddings} already exists"

    assert path_to_metadata.endswith(
        ".json.gz"
    ), "path_to_metadata must end with .json.gz"
    with gzip.open(path_to_metadata, "r") as fh:
        meta = json.load(fh)

    target_words = set(target_words)
    print("generating training and eval splits for", target_words)

    # for each target, list file ids as .wavs (target/common_voice_id_12345.wav) by split
    splits_for_target = {
        target: dict(train=[], dev=[], test=[]) for target in target_words
    }
    with open(path_to_splits_csv, "r") as fh:
        reader = csv.reader(fh)
        next(reader)  # SET,LINK,WORD,VALID,SPEAKER,GENDER
        for row in tqdm.tqdm(reader):
            clip = row[1]  # aachen/common_voice_en_18833718.opus
            word = row[2]
            split = row[0].lower()
            if word in target_words:
                parquet_clip_id = str(Path(clip).parent / (Path(clip).stem + ".wav"))
                splits_for_target[word][split].append(parquet_clip_id)

    train_embeddings.mkdir(exist_ok=True)
    eval_embeddings.mkdir(exist_ok=True)

    # save target embeddings
    allowed_training_set = dict(targets={}, nontargets=[])
    eval_set = dict(targets={}, nontargets=[])
    for target, splits in splits_for_target.items():
        source_df = pd.read_parquet(Path(path_to_embeddings) / (target + ".parquet"))
        train_df = source_df[source_df["clip_id"].isin(splits["train"])]
        dev_df = source_df[source_df["clip_id"].isin(splits["dev"])]
        test_df = source_df[source_df["clip_id"].isin(splits["test"])]
        # combine dev_df and test_df for eval
        eval_df = pd.concat([dev_df, test_df])

        allowed_training_set["targets"][target] = splits["train"]
        eval_set["targets"][target] = splits["dev"] + splits["test"]
        train_df.to_parquet(train_embeddings / (target + ".parquet"), index=False)
        eval_df.to_parquet(eval_embeddings / (target + ".parquet"), index=False)

    # save parquets for nontarget samples
    nontarget_samples = select_nontarget_samples(meta, language_isocode, target_words)
    # training and eval (known nontargets)
    for (word, wavs) in tqdm.tqdm(nontarget_samples["train_eval_nontarget"], desc="known nontargets"):
        source_df = pd.read_parquet(Path(path_to_embeddings) / (word + ".parquet"))
        train_eval_df = source_df[source_df["clip_id"].isin(wavs)]

        n_rows = train_eval_df.shape[0]
        train_df = train_eval_df.iloc[: n_rows // 2]
        eval_df = train_eval_df.iloc[n_rows // 2 :]

        allowed_training_set["nontargets"].extend(train_df.clip_id)
        eval_set["nontargets"].extend(eval_df.clip_id)

        train_df.to_parquet(train_embeddings / (word + ".parquet"), index=False)
        eval_df.to_parquet(eval_embeddings / (word + ".parquet"), index=False)

    # unknown nontargets (which the classifier has not seen before)
    for (word, wavs) in tqdm.tqdm(nontarget_samples["unknown_nontarget"], desc="unknown nontargets"):
        source_df = pd.read_parquet(Path(path_to_embeddings) / (word + ".parquet"))
        eval_df = source_df[source_df["clip_id"].isin(wavs)]
        eval_set["nontargets"].extend(eval_df.clip_id)

        eval_df.to_parquet(eval_embeddings / (word + ".parquet"), index=False)

    with open(outdir / "allowed_training_set.yaml", "w") as fh:
        yaml.dump(
            allowed_training_set, fh, default_flow_style=None, sort_keys=False,
        )
    with open(outdir / "eval.yaml", "w") as fh:
        yaml.dump(eval_set, fh, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    fire.Fire(main)
