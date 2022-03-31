from pathlib import Path
import fire
import yaml
from selection.selection import TrainingSetSelection
import pandas as pd
import numpy as np

label_idx = 0 #IDs look like "en/clips/and/common_voice..." this is the index of the label


def main(
            train_embeddings_dir="embeddings/en", 
            audio_dir=None,
            allowed_training_set="config_files/allowed_training_set.yaml", 
            config_file="config_files/dataperf_speech_config.yaml",
            outdir="/workdir"
):
    with open(config_file, "r") as fh:
        config = yaml.safe_load(fh)
    train_set_size_limit = config["train_set_size_limit"]

    embedding_dataset = Path(train_embeddings_dir)

    with open(allowed_training_set, "r") as fh:
        allowed_training_embeddings = yaml.safe_load(fh) #dict {"targets": {"dog":[list]}, "nontargets": [list]}


    train_embeddings = {'targets':{}, 'non_targets':[]} # {"targets": {"dog":[{'ID':string,'feature_vector':np.array,'audio':np.array}, ...], ...}, "nontargets": [list]}
    for target, id_list in allowed_training_embeddings['targets'].items():
        train_embeddings['targets'][target] = []
        target_parquet = pd.read_parquet(embedding_dataset / (target + ".parquet"))
        allowed_ids_mask = target_parquet["clip_id"].isin(id_list)
        for row in target_parquet[allowed_ids_mask].itertuples(): 
            train_embeddings['targets'][target].append({'ID':row.clip_id, 'feature_vector':row.mswc_embedding_vector})

    for id in allowed_training_embeddings["nontargets"]:
        label = Path(id).parts[label_idx]
        parquet_file = pd.read_parquet(embedding_dataset / (label + ".parquet"))
        row = parquet_file.loc[parquet_file['clip_id'] == id].iloc[0]
        train_embeddings['non_targets'].append({'ID':row['clip_id'], 'feature_vector':row['mswc_embedding_vector']})
    
    audio_flag = False
    if audio_dir is not None:
        #TODO Test with wav files
        from scipy.io import wavfile
        audio_flag = True
        for target, sample_list in train_embeddings['targets'].items():
            for sample in sample_list:
                _, audio = wavfile.read(audio_dir + '/' + sample['ID'])
                sample['audio'] = audio
        for sample in train_embeddings['non_targets']:
            _, audio = wavfile.read(audio_dir + '/' + sample['ID'])
            sample['audio'] = audio



    selection = TrainingSetSelection(
        train_embeddings = train_embeddings,
        train_set_size = train_set_size_limit,
        audio_flag = audio_flag
    )

    train = selection.select()

    assert Path(
        outdir
    ).is_dir(), (
        f"{outdir} does not exist, please specify --outdir as a command line argument"
    )
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
