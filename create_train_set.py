import yaml
import pandas as pd
from pathlib import Path
import random


embedding_dir = "embeddings/"
target_lang = 'en'
target_words = ["left", "right", "down", "yes", "off"]
num_per_target = [400] * len(target_words)
num_nontarget = 2000
num_per_word_nontarget = 10
random_seed = 2
random.seed(random_seed)

source_dir = Path(embedding_dir) / target_lang
target_files = [ source_dir / Path(word + ".parquet") for word in target_words]

all_files = source_dir.glob("**/*.parquet")
nontarget_files = list(set(all_files).difference(target_files))

targets = {}
nontargets = []

for i, file in enumerate(target_files):
    target_df = pd.read_parquet(file)
    sampled_df = target_df["clip_id"].sample(n=num_per_target[i], random_state=random_seed)
    targets[file.stem] = sampled_df.tolist()


while len(nontargets)<num_nontarget:
    file = random.choice(nontarget_files)
    nontarget_df = df = pd.read_parquet(file)
    if (nontarget_df.shape[0] < (num_per_word_nontarget*10)):
        continue
    sampled_df = nontarget_df["clip_id"].sample(n=num_per_word_nontarget, random_state=random_seed)
    nontargets.extend(sampled_df.tolist())
    

with open("allowed_training_set.yaml", "w") as fh:
        yaml.dump(
            dict(targets=targets, nontargets=nontargets),
            fh,
            default_flow_style=None,
            sort_keys=False,
        )