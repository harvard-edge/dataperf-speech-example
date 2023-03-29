import subprocess
import tqdm
from pathlib import Path
import json
import numpy as np

xfold = 20
results_file = Path.home() / "sensitivity_analysis2" /f"testing_xfold{xfold}.json"
assert not results_file.exists()
results_json = {}
unsatisfiable_log = Path.home() / "sensitivity_analysis2" / f"unsatisfiable_xfold{xfold}.txt"
assert not unsatisfiable_log.exists()

# generate experiment params

def run_experiment(config):
    global results_json # real cool :(

    # unpack config
    lang = config["lang"]
    xfold = config["xfold"]
    target_frac = config["target_frac"]
    set_size = config["set_size"]
    experiment_name = f"lang_{lang}_xfold_{xfold}_tf_{target_frac}_set_size_{set_size}"
    print(f"\n\nRunning experiment :::::::::::::::::::::::::::::::::::::::")
    print(f"Experiment name: {experiment_name}")
    outdir = Path.home() / "sensitivity_analysis2" /  f"sensitivity_testing_xfold{xfold}_tf_{target_frac}" / lang / f"set_size_{set_size:03d}"
    outdir.mkdir(parents=True, exist_ok=False)
    sel_cmd = f"cd dataperf-speech-example && python -m selection.main --language {lang} --allowed_training_set /home/mark/dataperf_datasets/dataperf_{lang}_data/allowed_training_set.yaml --train_embeddings_dir /home/mark/dataperf_datasets/dataperf_{lang}_data/train_embeddings/ --outdir {outdir} --train_set_size_limit {set_size} --target_frac {target_frac} --xfold {xfold}"
    subprocess.run(sel_cmd, shell=True)

    # if the train.json file is not present, the experiment was unsatisfiable and we should early-exit
    if not (outdir / f"{lang}_train.json").exists():
        print(f"Skipping experiment {experiment_name} because it was unsatisfiable")
        # append experiment to unsatisfiable log
        with unsatisfiable_log.open("a") as f:
            f.write(experiment_name + "\n")
        print("-------------------")
        # remove outdir if empty
        if not list(outdir.glob("*")):
            outdir.rmdir()
        return

    eval_cmd = f"cd dataperf-speech-example && python eval.py --language {lang} --eval_embeddings_dir /home/mark/dataperf_datasets/dataperf_{lang}_data/eval_embeddings --allowed_training_set /home/mark/dataperf_datasets/dataperf_{lang}_data/allowed_training_set.yaml --train_embeddings_dir /home/mark/dataperf_datasets/dataperf_{lang}_data/train_embeddings/ --eval_file /home/mark/dataperf_datasets/dataperf_{lang}_data/eval.yaml --train_file {outdir}/{lang}_train.json --config_file workspace/dataperf_speech_config.yaml --train_set_size_limit {set_size}"
    eval_results = subprocess.check_output(eval_cmd, shell=True).decode("utf-8")
    print("-------------------")
    print(eval_results)
    results = outdir / f"results_{experiment_name}.txt"
    results.write_text(eval_results)
    results_json[experiment_name] = dict(desc=config, results=eval_results)
    # overwrites after each experiment
    results_file.write_text(json.dumps(results_json, indent=4, sort_keys=True))

exp_configs = []
for lang in ["id", "en", "pt"]:
    for set_size in range(9, 61, 3):
        for target_frac in np.arange(0.1, 1.0, 0.175):
            target_frac = round(target_frac, 5)
            exp_config = dict(lang=lang, xfold=xfold, target_frac=target_frac, set_size=set_size)
            exp_configs.append(exp_config)

print(len(exp_configs))
np.random.default_rng(0).shuffle(exp_configs)
for exp_config in tqdm.tqdm(exp_configs):
    run_experiment(exp_config)

results_file.write_text(json.dumps(results_json, indent=4, sort_keys=True))
print("Done")