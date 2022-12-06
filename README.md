# Dataperf-Selection-Speech Alpha
Dataperf-Selection-Speech is a benchmark that measures the performance of dataset selection algorithms. The model training component is frozen and participants can only improve the accuracy by selecting the best training set. The benchmark is intended to encompass the tasks of dataset cleaning and coreset selection for a keyword spotting application.

More specifically, you are given a classification training dataset and your goal is to produce an algorithm that selects a subset of *M* examples from this dataset (*M* is specified as the `train_set_size_limit` in **dataperf_speech_config.yaml**). Evaluation proceeds by subsequently training a fixed model (`sklearn.ensemble.VotingClassifier` with various constituent classifiers) on your chosen subset, and then scoring the model's predictions on fixed test data via the `sklearn.metrics.balanced_accuracy_score` metric.

Component Ownership Diagram:

![Simple workflow](https://docs.google.com/drawings/d/e/2PACX-1vSlVN0uRWKySxu2ghuRhori-YxnQG859kg7zxan9xKXwarb1lQkRw9qVlnsOGEDqeVImxIplBvPDe5O/pub?w=635&h=416)

## MLCube Workflow
Participants are encuraged to use the [MLCube](https://github.com/mlcommons/mlcube) workflow to simplify development on the users machine and increase reproducibility.

To run the baseline selection algorithm:

Create Python environment and install MLCube Docker runner:
```bash 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
```

Run download task (only required once):
```bash 
mlcube run --task=download -Pdocker.build_strategy=always
```

Run selection:
```bash 
mlcube run --task=select -Pdocker.build_strategy=always
```

Run offline evaluation:
```bash 
mlcube run --task=evaluate -Pdocker.build_strategy=always
```

MLCube is optional, you can download the data as described further below, you will need to download the [MSWC metadata](https://storage.googleapis.com/public-datasets-mswc/metadata.json.gz) and the [MSWC embeddings](https://drive.google.com/file/d/1Lj1l7-FxipKF6ZtVpy7nSRMEWMD2yfCd/view).

### Algorithm Development
To develop their own selection algorithm, participants should:
- Duplicate and rename `random_selection.py` in `selection/implementations`
- Modify `Select()` in your new implementation file to your selection algorithm
- Change `selection_algorithm_module` and `selection_algorithm_class` in `workspace/dataperf_speech_config.yaml` to match the name of your selection implementation
- Run and evaluate your selection algorithm with the MLCube commands above

### Submission
Once Beta participants are satisfied with their selection algorithm they should submit their `train.json` file to [DynaBench](https://dynabench.org/tasks/speech-selection).

## Files

* `train_vectors` : The directory that contains the embedding vectors that can be selected for training. The file structure follows the pattern `train_vectors/en/left.parquet`. Each parquet file contains a "clip_id" column and a "mswc_embedding_vector" column.

* `eval_vectors` : The directory that contains the embedding vectors that are used for evaluation. The structure is identical to `train_vectors`

* `allowed_train_set.yaml` : A file that specifies which sample IDs are valid training samples. The file contrains the following structure `{"targets": {"left":[list]}, "nontargets": [list]}`

* `eval.yaml` : The evaluation set for eval.py. It follows the same structure as `allowed_train_set.yaml`.
* `train.json` : The file produced by `selection:main` that specifies the training set for eval.py.  It follows the same structure as `allowed_train_set.yaml`

* `dataperf_speech_config.yaml` : This file contains the configuration for the dataperf-speech-example workflow.

#### Optional Files

* `mswc_vectors` : The unified directory of all embedding vectors. This directory can be used to generate new `train_vectors` and `eval_vectors` directories.

* `train_audio` : The directory of wav files that can optionally be used in the selection algorithm.


## Running Selection/Eval Directly

You can run the selection and eval files directly, without needing MLCube

If your code has additional dependencies, make sure to edit `requirements.txt` and/or the `Dockerfile` to include these.

You can run your selection algorithm locally (outside of docker/MLCube) with the following command:

```
python -m selection.main \
  --allowed_training_set workspace/allowed_training_set.yaml \
  --train_embeddings_dir workspace/train_embeddings/ \
  --outdir workspace/
```

This will write out `train.json` into the directory specified by `--outdir` (which can be the same `workspace/` directory).

To evaluate your training set run:

```
python eval.py \
  --eval_embeddings_dir workspace/eval_embeddings/ \
  --train_embeddings_dir workspace/train_embeddings/ \
  --allowed_training_set workspace/allowed_training_set.yaml \
  --eval_file workspace/eval.yaml \
  --train_file workspace/train.json \
  --config_file workspace/dataperf_speech_config.yaml

```


MSWC metadata is [available here](https://storage.googleapis.com/public-datasets-mswc/metadata.json.gz)

MSWC train/dev/test splits can be downloaded at <https://mlcommons.org/words>. For example, English splits are [available here](https://storage.googleapis.com/public-datasets-mswc/splits/en.tar.gz)

MSWC embeddings can be downloaded here: `https://drive.google.com/file/d/1Lj1l7-FxipKF6ZtVpy7nSRMEWMD2yfCd/view`


### Using .wav Files for Selection

To use the raw audio in selection in addition to the embedding vectors:

* Download [the .wav version of the MSWC dataset](TODO).
* Pass the MSWC audio directory to selection:main as the `audio_dir` argument.
* Access the raw audio of a sample in a selection implementation with the `['audio']` label

## Glossary

* Keyword spotting model (KWS model): Also referred to as a wakeword, hotword, or voice trigger detection model, this is a small ML speech model that is designed to recognize a small vocabulary of spoken words or phrases (e.g., Siri, Google Voice Assistant, Alexa)
* Target sample: An example 1-second audio clip of a keyword used to train or evaluate a keyword-spotting model
* Nontarget sample: 1-second audio clips of words which are outside of the KWS model's vocabulary, used to train or measure the model's ability to minimize false positive detections on non-keywords.
* MSWC dataset: the [Multilingual Spoken Words Corpus](https://mlcommons.org/words), a dataset of 340,000 spoken words in 50 languages.
* Embedding vector representation: An n-dimensional vector which provides a feature representation of an audio word. We have trained a large classifier on keywords in MSWC, and we provide a 1024-element feature vector by using the penultimate layer of the classifer. 
<!-- Other embeddings, such as [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) are also available **[TODO: we may provide a flag for users to select which embedding they wish to use for training and evaluation, or we may restrict to only one embedding - TBD]** -->
