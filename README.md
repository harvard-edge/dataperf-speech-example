# Dataperf-Selection-Speech
Dataperf-Selection-Speech is a benchmark that measures the performance of dataset selection algorithms. The model training component is frozen and participants can only improve the accuracy by selecting the best training set. The benchmark is intended to encompass the tasks of dataset cleaning and coreset selection for a keyword spotting application.


Component Ownership Diagram:

![Simple workflow](https://docs.google.com/drawings/d/e/2PACX-1vSlVN0uRWKySxu2ghuRhori-YxnQG859kg7zxan9xKXwarb1lQkRw9qVlnsOGEDqeVImxIplBvPDe5O/pub?w=635&h=416)


## Evaluation Metric

You are given a training dataset for spoken word classification and your goal is to produce an algorithm that selects a subset of size *M* examples (a coreset) from this dataset\*. Evaluation proceeds by subsequently training a fixed model (`sklearn.ensemble.VotingClassifier` with various constituent classifiers) on your chosen subset, and then scoring the model's predictions on fixed test data via the `sklearn.metrics.f1_score` metric with `average = macro`.

\* *M* is user defined, but Dynabench will host two leaderboards per language with coreset size caps of 25 and 60.

For each language, the challenge includes two leaderboards on Dynabench (six leaderboards in total). Each leaderboard corresponds to a language and a fixed maximum number of training samples (your submission can specifiy fewer samples than the maximum coreset size).

The training dataset consists of embedding vectors produced by a [pretrained keyword spotting model](https://arxiv.org/abs/2104.01454) ([model checkpoint weights](https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz)) for five target words in each of three languages (English, Portuguese, and Indonesian) taken from the [Multilingual Spoken Words Corpus](https://mlcommons.org/words). The classifier also includes a `nontarget` category representing unknown words which are distinct from one of the five target words. To train and evaluate the classifier's ability to recognize nontarget words, we include a large set of embedding vectors drawn from each respective language. The total number of target and nontarget samples for each language is shown in the figure below:


![Sample Counts](https://docs.google.com/drawings/d/e/2PACX-1vTu7bLLDu9QcUdzNzUOnHo53sCxsaiFs5aCsi__q32PiZdG2BTz0ovdfTSFutgKiWL39mnlMw-orBhk/pub?w=921&h=376)

Solutions should be algorithmic in nature (i.e., they should not involve human-in-the-loop audio sample listening and selection). We warmly encourage open-source submissions. If a participant team does not wish to open-source their solution, we ask that they allow the DataPerf organization to independently verify their solution and approach to ensure it is within the challenge rules.

# Challenge 

The challenge is hosted on [dataperf.org](https://dataperf.org) and will run from March 30 2023 through May 26 2023. Participants can submit solutions to [DynaBench](https://dynabench.org/tasks/speech-selection).

## Changelog

In case bugs or concerns are found, we will include a description of any changes to the evaluation metric, datasets, or support code here. Participants can re-submit their solutions to a new round on DynaBench which will reflect these changes.

* April 26 2023: Evaluation dataset publicly released
* March 30 2023: Challenge launch

## Downloading The Required Files

```
python utils/download_data.py --output_path workspace/data
```

This will automatically download and extract the train and eval embeddings for English, Inodnesian, and Portuguese.

## Running Selection/Eval
Run and evaluate the baseline selection algorithm. The target language can be changed by modifying the `--language` argument (English: `en`, Indonesian: `id`, Portuguese: `pt`). The training set size can be changed by modifying the `--train_size` argument (in particular, for each language, you will run two iterations of your training set selection algorithm, one for each `--train_size` leaderboard - in other words, you will perform six coreset generations in total per submission to Dynabench).


Run selection:

```
python -m selection.main --language en --train_size 25
```

This will write out `en_25_train.json` into the directory specified by `--outdir` (default is the `workspace/` directory), where `25` refers to the maximum size of the coreset.

You can run evaluation locally on your training set, but **please note the following:**

## :exclamation:  Do not use evaluation data during your selection algorithm development and optimization

Please see the challenge rules on [dataperf.org](https://dataperf.org) for more details - in particular, we ask you not to optimize your result using any of the challenge evaluation data. Optimization (e.g., cross-validation) should be performed on the samples in `allowed_training_set.yaml` for each language, and solutions **should not** be optimized against any of the samples listed in `eval.yaml` for any of the languages.

Since this speech challenge is fully open, there is no hidden test set. A locally-computed evaluation score is unofficial, but should match the results on DynaBench, and included here solely to allow for double-checking of DynaBench-computed results only if necessary. Official evaluations will only be performed on DynaBench. The following command performs local (offline) evaluation:

```
python eval.py --language en --train_size 25
```

This will output the macro f1 score of a model trained on the selected training set, against the official evaluation samples. 

### Algorithm Development
To develop their own selection algorithm, participants should:
- Create a new `selection.py` algorithm in `selection/implementations` which subclasses [`TrainingSetSelection`](https://github.com/harvard-edge/dataperf-speech-example/blob/main/selection/selection.py#L16)
- Implement `select()` in your class to use your selection algorithm
- Change `selection_algorithm_module` and `selection_algorithm_class` in `workspace/dataperf_speech_config.yaml` to match the name of your selection implementation
- optionally, add experiment configs to `workspace/dataperf_speech_config.yaml` (this can be accessed via `self.config` in )
- Run your selection strategy and submit your results to DynaBench

### Submission
Once participants are satisfied with their selection algorithm they should submit their `{lang}_{size}_train.json` files to [DynaBench](https://dynabench.org/tasks/speech-selection).
A seperate file is required for each language and training set size conbination (6 total).

## Files
Each supported language has the following files:

* `train_vectors` : The directory that contains the embedding vectors that can be selected for training. The file structure follows the pattern `train_vectors/en/left.parquet`. Each parquet file contains a "clip_id" column and a "mswc_embedding_vector" column.

* `eval_vectors` : The directory that contains the embedding vectors that are used for evaluation. The structure is identical to `train_vectors`

* `allowed_train_set.yaml` : A file that specifies which sample IDs are valid training samples. The file contrains the following structure `{"targets": {"left":[list]}, "nontargets": [list]}`

* `eval.yaml` : The evaluation set for eval.py. It follows the same structure as `allowed_train_set.yaml`. Participants should never use this data for training set selection algorithm development.

* `{lang}_{size}_train.json` : The file produced by `selection:main` that specifies the language specific training set for eval.py.

All languages share the following files:
* `dataperf_speech_config.yaml` : This file contains the configuration for the dataperf-speech-example workflow. Participants can extend this configuration file as needed.

#### Optional Files

* `mswc_vectors` : The unified directory of all embedding vectors. This directory can be used to generate new `train_vectors` and `eval_vectors` directories.

* `train_audio` : The directory of wav files that can optionally be used in the selection algorithm.


### Using .wav Files for Selection

To use the raw audio in selection in addition to the embedding vectors:

* Download [the .wav version of the MSWC dataset](TODO).
* Pass the MSWC audio directory to selection:main as the `audio_dir` argument.
* Access the raw audio of a sample in a selection implementation with the `['audio']` label

## Optional MLCube Workflow

Participants may use the [MLCube](https://github.com/mlcommons/mlcube) workflow to simplify development on the users machine and increase reproducability. 

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

## Glossary

* Keyword spotting model (KWS model): Also referred to as a wakeword, hotword, or voice trigger detection model, this is a small ML speech model that is designed to recognize a small vocabulary of spoken words or phrases (e.g., Siri, Google Voice Assistant, Alexa)
* Target sample: An example 1-second audio clip of a keyword used to train or evaluate a keyword-spotting model
* Nontarget sample: 1-second audio clips of words which are outside of the KWS model's vocabulary, used to train or measure the model's ability to minimize false positive detections on non-keywords.
* MSWC dataset: the [Multilingual Spoken Words Corpus](https://mlcommons.org/words), a dataset of 340,000 spoken words in 50 languages.
* Embedding vector representation: An n-dimensional vector which provides a feature representation of an audio word. We have trained a large classifier on keywords in MSWC, and we provide a 1024-element feature vector by using the penultimate layer of the classifer. 
<!-- Other embeddings, such as [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) are also available **[TODO: we may provide a flag for users to select which embedding they wish to use for training and evaluation, or we may restrict to only one embedding - TBD]** -->
