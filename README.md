# dataperf-speech-example
Example workflow for our data-centric speech benchmark

### Terminology  TODO UPDATE

* Keyword spotting model (KWS model): Also referred to as a wakeword or hotword model, or a voice trigger detection model, this is a small ML speech model that is designed to recognize a small vocabulary of spoken words or phrases (e.g., Siri, Google Voice Assistant, Alexa)
* Target sample: An example 1-second audio clip of a keyword used to train or evaluate a keyword-spotting model
* Nontarget sample: 1-second audio clips of words which are outside of the KWS model's vocabulary, used to train or measure the model's ability to minimize false positive detections on non-keywords.
* MSWC dataset: the [Multilingual Spoken Words Corpus](https://mlcommons.org/words), a dataset of 340,000 spoken words in 50 languages.
* Embedding vector representation: An n-dimensional vector which provides a feature representation of an audio word. We have trained a large classifier on keywords in MSWC, and we provide a 1024-element feature vector by using the penultimate layer of the classifer. Other embeddings, such as [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) are also available **[TODO: we may provide a flag for users to select which embedding they wish to use for training and evaluation, or we may restrict to only one embedding - TBD]**

### Files
* `train_vectors` : 
* `eval_vectors` : 
* `allowed_train_set.yaml` : 
* `eval.yaml` : 
* `train.yaml` : 
* `dataperf_speech_config.yaml` : 

#### Optional Files
* `MSWC_vectors` : 
* `MSWC_audio` : 

On the evaluation server, we will have distinct, hidden versions of files using different keywords in different languages, in order to calculate the official score for our leaderboard. This ensures the submitted selection algoithm can generalize to other words and languages. We encurage participants to test out other target words to ensure their solution generalizes. **[TODO: provide link to scoring function]**

#### File Diagram
![File Diagram](https://docs.google.com/drawings/d/1h5uIaUZVWzO_bGhtsdwr7tA4J6xC5Anh2GFN2BBr2yA/edit?usp=sharing)

### Developing a custom training set selection algorithm

Edit the function `select()` in `selection/selection.py` to include your custom training set selection algorithm. 

If your code has additional dependencies, make sure to edit `requirements.txt` and/or the `Dockerfile` to include these.  Please make sure not to change the behavior of `selection/main.py` or the docker entrypoint (this is how we automate evaluation on the server).

You can run your selection algorithm locally (outside of docker) with the following command:

```
python -m selection.main --outdir=./workdir
```

This will write out `train.yaml` into the specified directory.

To evaluate your training set run:

```
python eval.py --eval_file=eval.yaml --train_file=workdir/train.yaml
```

### Creating a submission

Once you have implemented your selection algorithm, build a new version of your submission container:

```
docker build -t dataperf-speech-submission:latest .
```

Test your submission container before submitting to the evaluation server. To do so, first create a working directory the output of the selection script

```
mkdir workdir
```

Then run your selection algorithm within the docker container:

```
docker run --rm  -u $(id -u):$(id -g) --network none -v $(pwd)/workdir:/workdir -v $(pwd)/embeddings:/embeddings -it dataperf-speech-submission:latest
```

There are several flags to note:

* `-u $(id -u):$(id -g)`: These flags are used so that the selection yaml (`train.yaml`) is written to disk as the user instead of as root 
* `-v $(pwd)/workdir:/workdir -v $(pwd)/embeddings:/embeddings`: these are a [mounted volumes](https://docs.docker.com/storage/volumes/), specifying the working directory used for the train.yaml output and the directory of the training vectors dataset.
* `--network none`: your submission docker container will not have network access during evaluation on the server. This is to prevent exposing our hidden evaluation keyword. 

Finally, test out the evaluation script on your selection algorithm's output (we will use the same `eval.py` script on the server, but with a different hidden `samples.pb` and `eval.pb` dataset)

```
python eval.py --eval_file=eval.yaml --train_file=workdir/train.yaml
```

#### Using .wav Files for Selection
**[TODO]**

### Submitting to the evaluation server

**[TODO]**