# dataperf-speech-example
Example workflow for our data-centric speech benchmark

### Terminology

* Keyword spotting model (KWS model): Also referred to as a wakeword or hotword model, or a voice trigger detection model, this is a small ML speech model that is designed to recognize a small vocabulary of spoken words or phrases (e.g., Siri, Google Voice Assistant, Alexa)
* Target sample: An example 1-second audio clip of a keyword used to train or evaluate a keyword-spotting model
* Nontarget sample: 1-second audio clips of words which are outside of the KWS model's vocabulary, used to train or measure the model's ability to minimize false positive detections on non-keywords.
* MSWC dataset: the [Multilingual Spoken Words Corpus](https://mlcommons.org/words), a dataset of 340,000 spoken words in 50 languages.
* Embedding vector representation: An n-dimensional vector which provides a feature representation of an audio word. We have trained a large classifier on keywords in MSWC, and we provide a 1024-element feature vector by using the penultimate layer of the classifer. Other embeddings, such as [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) are also available **[TODO: we may provide a flag for users to select which embedding they wish to use for training and evaluation, or we may restrict to only one embedding - TBD]**

### Files
* Input to selection algorithm: `samples.pb` - a protocol buffer encoded file of target and nontarget keyword samples. For each sample we provide an embedding representation and the corresponding sample ID (i.e., the audio file name) from the MSWC dataset. Your training set selection algorithm will choose a subset of these embedding vectors which maximize a simple classifier's performance 
* Output from selection algorithm: `train.npz` - a numpy array containing a selected subset of embedding vectors used to train the classifier.
* Input to `eval.py`:
    * `train.npz`: the selected embedding vectors used to train a classifier
    * `eval.pb`: a protocol buffer encoded file of test samples which are distinct from the training samples in `samples.pb` - this is the dataset we use to compute the classifier's score

On the evaluation server, we will have distinct, hidden `samples.pb` and `eval.pb` files using different keywords in different languages, in order to calculate the official score for our leaderboard. **[TODO: provide link to scoring function]**

### Developing a custom training set selection algorithm

Edit the function `select()` in `selection/selection.py` to include your custom training set selection algorithm. 

If your code has additional dependencies, make sure to edit `requirements.txt` and/or the `Dockerfile` to include these.  Please make sure not to change the behavior of `selection/main.py` or the docker entrypoint (this is how we automate evaluation on the server).

You can run your selection algorithm locally (outside of docker) with the following command:

```
python -m selection.main --input_samples path/to/samples.pb --outdir=.
```

This will write out `train.npz` in your current directory (you can change this by specifying a different `--outdir`).

### Creating a submission

Once you have implemented your selection algorithm, build a new version of your submission container:

```
docker build -t dataperf-speech-submission:latest .
```

Test your submission container before submitting to the evaluation server. To do so, first create a working directory for loading `samples.pb`. This will also be the destination for the docker container to write out the `train.npz` array containing your selected embedding vectors (used to train the classifier in `eval.py` in the evaluation step).

```
mkdir workdir
cp ~/path/to/samples.pb workdir/
```

Then run your selection algorithm within the docker container:

```
docker run --rm  -u $(id -u):$(id -g) --network none -v $(pwd)/workdir:/workdir -it dataperf-speech-submission:latest --input_samples /workdir/samples.pb --outdir=/workdir
```

There are several flags to note:

* `-u $(id -u):$(id -g)`: These flags are used so that the selection numpy array (`train.npz`) is written to disk as the user instead of as root 
* `-v $(pwd)/workdir:/workdir`: this is a [mounted volume](https://docs.docker.com/storage/volumes/), specifying the working directory we use to read in `samples.pb` and write out `train.npz` - you can change this to point to another location, but if you change the mapped name (`/workdir`) be sure to also reflect this in the entrypoint arguments (`--input_samples /workdir/samples.pb --outdir=/workdir`)
* `--network none`: your submission docker container will not have network access during evaluation on the server. This is to prevent exposing our hidden evaluation keyword. 

Finally, test out the evaluation script on your selection algorithm's output (we will use the same `eval.py` script on the server, but with a different hidden `samples.pb` and `eval.pb` dataset)

```
python eval.py --eval_file=path/to/eval.pb --train_file=workdir/train.npz
```