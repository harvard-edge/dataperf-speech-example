# dataperf-speech-example
Example workflow for our data-centric speech benchmark

Edit the function `select()` in `selection/selection.py` to include your custom training set selection algorithm. 


If your code has additional dependencies, make sure to edit `requirements.txt` or the `Dockerfile` to include these.  Please make sure not to change the behavior of `selection/main.py` or the docker entrypoint (this is how we automate evaluation).

Once you have implemented your selection algorithm, build a new version of your submission container:

```
docker build -t dataperf-speech-submission:latest .
```

Test your submission container before submitting to the evaluation server. To do so, first create a working directory for loading `samples.pb` and writing out the `train.npz` array containing your selected embedding vectors (which will be used to train the logistic regression classifier in `eval.py` in the following evaluation step).

```
mkdir workdir
cp ~/path/to/samples.pb workdir/
```

Then run your selection algorithm within the docker container [**TODO:** add an offline mode flag, and a flag for out_dir]

```
docker run --rm  -u $(id -u):$(id -g) --network none -v $(pwd)/workdir:/workdir -it dataperf-speech-example --input_samples /workdir/samples.pb --out_dir /workdir/
```

There are several flags to note:

* `-u $(id -u):$(id -g)` These flags are used so that the selection numpy array files are not written to disk as root
* `-v $(pwd)/workdir:/workdir` this is the working directory we use to read `samples.pb` and write out `train.npz`
* `--network none` - your submission docker container will not have network access during evaluation on the server. This is to prevent exposing our hidden evaluation keyword. On the evaluation server, we will have separate `samples.pb` and `eval.pb` files using different keywords in different languages, in order to calculate the official score for our leaderboard. **[TODO: provide link to scoring function]**