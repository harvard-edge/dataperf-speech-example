# dataperf-speech-example
Example workflow for our data-centric speech benchmark

Edit the function `select()` in `selection/selection.py` to include your custom training set selection algorithm. 

Make sure not to change the behavior of `selection/main.py` (either the arguments used as input or output).

If your code has additional dependencies, make sure to edit `requirements.txt` or the `Dockerfile` to include these.

Once you have implemented your selection algorithm, build a new version of your submission container:

```
docker build -t dataperf-speech-example:latest .
```

Test your submission container before submitting to the evaluation server. To do so, first create a working directory for loading `train.pb` and writing out the `.npy` arrays (which will be used to train the logistic regression classifier in `eval.py` in the following step)

```
mkdir workdir
cp ~/path/to/train.pb workdir/
```

Then run your selection algorithm within the docker container [**TODO:** add an offline mode flag, and a flag for out_dir]

```
docker run --rm  -u $(id -u):$(id -g) -v $(pwd)/workdir:/workdir -it dataperf-speech-example --train_file /workdir/train.pb --out_dir /workdir/
```
