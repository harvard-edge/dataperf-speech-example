FROM tensorflow/tensorflow:2.7.1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY selection/ ./selection
COPY dataperf_speech_config.yaml ./dataperf_speech_config.yaml
COPY allowed_training_set.yaml ./allowed_training_set.yaml

ENTRYPOINT ["python", "-m", "selection.main"]