FROM tensorflow/tensorflow:2.7.1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt


COPY selection/ ./selection/

ENTRYPOINT ["python", "-m", "selection.main"]