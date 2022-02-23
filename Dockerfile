FROM tensorflow/tensorflow:2.7.1

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app/
WORKDIR /app/

ENTRYPOINT ["python", "-m", "selection.main"]