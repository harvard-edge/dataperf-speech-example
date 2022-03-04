#!/bin/bash

python3 eval.py ${eval_path}/eval.pb ${eval_path}/samples.pb --train_file=${train_path}/train.yaml 2>&1 | tee tee ${log_path}