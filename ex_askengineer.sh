#!/bin/bash
/user/HS500/cj00677/mycondor/miniconda3/envs/pytorch_new/bin/python  /user/HS500/cj00677/mycondor/llm-based-annotation.py  --model_name mistralai/Mistral-7B-Instruct-v0.2     --dataset_file /user/HS500/cj00677/mycondor/dataset/AskEngineers.json  --output_file /user/HS500/cj00677/mycondor/AskEngineers-annotated.jsonl  --task_name question_annotation
