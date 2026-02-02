# CLARITY-Unmasking-Political-Question-Evasions

This project aims to develop a multi-task classification system for analyzing question-and-answer pairs in political discourse, evaluating them across two dimensions:

1. Clarity Level Classification - Classifying answers into 3 classes:
Clear Reply/Ambiguous/Clear No Reply

2. Avoidance Level Classification - Identifying 9 different avoidance techniques


Evaluation Metrics: Macro F1 score and accuracy


Dataset
source: https://huggingface.co/datasets/ailsntua/QEvasion

Training Set: 3,450 labeled question-and-answer pairs
Test Set: 308 question-and-answer pairs


To quick start:

run bash:
pip install -r requirements.txt
python train.py

The default model is microsoft/deberta-v3-base
You can change model configurations through model_config.py


For reference, we also fine-tuned a decoder-only model, llama.
run bash:
pip install -r requirements_llama.txt
python llama_fine_tuning.py


Author: Mufan Zhang, Yuan Qiu, Zichen Yin
