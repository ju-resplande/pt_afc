FROM huggingface/transformers-pytorch-gpu:4.35.2

RUN apt update -y
RUN apt install -y nano python3-setuptools 
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install -U simpletransformers wandb pandas python-dotenv loguru
