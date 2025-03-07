# Download trained roberta-based classifier
git lfs install
git clone https://huggingface.co/weizhou03/roberta-old-AI-detector

GPT2_DATA_DIR= # path to gpt2-output-dataset
TB_DATA_DIR= # path to TuringBench dataset
# Download GPT-2 Output Dataset
#!/bin/bash
curl -L -o DATA_DIR/gpt2-output-data.zip \
  https://www.kaggle.com/api/v1/datasets/download/abhishek/gpt2-output-data

unzip DATA_DIR/gpt2-output-data.zip -d $GPT2_DATA_DIR

# Download TuringBench Dataset
wget https://huggingface.co/datasets/turingbench/TuringBench/resolve/main/TuringBench.zip

unzip TuringBench.zip -d $TB_DATA_DIR