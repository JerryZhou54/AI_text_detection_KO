# AI_text_detection_KO

## Dataset for Old LLMs

Our dataset of old LLMs is comprised of GPT2 output dataset and the TuringBench dataset. Run the following command to install these two datasets into your local dir:
```
bash prepare_model_and_data.sh
```

## Dataset for New LLMs

The synthetic dataset used in the final report experiments can be found [here](https://drive.google.com/file/d/1lT_kt4UYOWYfIlDwppAo-LT51GAIt-u7/view?usp=sharing)

Run the following command to create your own synthetic dataset by paraphrasing each entry in the human-written set (in the GPT2 output dataset) using GPT4o-mini:
```
python new_detector/generate_data.py
```

It has two features:
1. The script will send multiple paraphrasing requests simultaneously the OpenAI server and will write to a common output csv. This can greatly shorten the data generation time.
2. It will write the generated texts periodically to the output csv to avoid impacts from unexpected crash! You can also pick up where you left off!

## Baseline Classifier trained on old LLMs

The current best RoBERTa AI text detector checkpoint (trained on both GPT2 output dataset and TuringBench dataset) can be found [here](https://drive.google.com/file/d/1nhLOxHZhNOoFhVy06icKT4mjws8a8ODC/view?usp=sharing)

To train the RoBERTa AI text detector on old LLMs, use the following script:
```
run_old_train.sh
```
After training, it will produce a tensorboard log, a best model checkpoint and two .pt files recording the train loss and train accuracy trajectory.

If you only want to run evaluation on old LLMs using the best checkpoint above, you can run:
```
bash run_old_eval.sh # Run eval on GPT2 output dataset
bash run_TB_eval.sh # Run eval on TuringBench dataset
```

## Fine-tune Classifier on new LLMs

The model checkpoint fine-tuned under design 1, 2, 3, 4 can be found [here](https://drive.google.com/file/d/18h8lTYAYI7tS0XCWOxRzaX78KUxfDDA9/view?usp=sharing)

To reproduce the experiments for Design 1, 2, 3, 4 mentioned in section 6.3: Mitigation Strategies Experiments in the final report, run the following commands:
```
bash run_new_train.sh # The config for all the designs is stored here. Make sure to uncomment and comment.
```

After fine-tuning, run the evaluation using:
```
bash run_old_eval.sh # Run evaluation on the GPT2 output dataset
bash run_synthetic_eval.sh # Run evaluation on the synthetic dataset
```
Make sure to set the correct directory path to your checkpoint in the scripts.

## File Structure
```
new_detector/generate_data.py # Generate your own synthetic dataset
old_detector/dataset.py # Contains data preprocessing logic and torch Dataset construction for GPT2 output dataset, TuringBench dataset and synthetic dataset
old_detector/eval.py # Evaluation Logic
old_detector/train.py # Train Logic
run_new_train.sh # Contains train command to kickstart the fine-tuning process of the pretrained RoBERTa detector under four designs.
run_old_train.sh # Contains train command to kickstart the baseline training process of the RoBERTa detector on GPT2 output dataset and TuringBench dataset.
run_old_eval.sh # Contains eval command to kickstart the evaluation on GPT2 output dataset.
run_synthetic_eval.sh # Contains eval command to kickstart the evaluation on the synthetic dataset.
run_TB_eval.sh # Contains eval command to kickstart the evaluation on the TuringBench dataset.
```
