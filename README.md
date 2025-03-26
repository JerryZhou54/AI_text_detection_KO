# AI_text_detection_KO

Check out `old_detector` to see more details.
The current best RoBERTa AI text detector checkpoint (trained on both GPT2 output dataset and TuringBench dataset) can be found [here](https://drive.google.com/file/d/1nhLOxHZhNOoFhVy06icKT4mjws8a8ODC/view?usp=sharing)

## Progress made for Progress Report 2
- Add `prepare_model_and_data.sh` script to download our current AI text classifier trained on texts generated by human in WebText and by varying configurations of GPT2.
- Add `run_old_eval.sh` script to automatically measure the accuracy of our classifier on the test set of the GPT2 Output Dataset.
- Add `old_detector/eval.py` to separate the training logic and evaluation logic.
- Add the `TuringBenchDataset` class in `old_detector/dataset.py` which prepares the data from TuringBench dataset for future training.
