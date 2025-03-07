# Run evaluation on GPT-2 Output Dataset
REAL_DATASET='webtext'
FAKE_DATASET='large-762M' # Choose from 'small-117M', 'large-762M', 'xl-1542M'
python old_detector/eval.py --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data --real-dataset $REAL_DATASET --fake-dataset $FAKE_DATASET --batch-size 1