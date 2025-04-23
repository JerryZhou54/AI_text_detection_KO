# Run training on GPT-2 Output Dataset
REAL_DATASET='webtext'
FAKE_DATASET='small-117M' # Choose from 'small-117M', 'large-762M', 'xl-1542M'
python old_detector/train.py \
    --max-epochs 1 \
    --batch-size 64 \
    --max-sequence-length 510 \
    --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data \
    --real-dataset $REAL_DATASET \
    --fake-dataset $FAKE_DATASET

# Train on TuringBench dataset
# python old_detector/train.py \
#     --max-epochs 1 \
#     --batch-size 64 \
#     --max-sequence-length 510 \
#     --data-dir /home/hice1/wzhou322/scratch/TuringBench/AA \
#     --learning-rate 3e-6 \
#     --ckpt YOUR_CKPT_DIR