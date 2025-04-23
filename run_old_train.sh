# Train on TuringBench dataset
python old_detector/train.py \
    --max-epochs 1 \
    --batch-size 64 \
    --max-sequence-length 510 \
    --data-dir /home/hice1/wzhou322/scratch/TuringBench/AA \
    --learning-rate 3e-6