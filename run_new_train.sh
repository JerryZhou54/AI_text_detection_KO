# Design 1 Training
# python old_detector/train.py \
#     --max-epochs 1 \
#     --batch-size 64 \
#     --max-sequence-length 510 \
#     --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data \
#     --fake-dataset gpt-4o-mini.webtext \
#     --learning-rate 3e-6 \
#     --weight-decay 1

# Design 2 Training
python old_detector/train.py \
    --max-epochs 1 \
    --batch-size 64 \
    --max-sequence-length 510 \
    --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data \
    --fake-dataset gpt-4o-mini.webtext \
    --learning-rate 5e-6 \
    --freeze_params

# Design 3 Training
# python old_detector/train.py \
#     --max-epochs 1 \
#     --batch-size 64 \
#     --max-sequence-length 510 \
#     --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data \
#     --real-dataset webtext \
#     --fake-dataset gpt-4o-mini.webtext \
#     --learning-rate 5e-6

# Design 4 Training
# python old_detector/train.py \
#     --max-epochs 1 \
#     --batch-size 64 \
#     --max-sequence-length 510 \
#     --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data \
#     --real-dataset webtext \
#     --fake-dataset gpt-4o-mini.webtext \
#     --extra-fake-dataset small-117M \
#     --learning-rate 5e-6