# Run evaluation on GPT4o-mini output texts
FAKE_DATASET='gpt-4o-mini.webtext'
python old_detector/eval.py \
    --data-dir /home/hice1/wzhou322/scratch/gpt2-output-data \
    --fake-dataset $FAKE_DATASET \
    --batch-size 1 \
    --ckpt /home/hice1/wzhou322/scratch/AI_text_detection_KO/logs/best-model.pt