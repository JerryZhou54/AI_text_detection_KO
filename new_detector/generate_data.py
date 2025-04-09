import pandas as pd
import openai
from openai import OpenAI
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context
import os

# ====== Configuration ======
INPUT_CSV = '/home/hice1/wzhou322/scratch/gpt2-output-data/webtext.valid.csv'
OUTPUT_CSV = '/home/hice1/wzhou322/scratch/gpt2-output-data/gpt-4o-mini.webtext.valid.csv'
TEXT_COLUMN = 'text'
API_MODEL = 'gpt-4o-mini'
MAX_RETRIES = 2
DELAY_BETWEEN_REQUESTS = 0  # seconds
FLUSH_EVERY = 50
NUM_WORKERS = 16
OPENAI_API_KEY = ''
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

# ====== GPT-4o Paraphrasing Function ======
def paraphrase_single(text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=API_MODEL,
                instructions="You are a helpful assistant that paraphrases text",
                input=f"Paraphrase this text: {text}",
                temperature=0.7
            )
            return response.output_text
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(2)
    return ""

# ====== Main Workflow ======
def main():
    df = pd.read_csv(INPUT_CSV)

    start_index = 0
    paraphrased_texts = []
    # Resume support
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        paraphrased_texts = existing[TEXT_COLUMN].tolist()
        start_index = len(paraphrased_texts)
        print(f"🔄 Resuming from index {start_index}")

    to_process = df[TEXT_COLUMN].iloc[start_index:].tolist()

    # Multiprocessing pool with safe context for Jupyter/macOS
    with get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        chunk = []
        for i, paraphrased in enumerate(tqdm(pool.imap(paraphrase_single, to_process), total=len(to_process)), start=start_index):
            paraphrased_texts.append(paraphrased)
            chunk.append(paraphrased)

            if (i + 1) % FLUSH_EVERY == 0 or (i + 1) == len(df):
                pd.DataFrame({TEXT_COLUMN: paraphrased_texts}).to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Flushed {len(paraphrased_texts)} rows to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
