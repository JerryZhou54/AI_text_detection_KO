import pandas as pd
import openai
import time
from tqdm import tqdm

# ====== Configuration ======
INPUT_CSV = '/home/hice1/wzhou322/scratch/gpt2-output-data/small-117M-k40.train.csv'
OUTPUT_CSV = 'paraphrased_texts.csv'
TEXT_COLUMN = 'text'
API_MODEL = 'gpt-4o'
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 1  # seconds
openai.api_key = 'YOUR_API_KEY'  # Replace this with your API key or load via env variable

# ====== GPT-4o Paraphrasing Function ======
def paraphrase(text):
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
                    {"role": "user", "content": f"Paraphrase this text: {text}"}
                ],
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(2)
    return text  # fallback to original if API fails

# ====== Main Workflow ======
def main():
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"'{TEXT_COLUMN}' column not found in input CSV.")

    paraphrased_texts = []
    for text in tqdm(df[TEXT_COLUMN], desc="Paraphrasing"):
        paraphrased = paraphrase(text)
        paraphrased_texts.append(paraphrased)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    df['paraphrased'] = paraphrased_texts
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved paraphrased texts to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
