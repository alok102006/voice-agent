import pandas as pd
from preprocessing.text_cleaning import clean_text, extract_key_sentences
from preprocessing.label_mapping import map_category


INPUT_PATH = "data/raw/australian_legal_cases.csv"
OUTPUT_PATH = "data/processed/legal_issues.csv"


def preprocess_dataset():
    df = pd.read_csv(INPUT_PATH)

    processed_rows = []

    for text in df['case_text'].dropna():
        reduced_text = extract_key_sentences(text)
        cleaned_text = clean_text(reduced_text)
        label = map_category(cleaned_text)

        if label:
            processed_rows.append({
                'text': cleaned_text,
                'label': label
            })

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[✓] Preprocessing complete. Saved {len(processed_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess_dataset()
