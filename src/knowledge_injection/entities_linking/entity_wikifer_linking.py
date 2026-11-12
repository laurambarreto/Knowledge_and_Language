## Revelou-se inutilizável para o português, não extrainda mais nenhuma entidade após algumas tentativas de uso.
## Estima-se que demorasse 2h20min

import pandas as pd
import os
from dotenv import load_dotenv
import requests
import time
from tqdm import tqdm
import json

# --- Wikifier Configuration ---
load_dotenv()
WIKIFIER_API_KEY = os.getenv("WIKIFIER_API_KEY")
WIKIFIER_URL = "http://www.wikifier.org/annotate-article"

if not WIKIFIER_API_KEY:
    print("ERROR: API key not found. Check your .env file.")
else:
    print("API Key loaded successfully.")

def get_wikifier_annotations(text):
    if not WIKIFIER_API_KEY:
        return []
    
    # Original configuration
    payload = {
        'text': text,
        'lang': 'pt',
        'userKey': WIKIFIER_API_KEY,
        'pageRankSqThreshold': '0.0',
        'applyPageRankSqThreshold': 'true',
        'nTopDfValuesToIgnore': '200',
        'nWordsToIgnoreFromList': '200',
        'wikiDataClasses': 'true',
        'wikiDataClassIds': 'false',
        'support': 'false',
        'ranges': 'true',
        'minLinkFrequency': '1',
        'includeCosines': 'true',
        'maxMentionEntropy': '3'
    }
    
    try:
        response = requests.post(WIKIFIER_URL, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("annotations", [])
    except Exception as e:
        # Print error but continue processing
        print(f"Error connecting to API: {e}")
        return []

def process_dataset(input_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    results = []
    print("Starting extraction and linking...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Clean the sentence
        sentence = str(row['sentences']).replace('"""', '').strip()
        classe = row['classe']
        
        # Get annotations from Wikifier
        wikifier_annots = get_wikifier_annotations(sentence)
        
        entities_found = []
        for ann in wikifier_annots:
            qid = ann.get('wikiDataItemId')
            title = ann.get('title')
            ranges = ann.get('ranges', [])
            
            if qid and ranges:
                entities_found.append({
                    'word': title,
                    'qid': qid,
                    'start_char': ranges[0][0],
                    'end_char': ranges[0][1]
                })
        
        # Print every 50 sentences to show progress and confirm extraction
        if len(results) % 2 == 0:
            print(f"\n--- Sentence {len(results)} ---")
            print(f"Text: {sentence[:80]}...")
            if len(entities_found) > 0:
                for ent in entities_found:
                    print(f"  -> Found: {ent['word']} (QID: {ent['qid']})")
            else:
                print("  -> No entities found for this sentence.")
                
        results.append({
            'id_sente': row['id_sente'],
            'classe': classe,
            'sentence': sentence,
            'entities': entities_found 
        })
        
        # Save every 100 sentences
        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            
        # Sleep to avoid rate limits
        time.sleep(0.1) 
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nAll results successfully saved to: {output_path}")
    return df_results

def show_statistics(df_results):
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    df_results['num_entities'] = df_results['entities'].apply(len)
    total = len(df_results)
    with_ents = (df_results['num_entities'] > 0).sum()
    
    print(f"Total sentences processed: {total}")
    print(f"Sentences with at least 1 entity: {with_ents} ({(with_ents/total)*100:.2f}%)")
    
    print("\nMean and Standard Deviation of entities per class:")
    print(df_results.groupby('classe')['num_entities'].agg(['mean', 'std', 'count']))
    
    # Top 15 entities
    all_words = []
    for ent_list in df_results['entities']:
        for e in ent_list:
            all_words.append(e['word'])
            
    if all_words:
        counts = pd.Series(all_words).value_counts().head(15)
        print("\nTop 15 Most Frequent Entities:")
        for word, count in counts.items():
            print(f"   {word}: {count}")
    else:
        print("\nNo entities were found in the entire dataset.")

    # 15 Examples
    print("\n15 Examples of Sentences with Detected Entities:")
    examples = df_results[df_results['num_entities'] > 0].head(15)
    if not examples.empty:
        for _, row in examples.iterrows():
            print(f"\nSentence (Class {row['classe']}): {row['sentence'][:100]}...")
            print("  Detected entities:")
            for ent in row['entities']:
                print(f"    - {ent['word']} (QID: {ent['qid']})")
    else:
        print("No examples available because no entities were extracted.")

if __name__ == "__main__":
    INPUT_CSV = "factnews.csv"
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_wikifier.csv"
    
    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)