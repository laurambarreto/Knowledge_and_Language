'''
==================================================
DATASET STATISTICS (spaCy + BERTimbau + Wikidata)
==================================================
Total de frases: 6191
Frases com pelo menos 1 entidade ligada: 4020 (64.93%)

Demorou 3 horas e 55 minutos

Média e Desvio Padrão de entidades por classe:
            mean       std  count
classe                           
-1      0.805176  1.136131   1391
 0      1.551391  1.658636   4242
 1      1.498208  1.508503    558

Top 15 Entidades Mais Frequentes:
   Lula: 338
   Bolsonaro: 266
   Brasil: 184
   PT: 164
   Jair Bolsonaro: 103
   PL: 85
   Luiz Inácio Lula da Silva: 84
   Senado: 76
   Câmara: 76
   STF: 70
   TSE: 68
   Renan: 59
   São Paulo: 57
   Moro: 51
   ONU: 45

15 Exemplos de Frases com Entidades:

Sentence (Class -1): "Não houve sobreviventes", disse Okala....
    - Okala (LOC) -> Q109791897 (Okalakom)

Sentence (Class 0): O avião explodiu e se incendiou, acrescentou o porta-voz da ONU em Kinshasa, Jean-Tobias Okala....
    - ONU (ORG) -> Q25188868 (Onu)
    - Kinshasa (LOC) -> Q645492 (N'Djili International Airport)

Sentence (Class 0): Todos morreram quando o avião, prejudicado pelo mau tempo, não conseguiu chegar à pista de aterrissa...
    - Bukavu (LOC) -> Q96057016 (Bukavu General Hospital)
'''

import pandas as pd
import os
import spacy
import requests
import time
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

# --- Setup ---
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Error: Model not found. Run '!python -m spacy download pt_core_news_lg' in a new cell.")
    exit()

# Load BERTimbau for semantic similarity
print("Loading BERTimbau for disambiguation...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased', device=device)

# --- Wikidata API Function ---
def get_wikidata_candidates(entity_text, lang="pt"):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": lang,
        "format": "json",
        "limit": 10
    }
    headers = {"User-Agent": "FactNews-NLP-Project/1.0 (contact: student@uc.pt)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        data = response.json()
        return data.get("search", [])
    except Exception:
        return []

# --- Disambiguation Function using BERT ---
def disambiguate_entity(sentence, candidates):
    if not candidates:
        return None, None
        
    # Encode the original sentence
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    
    best_qid = None
    best_label = None
    best_score = -1
    
    for cand in candidates:
        qid = cand.get("id")
        label = cand.get("label")
        desc = cand.get("description", "")
        
        # Create a text combining label and description for better context
        cand_text = f"{label}: {desc}" if desc else label
        
        # Encode the candidate description
        cand_embedding = model.encode(cand_text, convert_to_tensor=True)
        
        # Calculate cosine similarity
        score = util.pytorch_cos_sim(sentence_embedding, cand_embedding).item()
        
        if score > best_score:
            best_score = score
            best_qid = qid
            best_label = label
            
    return best_qid, best_label

# --- Main Processing Function ---
def process_dataset(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    results = []
    print("Extracting entities with spaCy and disambiguating with BERT...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sentence = str(row['sentences']).replace('"""', '').strip()
        classe = row['classe']
        
        doc = nlp(sentence)
        entities_found = []
        
        for ent in doc.ents:
            # 1. Get candidates from Wikidata (try PT, then EN)
            candidates = get_wikidata_candidates(ent.text, lang="pt")
            if not candidates:
                candidates = get_wikidata_candidates(ent.text, lang="en")
            
            # 2. Disambiguate using BERT
            if candidates:
                qid, wikidata_label = disambiguate_entity(sentence, candidates)
                
                if qid: 
                    entities_found.append({
                        'word': ent.text,
                        'spacy_label': ent.label_,
                        'qid': qid,
                        'wikidata_label': wikidata_label,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    })
            
            time.sleep(0.1) # Be polite to API
                
        # Print every 50 sentences
        if len(results) % 50 == 0:
            print(f"\n--- Sentence {len(results)} ---")
            print(f"Text: {sentence[:80]}...")
            if entities_found:
                for e in entities_found:
                    print(f"  -> {e['word']} ({e['spacy_label']}) -> {e['qid']} ({e['wikidata_label']})")
            else:
                print("  -> No entities linked.")
                
        results.append({
            'id_sente': row['id_sente'],
            'classe': classe,
            'sentence': sentence,
            'entities': entities_found 
        })
        
        # Save every 100 sentences
        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return df_results

# --- Statistics Function ---
def show_statistics(df_results):
    print("\n" + "="*50)
    print("DATASET STATISTICS (spaCy + BERTimbau + Wikidata)")
    print("="*50)
    
    df_results['num_entities'] = df_results['entities'].apply(len)
    total = len(df_results)
    with_ents = (df_results['num_entities'] > 0).sum()
    
    print(f"Total de frases: {total}")
    print(f"Frases com pelo menos 1 entidade ligada: {with_ents} ({(with_ents/total)*100:.2f}%)")
    
    print("\nMédia e Desvio Padrão de entidades por classe:")
    print(df_results.groupby('classe')['num_entities'].agg(['mean', 'std', 'count']))
    
    # Top 15 Entidades
    all_words = []
    for ent_list in df_results['entities']:
        for e in ent_list:
            all_words.append(e['word'])
            
    if all_words:
        counts = pd.Series(all_words).value_counts().head(15)
        print("\nTop 15 Entidades Mais Frequentes:")
        for word, count in counts.items():
            print(f"   {word}: {count}")
    else:
        print("\nNenhuma entidade encontrada no dataset.")

    # 15 Exemplos
    print("\n15 Exemplos de Frases com Entidades:")
    examples = df_results[df_results['num_entities'] > 0].head(15)
    if not examples.empty:
        for _, row in examples.iterrows():
            print(f"\nSentence (Class {row['classe']}): {row['sentence'][:100]}...")
            for ent in row['entities']:
                print(f"    - {ent['word']} ({ent['spacy_label']}) -> {ent['qid']} ({ent['wikidata_label']})")
    else:
        print("Nenhum exemplo disponível.")

if __name__ == "__main__":
    INPUT_CSV = "factnews.csv"
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_bert_disambiguated.csv"
    
    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)