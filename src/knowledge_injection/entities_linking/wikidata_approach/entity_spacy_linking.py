''' 
Revelou-se razoável com 75/80% de entidades corretamente extraídas (deep seek) e 31% de frases com entidades
Demorou 26 minutos para processar 6191 sentenças (0.25s por frase)

O código falhou redondamente em (Siglas e Acrónimos, Nomes Ambíguos, Entidades Políticas e Geográficas Confundidas):
Lula -> Teuthida (a lula animal)
PCC (Primeiro Comando da Capital) -> Chinese Communist Party (Partido Comunista Chinês)
PT (Partido dos Trabalhadores) -> Portugal
Globo (TV Globo) -> Earth (a Terra)
Santo André (cidade no estado de São Paulo) -> Andrew the Apostle (o apóstolo André)
MP (Ministério Público) -> Northern Mariana Islands
CET (Companhia de Engenharia de Tráfego) -> UTC+01:00
Pinho -> pine (árvore). Devia ser o sobrenome da pessoa
Mantega -> butter (manteiga). Devia ser o ministro Guido Mantega.

==================================================
DATASET STATISTICS (spaCy + Wikidata)
==================================================
Total sentences: 6191
Sentences with at least 1 entity linked: 1973 (31.87%)

Mean and Std Dev of entities per class:
            mean       std  count
classe                           
-1      0.404745  0.856263   1391
 0      0.723951  1.398263   4242
 1      0.732975  1.273408    558

Top 15 Most Frequent Entities: -> Destas 15, 10 estão corretas
   Lula: 162                         --> mal extraída
   Bolsonaro: 128
   Brasil: 98
   PT: 90                            --> mal extraída
   Câmara: 50                        --> mal extraída
   Jair Bolsonaro: 45
   Luiz Inácio Lula da Silva: 39
   PL: 37                            --> mal extraída
   São Paulo: 32
   TSE: 32
   Senado: 30
   STF: 29
   ONU: 28
   TAM: 23                            --> mal extraída
   Brasília: 23

3 Examples of Sentences with Detected Entities:

Sentence (Class -1): "Não houve sobreviventes", disse Okala....
    - Okala (LOC) -> Q37995 (Yangon)

Sentence (Class 0): O avião explodiu e se incendiou, acrescentou o porta-voz da ONU em Kinshasa, Jean-Tobias Okala....
    - ONU (ORG) -> Q1065 (United Nations)
    - Kinshasa (LOC) -> Q3838 (Kinshasa)

Sentence (Class 0): Todos morreram quando o avião, prejudicado pelo mau tempo, não conseguiu chegar à pista de aterrissa...
    - Bukavu (LOC) -> Q222279 (Bukavu)


'''

import pandas as pd
import os
import spacy
import requests
import time
from tqdm import tqdm

# --- Setup ---
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Error: Model not found. Run '!python -m spacy download pt_core_news_lg' in a new cell.")
    exit()

# --- Wikidata API Function ---
def get_wikidata_qid(entity_text, spacy_label):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": "pt",
        "format": "json",
        "limit": 5
    }
    
    # Wikidata requires a User-Agent to avoid blocking silent requests
    headers = {
        "User-Agent": "FactNews-NLP-Project/1.0 (contact: student@uc.pt)"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        results = data.get("search", [])
        if not results:
            return None, None
            
        # Basic heuristic to avoid obvious mismatches
        for res in results:
            desc = res.get("description", "").lower()
            qid = res.get("id")
            
            if spacy_label == "PER" and any(w in desc for w in ["político", "pessoa", "ator", "atriz", "jornalista", "jogador", "escritor"]):
                return qid, res.get("label")
            if spacy_label == "LOC" and any(w in desc for w in ["país", "cidade", "capital", "estado", "região", "rio", "localidade", "município"]):
                return qid, res.get("label")
            if spacy_label == "ORG" and any(w in desc for w in ["organização", "empresa", "partido", "clube", "instituição", "associação", "companhia"]):
                return qid, res.get("label")
                
        # Fallback: if heuristic fails, just take the top candidate
        top = results[0]
        return top.get("id"), top.get("label")
        
    except Exception:
        return None, None

# --- Main Processing Function ---
def process_dataset(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    results = []
    print("Extracting entities with spaCy and querying Wikidata...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sentence = str(row['sentences']).replace('"""', '').strip()
        classe = row['classe']
        
        # 1. Extract entities using spaCy
        doc = nlp(sentence)
        
        entities_found = []
        for ent in doc.ents:
            # 2. Link to Wikidata
            qid, wikidata_label = get_wikidata_qid(ent.text, ent.label_)
            
            if qid: # Only save if we got a valid QID
                entities_found.append({
                    'word': ent.text,
                    'spacy_label': ent.label_,
                    'qid': qid,
                    'wikidata_label': wikidata_label,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char
                })
                time.sleep(0.1) # Be polite to Wikidata API
                
        # Print every 50 sentences to monitor progress
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
    print("DATASET STATISTICS (spaCy + Wikidata)")
    print("="*50)
    
    df_results['num_entities'] = df_results['entities'].apply(len)
    total = len(df_results)
    with_ents = (df_results['num_entities'] > 0).sum()
    
    print(f"Total sentences: {total}")
    print(f"Sentences with at least 1 entity linked: {with_ents} ({(with_ents/total)*100:.2f}%)")
    
    print("\nMean and Std Dev of entities per class:")
    print(df_results.groupby('classe')['num_entities'].agg(['mean', 'std', 'count']))
    
    # Top 15 Entities
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
        print("\nNo entities were found in the dataset.")

    # 15 Examples
    print("\n15 Examples of Sentences with Detected Entities:")
    examples = df_results[df_results['num_entities'] > 0].head(15)
    if not examples.empty:
        for _, row in examples.iterrows():
            print(f"\nSentence (Class {row['classe']}): {row['sentence'][:100]}...")
            for ent in row['entities']:
                print(f"    - {ent['word']} ({ent['spacy_label']}) -> {ent['qid']} ({ent['wikidata_label']})")
    else:
        print("No examples available.")

if __name__ == "__main__":
    INPUT_CSV = "factnews.csv"
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_spacy_wikidata.csv"
    
    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)


