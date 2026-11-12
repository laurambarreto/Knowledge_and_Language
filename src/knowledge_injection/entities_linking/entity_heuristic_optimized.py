''' Tentar DBpedia Portuguese mas tem menor cobertura q wikidata
Demorou 47 min
==================================================
DATASET STATISTICS (Heuristic + Fixes)
==================================================
Total de frases: 6191
Frases com pelo menos 1 entidade ligada: 2912 (47.04%)

Média e Desvio Padrão de entidades por classe:
            mean       std  count
classe                           
-1      0.485262  0.853873   1391
 0      0.925035  1.247885   4242
 1      0.892473  1.158489    558

Top 15 Entidades Mais Frequentes: Todas corretas :)
   Lula: 338                                            https://www.wikidata.org/wiki/Q37181
   Bolsonaro: 266
   Brasil: 184
   PT: 164
   Jair Bolsonaro: 103
   PL: 85
   Câmara: 62
   STF: 59
   São Paulo: 57
   Renan: 53
   Moro: 46
   ONU: 42
   Brasília: 42
   PSDB: 40
   Supremo Tribunal Federal: 33
   '''
import pandas as pd
import os
import spacy
import requests
import time
from tqdm import tqdm
import torch

# --- Setup ---
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Error: Model not found. Run '!python -m spacy download pt_core_news_lg' in a new cell.")
    exit()

# --- DICIONÁRIO DE EXCEÇÕES (Opção B) ---
# Força os QIDs corretos para entidades que a Wikidata confunde.
QIDS_FIXOS = {
    "Lula": "Q37181",
    "PT": "Q657089", # ou Q1322897
    "PCC": "Q1851018",
    "Globo": "Q106562841",
    "CET": "Q18477591",
    "Mantega": "Q1359689",
    "RDC": "Q974",
    "Congo": "Q974",
    "Santo André": "Q134678", # Cidade de SP
    "Pinho": "Q1503211",
    "MP": "Q11775750", # Ministério Público
    "Europa": "Q46",
    "União Soviética": "Q15180",
    "Jair Bolsonaro": "Q10304982",
    "PL": "Q2731720",
    "Soraya": "Q59284361",
    "Ciro": "Q744798",
    "Bolsonaro": "Q10304982",
    "Tebet": "Q193037",
    "MDB": "Q1322751",
    "PSDB": "Q1322897",
    "PSOL": "Q1754827",
    "PDT": "Q1414708",
    "PTB": "Q1576143",
    "União Brasil": "Q108837483",
    "Novo": "Q818846",
    "São Paulo": "Q174",
    "Rio de Janeiro": "Q8678",
    "Brasília": "Q2844",
    "Brasil": "Q155",
}

# --- Wikidata API Function ---
def get_wikidata_qid(entity_text, spacy_label):
    # 1. Verifica primeiro o dicionário de exceções (Regra de Ouro)
    if entity_text in QIDS_FIXOS:
        return QIDS_FIXOS[entity_text], entity_text
    
    # 2. Se não estiver no dicionário, usa a heurística com fallback em inglês
    # Tenta em Português com heurística
    qid, label = search_wikidata(entity_text, spacy_label, lang="pt", strict=True)
    if qid:
        return qid, label
    
    # Tenta em Inglês com heurística
    qid, label = search_wikidata(entity_text, spacy_label, lang="en", strict=True)
    if qid:
        return qid, label
    
    # Se falhar em ambos, devolve None (melhor não ter do que ter errado)
    return None, None

def search_wikidata(entity_text, spacy_label, lang="pt", strict=True):
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
        results = data.get("search", [])
        
        if not results:
            return None, None
        
        if not strict:
            return results[0].get("id"), results[0].get("label")
        
        # Heurística Melhorada (Opção A)
        for res in results:
            desc = res.get("description", "").lower()
            qid = res.get("id")
            
            if spacy_label == "PER":
                palavras = ["político", "pessoa", "ator", "atriz", "jornalista", "jogador", "escritor", "advogado", "médico", "professor", "ministro", "presidente", "senador", "deputado", "governador", "prefeito", "candidato", "líder", "empresário", "economista", "human", "person", "politician"]
                if any(w in desc for w in palavras):
                    return qid, res.get("label")
            
            if spacy_label == "LOC":
                palavras = ["país", "cidade", "capital", "estado", "região", "rio", "localidade", "município", "província", "distrito", "bairro", "vila", "aldeia", "ilha", "continente", "montanha", "oceano", "mar", "lago", "aeroporto", "estação", "ponte", "avenida", "rua", "country", "city", "capital", "river", "region", "state", "municipality", "airport", "station", "street", "avenue"]
                if any(w in desc for w in palavras):
                    return qid, res.get("label")
            
            if spacy_label == "ORG":
                palavras = ["organização", "empresa", "partido", "clube", "instituição", "associação", "companhia", "banco", "hospital", "universidade", "escola", "instituto", "fundação", "agência", "órgão", "ministério", "secretaria", "tribunal", "corte", "polícia", "exército", "igreja", "sindicato", "federação", "confederação", "organization", "company", "party", "club", "institution", "association", "bank", "hospital", "university", "school", "institute", "foundation", "agency", "ministry", "court", "police", "army", "church", "union", "federation", "confederation"]
                if any(w in desc for w in palavras):
                    return qid, res.get("label")
        
        # Se a heurística falhar, NÃO devolve o primeiro resultado. Devolve None.
        return None, None
        
    except Exception as e:
        return None, None

# --- Main Processing Function ---
def process_dataset(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    results = []
    print("Extracting entities with spaCy and querying Wikidata (Heuristic + Fixes)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sentence = str(row['sentences']).replace('"""', '').strip()
        classe = row['classe']
        
        doc = nlp(sentence)
        
        entities_found = []
        for ent in doc.ents:
            qid, wikidata_label = get_wikidata_qid(ent.text, ent.label_)
            
            if qid: 
                entities_found.append({
                    'word': ent.text,
                    'spacy_label': ent.label_,
                    'qid': qid,
                    'wikidata_label': wikidata_label,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char
                })
                time.sleep(0.1)
                
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
        
        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return df_results

# --- Statistics Function ---
def show_statistics(df_results):
    print("\n" + "="*50)
    print("DATASET STATISTICS (Heuristic + Fixes)")
    print("="*50)
    
    df_results['num_entities'] = df_results['entities'].apply(len)
    total = len(df_results)
    with_ents = (df_results['num_entities'] > 0).sum()
    
    print(f"Total de frases: {total}")
    print(f"Frases com pelo menos 1 entidade ligada: {with_ents} ({(with_ents/total)*100:.2f}%)")
    
    print("\nMédia e Desvio Padrão de entidades por classe:")
    print(df_results.groupby('classe')['num_entities'].agg(['mean', 'std', 'count']))
    
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

if __name__ == "__main__":
    INPUT_CSV = "factnews.csv"
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_heuristic_fixed.csv"
    
    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)