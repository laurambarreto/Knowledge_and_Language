''' https://github.com/cpdoc/dhbb

VERSÃO 1 — Baseline Híbrido
- Ordem de prioridade: Fixed_Dict -> DHBB -> Wikidata (só se o DHBB falhar).
- Fixed_Dict: apenas 10 entidades principais.
- DHBB: correspondência exata de palavras (regex \b).
- Wikidata: desambiguação com BERTimbau (limiar > 0.7).
- Resultado: 35.70% cobertura | Wikidata teve 0 ligações (limiar demasiado alto).

==================================================
DATASET STATISTICS
==================================================
demorou 1h versão 1
Frases com entidades: 2210 (35.70%)

Média por classe:
            mean       std  count
classe                           
-1      0.388929  0.719007   1391
 0      0.595002  0.973138   4242
 1      0.596774  0.936277    558

Origem das Entidades:
DHBB          2196
Fixed_Dict    1202
Name: count, dtype: int64
'''

#%%writefile entity_linking_hybrid.py
import pandas as pd
import os
import spacy
import requests
import time
from tqdm import tqdm
import torch
import json
import re
from sentence_transformers import SentenceTransformer, util

# --- Setup ---
nlp = spacy.load("pt_core_news_lg")
print("A carregar BERTimbau...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased', device=device)

try:
    with open("dhbb_corpus.json", "r", encoding="utf-8") as f:
        DHBB_CORPUS = json.load(f)
    print(f"DHBB carregado: {len(DHBB_CORPUS)} entidades.")
except FileNotFoundError:
    DHBB_CORPUS = {}
    print("Aviso: dhbb_corpus.json não encontrado.")

QIDS_FIXOS = {
    "Lula": "Q37181", "Bolsonaro": "Q10304982", "Jair Bolsonaro": "Q10304982",
    "PT": "Q657089", "PCC": "Q1851018", "Mantega": "Q1359689", "Brasil": "Q155",
    "São Paulo": "Q174", "Rio de Janeiro": "Q8678", "Brasília": "Q2844"
}

def search_wikidata(entity_text, spacy_label, lang="pt"):
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "search": entity_text, "language": lang, "format": "json", "limit": 5}
    headers = {"User-Agent": "FactNews-NLP-Project/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        results = response.json().get("search", [])
        valid = []
        for res in results:
            desc = res.get("description", "").lower()
            qid = res.get("id")
            label = res.get("label")
            if spacy_label == "PER" and any(w in desc for w in ["político", "pessoa", "human", "person"]):
                valid.append({"id": qid, "label": label, "desc": f"{label}: {desc}", "source": "Wikidata"})
            elif spacy_label == "LOC" and any(w in desc for w in ["país", "cidade", "estado", "country", "city"]):
                valid.append({"id": qid, "label": label, "desc": f"{label}: {desc}", "source": "Wikidata"})
            elif spacy_label == "ORG" and any(w in desc for w in ["organização", "empresa", "partido", "organization", "company", "party"]):
                valid.append({"id": qid, "label": label, "desc": f"{label}: {desc}", "source": "Wikidata"})
        return valid
    except:
        return []

def process_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    results = []
    print("A extrair e ligar entidades...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="A processar"):
        sentence = str(row['sentences']).replace('"""', '').strip()
        doc = nlp(sentence)
        entities_found = []

        for ent in doc.ents:
            word = ent.text

            if word in QIDS_FIXOS:
                qid = QIDS_FIXOS[word]
                desc = DHBB_CORPUS.get(word, f"Entidade oficial: {word}")
                entities_found.append({
                    'word': word, 'qid': qid, 'description': desc, 'source': 'Fixed_Dict',
                    'start_char': ent.start_char, 'end_char': ent.end_char
                })
                continue

            # 2. Tenta o DHBB (Pesquisa Parcial Segura com Regex)
            dhbb_match = None
            
            for dhbb_title in DHBB_CORPUS.keys():
                # Compara usando 'word boundaries' (\b) para garantir que é a palavra inteira
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, dhbb_title, re.IGNORECASE) or re.search(r'\b' + re.escape(dhbb_title) + r'\b', word, re.IGNORECASE):
                    dhbb_match = dhbb_title
                    break
            
            if dhbb_match:
                entities_found.append({
                    'word': word, 'qid': f"DHBB_{dhbb_match}", 'description': DHBB_CORPUS[dhbb_match], 
                    'source': 'DHBB', 'start_char': ent.start_char, 'end_char': ent.end_char
                })
                continue

            w_cands = search_wikidata(word, ent.label_)
            if w_cands:
                sentence_emb = model.encode(sentence, convert_to_tensor=True)
                best_cand = None
                best_score = -1

                for cand in w_cands:
                    cand_emb = model.encode(cand['desc'][:300], convert_to_tensor=True)
                    score = util.pytorch_cos_sim(sentence_emb, cand_emb).item()
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_score > 0.7:
                    entities_found.append({
                        'word': word, 'qid': best_cand['id'], 'description': best_cand['desc'],
                        'source': 'Wikidata_BERT', 'start_char': ent.start_char, 'end_char': ent.end_char
                    })

            time.sleep(0.05)

        results.append({
            'id_sente': row['id_sente'],
            'classe': row['classe'],
            'sentence': sentence,
            'entities': entities_found
        })

        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    return df_results

def show_statistics(df_results):
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    df_results['num_entities'] = df_results['entities'].apply(len)
    total = len(df_results)
    with_ents = (df_results['num_entities'] > 0).sum()
    print(f"Frases com entidades: {with_ents} ({(with_ents/total)*100:.2f}%)")
    print("\nMédia por classe:")
    print(df_results.groupby('classe')['num_entities'].agg(['mean', 'std', 'count']))

    sources = [e['source'] for ent_list in df_results['entities'] for e in ent_list]
    if sources:
        print("\nOrigem das Entidades:")
        print(pd.Series(sources).value_counts())

if __name__ == "__main__":
    # --- ATENÇÃO: Muda o caminho do INPUT_CSV para onde tens o ficheiro no teu Drive ---
    INPUT_CSV = "/content/drive/MyDrive/FactNews_Project/factnews.csv"

    # --- O output vai ser guardado na mesma pasta do Drive ---
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_hybrid_linked_v1.csv"

    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)