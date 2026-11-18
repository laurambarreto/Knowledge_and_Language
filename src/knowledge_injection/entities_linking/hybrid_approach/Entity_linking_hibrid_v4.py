'''

VERSÃO 4 — Heurística + Cache + Descrições Reais
- Fixed_Dict: passa a ir buscar descrições oficiais à Wikidata (PT/EN).
- DHBB: descrições encurtadas para a 1.ª frase.
- Wikidata: removido o BERTimbau; substituído por filtro heurístico (PER/LOC/ORG) + cache.
- Resultado: 38.88% cobertura | Wikidata desbloqueada (581 ligações).


==================================================
DATASET STATISTICS (FINAL)
==================================================
demorou 1h versão 4
Frases com entidades ligadas: 2407 (38.88%)

Média por classe:
            mean       std  count
classe                           
-1      0.391086  0.728282   1391
 0      0.691891  1.085835   4242
 1      0.727599  1.064015    558

Origem das Entidades:
Fixed_Dict            1878
DHBB                  1426
Wikidata_Heuristic     581
Name: count, dtype: int64'''

#%%writefile entity_linking_final.py
import pandas as pd
import os
import spacy
import requests
import time
from tqdm import tqdm
import json
import re

# --- Setup ---
nlp = spacy.load("pt_core_news_lg")

try:
    with open("dhbb_corpus.json", "r", encoding="utf-8") as f:
        DHBB_CORPUS = json.load(f)
    print(f"DHBB carregado: {len(DHBB_CORPUS)} entidades.")
except FileNotFoundError:
    DHBB_CORPUS = {}
    print("Aviso: dhbb_corpus.json não encontrado.")

QIDS_FIXOS = {
    "Lula": "Q37181", "Luiz Inácio Lula da Silva": "Q37181",
    "Bolsonaro": "Q10304982", "Jair Bolsonaro": "Q10304982",
    "Ciro": "Q744798", "Ciro Gomes": "Q744798",
    "Soraya": "Q59284361", "Soraya Thronicke": "Q59284361",
    "Mantega": "Q1359689", "Guido Mantega": "Q1359689",
    "Moro": "Q10376136", "Sergio Moro": "Q10376136", "Sérgio Moro": "Q10376136",
    "Renan": "Q16940860", "Renan Calheiros": "Q16940860",
    "Dunga": "Q215890", "Ronaldinho": "Q39444", "Ronaldinho Gaúcho": "Q39444",
    "Thiago Pereira": "Q138656",
    "PT": "Q657089", "PL": "Q2731720", "PSDB": "Q1322897", "MDB": "Q1322751",
    "PDT": "Q1414708", "PSOL": "Q1754827", "PTB": "Q1576143", "União Brasil": "Q108837483",
    "PCC": "Q1851018", "Globo": "Q106562841", "TV Globo": "Q106562841",
    "CET": "Q18477591", "MP": "Q11775750", "Ministério Público": "Q11775750",
    "ONU": "Q1065", "Nações Unidas": "Q1065",
    "Brasil": "Q155", "São Paulo": "Q174", "Rio de Janeiro": "Q8678", "Brasília": "Q2844",
    "Europa": "Q46", "União Europeia": "Q458", "União Européia": "Q458",
    "União Soviética": "Q15180", "EUA": "Q30", "Estados Unidos": "Q30",
    "Congo": "Q974", "RDC": "Q974", "República Democrática do Congo": "Q974",
    "Kinshasa": "Q3838", "Bukavu": "Q222279"
}
BLACKLIST = {
    "estado", "casa", "novo", "terra", "rio", "país", "câmara", "senado", "república",
    "justiça", "norte", "sul", "leste", "oeste", "centro", "defesa", "conselho", "ação",
    "democracia", "progresso", "saúde", "educação", "nacional"
}

def encurtar_descricao(descricao):
    if not descricao:
        return ""
    descricao_curta = descricao.split('.')[0].strip()
    return f"{descricao_curta}."

def carregar_descricoes_fixas(qids_dict):
    """Vai buscar as descrições reais à Wikidata em lotes de 50 QIDs."""
    qids_unicos = list(set(qids_dict.values()))
    mapa_descricoes = {}
    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "FactNews-NLP-Project/1.0"}

    print(f"A obter descrições oficiais para {len(qids_unicos)} QIDs fixos...")
    for i in range(0, len(qids_unicos), 50):
        lote = qids_unicos[i:i+50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(lote),
            "props": "labels|descriptions",
            "languages": "pt|en",
            "format": "json"
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20).json()
            for qid, dados in resp.get("entities", {}).items():
                desc = (dados.get("descriptions", {}).get("pt", {}).get("value") or
                        dados.get("descriptions", {}).get("en", {}).get("value") or "")
                label = (dados.get("labels", {}).get("pt", {}).get("value") or
                         dados.get("labels", {}).get("en", {}).get("value") or qid)
                mapa_descricoes[qid] = f"{label}: {desc}." if desc else f"{label}."
        except Exception as e:
            print(f"Aviso ao obter lote de QIDs: {e}")
    return mapa_descricoes

# Carrega as descrições reais uma única vez no início
DESCRICOES_QIDS_FIXOS = carregar_descricoes_fixas(QIDS_FIXOS)
WIKIDATA_CACHE = {}

def search_wikidata_heuristic(entity_text, spacy_label, lang="pt"):
    chave = (entity_text.lower(), spacy_label, lang)
    if chave in WIKIDATA_CACHE:
        return WIKIDATA_CACHE[chave]

    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "search": entity_text, "language": lang, "format": "json", "limit": 5}
    headers = {"User-Agent": "FactNews-NLP-Project/1.0"}
    
    resultado = (None, None, None)
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        results = response.json().get("search", [])
        
        for res in results:
            desc = res.get("description", "").lower()
            qid = res.get("id")
            label = res.get("label")
            
            if spacy_label == "PER":
                if any(w in desc for w in ["político", "pessoa", "ator", "jornalista", "jogador", "escritor", "ministro", "presidente", "human", "person", "politician"]):
                    resultado = (qid, label, desc)
                    break
            elif spacy_label == "LOC":
                if any(w in desc for w in ["país", "cidade", "capital", "estado", "município", "província", "aeroporto", "country", "city", "region", "state"]):
                    resultado = (qid, label, desc)
                    break
            elif spacy_label == "ORG":
                if any(w in desc for w in ["organização", "empresa", "partido", "banco", "universidade", "instituto", "ministério", "tribunal", "organization", "company", "party"]):
                    resultado = (qid, label, desc)
                    break
    except:
        pass

    WIKIDATA_CACHE[chave] = resultado
    time.sleep(0.25)  # Só espera quando faz pedido real à API
    return resultado

def process_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    results = []
    print("A extrair e ligar entidades (Heurística + Cache + Descrições Reais)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="A processar"):
        sentence = str(row['sentences']).replace('"""', '').strip()
        doc = nlp(sentence)
        entities_found = []

        for ent in doc.ents:
            word = ent.text

            # 1. Bloqueio imediato da Blacklist
            if word.lower() in BLACKLIST:
                continue

            # 2. Dicionário de Ouro (agora com descrição real da Wikidata)
            if word in QIDS_FIXOS:
                qid = QIDS_FIXOS[word]
                desc = DESCRICOES_QIDS_FIXOS.get(qid, f"{word}.")
                entities_found.append({
                    'word': word, 'qid': qid, 'description': desc, 'source': 'Fixed_Dict',
                    'start_char': ent.start_char, 'end_char': ent.end_char
                })
                continue

            # 3. DHBB (Regex rigoroso)
            dhbb_match = None
            for dhbb_title in DHBB_CORPUS.keys():
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, dhbb_title, re.IGNORECASE) or re.search(r'\b' + re.escape(dhbb_title) + r'\b', word, re.IGNORECASE):
                    dhbb_match = dhbb_title
                    break

            if dhbb_match:
                desc_curta = encurtar_descricao(DHBB_CORPUS[dhbb_match])
                entities_found.append({
                    'word': word, 'qid': f"DHBB_{dhbb_match}", 'description': desc_curta,
                    'source': 'DHBB', 'start_char': ent.start_char, 'end_char': ent.end_char
                })
                continue

            # 4. Wikidata (Heurística + Cache)
            qid, w_label, w_desc = search_wikidata_heuristic(word, ent.label_, lang="pt")
                
            if qid:
                entities_found.append({
                    'word': word, 'qid': qid, 'description': f"{w_label}: {w_desc}.",
                    'source': 'Wikidata_Heuristic', 'start_char': ent.start_char, 'end_char': ent.end_char
                })

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
    print("DATASET STATISTICS (FINAL)")
    print("="*50)
    df_results['num_entities'] = df_results['entities'].apply(len)
    total = len(df_results)
    with_ents = (df_results['num_entities'] > 0).sum()
    print(f"Frases com entidades ligadas: {with_ents} ({(with_ents/total)*100:.2f}%)")
    print("\nMédia por classe:")
    print(df_results.groupby('classe')['num_entities'].agg(['mean', 'std', 'count']))

    sources = [e['source'] for ent_list in df_results['entities'] for e in ent_list]
    if sources:
        print("\nOrigem das Entidades:")
        print(pd.Series(sources).value_counts())

if __name__ == "__main__":
    INPUT_CSV = "/content/drive/MyDrive/FactNews_Project/factnews.csv"
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_hybrid_linked_v4.csv"

    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)