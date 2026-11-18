'''

VERSÃO 2 — Expansão + Blacklist
- Fixed_Dict: lista alargada (políticos, partidos, desporto, locais) + descrições encurtadas.
- Blacklist: adicionada para ignorar palavras ambíguas (ex: estado, casa, rio).
- Wikidata: limiar do BERTimbau baixado de 0.7 para 0.55.
- Resultado: 35.76% cobertura | Wikidata continuou quase sem efeito (apenas 7 ligações).

==================================================
DATASET STATISTICS
==================================================
demorou 1h versão 2
Frases com entidades: 2214 (35.76%)

Média por classe:
            mean       std  count
classe                           
-1      0.359454  0.700603   1391
 0      0.613154  1.022983   4242
 1      0.670251  1.000293    558

Origem das Entidades:
Fixed_Dict       2205
DHBB             1263
Wikidata_BERT       7
Name: count, dtype: int64'''

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
    # --- POLÍTICOS (Nome Completo e Variações) ---
    "Lula": "Q37181", 
    "Luiz Inácio Lula da Silva": "Q37181",
    "Bolsonaro": "Q10304982", 
    "Jair Bolsonaro": "Q10304982",
    "Ciro": "Q744798", 
    "Ciro Gomes": "Q744798",
    "Tebet": "Q193037", 
    "Simone Tebet": "Q193037",
    "Soraya": "Q59284361", 
    "Soraya Thronicke": "Q59284361",
    "Felipe": "Q10280975", 
    "Felipe d'Ávila": "Q10280975",
    "Alckmin": "Q903460",
    "Geraldo Alckmin": "Q903460",
    "Mantega": "Q1359689",
    "Guido Mantega": "Q1359689",
    "Moro": "Q10376136",
    "Sergio Moro": "Q10376136",
    "Sérgio Moro": "Q10376136",
    "Renan": "Q16940860",
    "Renan Calheiros": "Q16940860",
    
    # --- DESPORTO (Figuras que aparecem muito no teu dataset) ---
    "Kaká": "Q184138",
    "Robinho": "Q174780",
    "Dunga": "Q215890",
    "Vagner Love": "Q222746",
    "Vágner Love": "Q222746",
    "Ronaldinho": "Q39444",
    "Ronaldinho Gaúcho": "Q39444",
    "Thiago Pereira": "Q138656",
    
    # --- PARTIDOS POLÍTICOS ---
    "PT": "Q657089", 
    "PL": "Q2731720",
    "PSDB": "Q1322897",
    "MDB": "Q1322751",
    "PDT": "Q1414708",
    "PSOL": "Q1754827",
    "PTB": "Q1576143",
    "União Brasil": "Q108837483",
    "Novo": "Q818846",
    "DEM": "Q1549021",
    
    # --- ORGANIZAÇÕES E INSTITUIÇÕES ---
    "PCC": "Q1851018",
    "Globo": "Q106562841",
    "TV Globo": "Q106562841",
    "CET": "Q18477591",
    "MP": "Q11775750", 
    "Ministério Público": "Q11775750",
    "ONU": "Q1065",
    "Nações Unidas": "Q1065",
    "STF": "Q4808383",
    "Supremo Tribunal Federal": "Q4808383",
    "TSE": "Q7840398",
    "Tribunal Superior Eleitoral": "Q7840398",
    "Senado": "Q213458",
    "Câmara": "Q1821034",
    
    # --- GEOGRAFIA E PAÍSES ---
    "Brasil": "Q155",
    "São Paulo": "Q174",
    "Rio de Janeiro": "Q8678",
    "Brasília": "Q2844",
    "Santo André": "Q134678",
    "Europa": "Q46",
    "União Europeia": "Q458",
    "União Européia": "Q458",
    "União Soviética": "Q15180",
    "EUA": "Q30",
    "Estados Unidos": "Q30",
    "Congo": "Q974", 
    "RDC": "Q974",
    "República Democrática do Congo": "Q974",
    "Kinshasa": "Q3838",
    "Bukavu": "Q222279",
    
    # --- PESSOAS ESPECÍFICAS/DIFÍCEIS DO DATASET ---
    "Okala": "Q109791897",
    "Jean-Tobias Okala": "Q109791897",
    "Pinho": "Q1503211",
}
# 2. Blacklist para matar ambiguidades (aumenta a correctness)
BLACKLIST = {
    "estado", "casa", "novo", "terra", "rio", "país", "câmara", "senado", "república",
    "justiça", "norte", "sul", "leste", "oeste", "centro", "defesa", "conselho", "ação",
    "democracia", "progresso", "saúde", "educação", "nacional"
}

def encurtar_descricao(descricao):
    if not descricao:
        return ""
    # Corta o texto no primeiro ponto final e volta a adicionar o ponto
    descricao_curta = descricao.split('.')[0].strip()
    return f"{descricao_curta}."

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

            # Bloqueio imediato da Blacklist
            if word.lower() in BLACKLIST:
                continue

            if word in QIDS_FIXOS:
                qid = QIDS_FIXOS[word]
                desc_original = DHBB_CORPUS.get(word, f"Entidade oficial: {word}")
                desc = encurtar_descricao(desc_original)
                entities_found.append({
                    'word': word, 'qid': qid, 'description': desc, 'source': 'Fixed_Dict',
                    'start_char': ent.start_char, 'end_char': ent.end_char
                })
                continue

            dhbb_match = None
            for dhbb_title in DHBB_CORPUS.keys():
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

                # 3. Limite corrigido para 0.55 para permitir que a Wikidata funcione!
                if best_score > 0.55:
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
    INPUT_CSV = "/content/drive/MyDrive/FactNews_Project/factnews.csv"
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_hybrid_linked_v2.csv"

    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)