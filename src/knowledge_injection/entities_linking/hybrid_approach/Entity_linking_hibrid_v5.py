'''

VERSÃO 5 — 100% PT + Desempate Semântico (Final)
- Idioma: 100% em português (uselang='pt' na API da Wikidata).
- Fixed_Dict: removemos os QIDs de Moro e Thiago Pereira; Renan passou para desempate.
- Desempate: quando DHBB e Wikidata encontram candidato, escolhe o melhor por similaridade
  de cosseno (Sentence-BERT: paraphrase-multilingual-MiniLM-L12-v2).
- Extras: prints de rastreabilidade dos desempates e retoma automática por checkpoint.
- Resultado: 39.22% cobertura | Fixed_Dict: 1799 | DHBB: 1274 | Wikidata: 808.


==================================================
DATASET STATISTICS (V5 - 100% PT + DESEMPATE)
==================================================
Frases com entidades ligadas: 2428 (39.22%) - 1h15min estimativa

Média por classe:
            mean       std  count
classe                           
-1      0.386772  0.730584   1391
 0      0.693069  1.073614   4242
 1      0.722222  1.018628    558

==================================================
RELATÓRIO DE COBERTURA E DESEMPATES (V5)
==================================================
Fixed_Dict (direto):          86
Só DHBB tinha candidato:      35
Só Wikidata tinha candidato:  6
Ambas tinham (Desempates):    1
  -> DHBB ganhou:             0 (0.0%)
  -> Wikidata ganhou:         1 (100.0%)

 --- [Desempate #1] Entidade: 'Estados' ---
Frase:    Foram reportados 'condutas abusivas' em 24 dos 26 Estados.
DHBB     (-0.020): Grupo de trabalho oficialmente instalado na órbita e no próprio prédio
do Ministério da Fazenda em 19 de julho de 1951.
Wikidata (0.165): Estados Unidos: país localizado principalmente na américa do norte.
-> Vencedor: Wikidata_Heuristic

Fixed_Dict          : 1273 frases (20.56% de cobertura)
DHBB                : 1028 frases (16.60% de cobertura)
Wikidata_Heuristic  :  666 frases (10.76% de cobertura)
 '''

#%%writefile entity_linking_v5.py
import pandas as pd
import os
import spacy
import requests
import time
import json
import re
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import ast

# --- 1. Setup ---
nlp = spacy.load("pt_core_news_lg")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"A carregar SentenceTransformer em {device}...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

try:
    with open("dhbb_corpus.json", "r", encoding="utf-8") as f:
        DHBB_CORPUS = json.load(f)
    print(f"DHBB carregado: {len(DHBB_CORPUS)} entidades.")
except FileNotFoundError:
    DHBB_CORPUS = {}
    print("Aviso: dhbb_corpus.json não encontrado.")

# QIDS_FIXOS 100% verificado (Moro e Thiago Pereira corrigidos; Renan removido para ir ao desempate)
QIDS_FIXOS = {
    "Lula": "Q37181", "Luiz Inácio Lula da Silva": "Q37181",
    "Bolsonaro": "Q10304982", "Jair Bolsonaro": "Q10304982",
    "Ciro": "Q744798", "Ciro Gomes": "Q744798",
    "Soraya": "Q59284361", "Soraya Thronicke": "Q59284361",
    "Mantega": "Q1359689", "Guido Mantega": "Q1359689",
    "Moro": "Q19607958", "Sergio Moro": "Q19607958", "Sérgio Moro": "Q19607958",
    "Dunga": "Q215890", "Ronaldinho": "Q39444", "Ronaldinho Gaúcho": "Q39444",
    "Thiago Pereira": "Q540294",
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
    """Vai buscar as descrições reais em PORTUGUÊS à Wikidata em lotes de 50 QIDs."""
    qids_unicos = list(set(qids_dict.values()))
    mapa_descricoes = {}
    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "FactNews-NLP-Project/1.0"}

    print(f"A obter descrições oficiais em PT para {len(qids_unicos)} QIDs fixos...")
    for i in range(0, len(qids_unicos), 50):
        lote = qids_unicos[i:i+50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(lote),
            "props": "labels|descriptions",
            "languages": "pt",
            "uselang": "pt",
            "format": "json"
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20).json()
            for qid, dados in resp.get("entities", {}).items():
                desc = dados.get("descriptions", {}).get("pt", {}).get("value", "")
                label = dados.get("labels", {}).get("pt", {}).get("value", qid)
                mapa_descricoes[qid] = f"{label}: {desc}." if desc else f"{label}."
        except Exception as e:
            print(f"Aviso ao obter lote de QIDs: {e}")
    return mapa_descricoes

DESCRICOES_QIDS_FIXOS = carregar_descricoes_fixas(QIDS_FIXOS)
WIKIDATA_CACHE = {}

def search_wikidata_heuristic_pt(entity_text, spacy_label):
    """Pesquisa na Wikidata apenas em português, com cache por (palavra, tipo)."""
    chave = (entity_text.lower(), spacy_label)
    if chave in WIKIDATA_CACHE:
        return WIKIDATA_CACHE[chave]

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": "pt",
        "uselang": "pt",
        "format": "json",
        "limit": 5
    }
    headers = {"User-Agent": "FactNews-NLP-Project/1.0"}

    resultado = (None, None, None)
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        results = response.json().get("search", [])

        for res in results:
            desc = res.get("description", "").lower()
            if not desc:
                continue
            qid = res.get("id")
            label = res.get("label")

            if spacy_label == "PER":
                if any(w in desc for w in ["político", "política", "pessoa", "ator", "atriz", "jornalista", "jogador", "futebolista", "escritor", "escritora", "ministro", "presidente", "governador", "senador", "deputado", "juiz", "advogado", "nadador", "técnico"]):
                    resultado = (qid, label, desc)
                    break
            elif spacy_label == "LOC":
                if any(w in desc for w in ["país", "cidade", "capital", "estado", "município", "província", "aeroporto", "região", "continente", "distrito"]):
                    resultado = (qid, label, desc)
                    break
            elif spacy_label == "ORG":
                if any(w in desc for w in ["organização", "empresa", "partido", "banco", "universidade", "instituto", "ministério", "tribunal", "órgão", "agência", "emissora", "jornal", "revista", "polícia", "força"]):
                    resultado = (qid, label, desc)
                    break
    except:
        pass

    WIKIDATA_CACHE[chave] = resultado
    time.sleep(0.25)
    return resultado

def search_dhbb(word):
    """Procura correspondência rigorosa no DHBB."""
    pattern = r'\b' + re.escape(word) + r'\b'
    for dhbb_title in DHBB_CORPUS.keys():
        if re.search(pattern, dhbb_title, re.IGNORECASE) or re.search(r'\b' + re.escape(dhbb_title) + r'\b', word, re.IGNORECASE):
            return f"DHBB_{dhbb_title}", encurtar_descricao(DHBB_CORPUS[dhbb_title])
    return None, None

def process_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Retomar do checkpoint se o ficheiro já existir
    if os.path.exists(output_path):
        df_done = pd.read_csv(output_path)
        df_done['entities'] = df_done['entities'].apply(ast.literal_eval)
        start_idx = len(df_done)
        results = df_done.to_dict('records')
        print(f"A retomar da frase {start_idx} de {len(df)} (só faltam {len(df) - start_idx})...")
    else:
        start_idx = 0
        results = []

    df_restante = df.iloc[start_idx:]
    desempates_count = 0
    stats = {"fixed_dict": 0, "only_dhbb": 0, "only_wiki": 0, "both_dhbb_won": 0, "both_wiki_won": 0}

    for idx, row in tqdm(df_restante.iterrows(), total=len(df_restante), desc="A retomar"):
        sentence = str(row['sentences']).replace('"""', '').strip()
        doc = nlp(sentence)
        entities_found = []
        sent_emb = None  # Só calcula embedding da frase se houver desempate

        for ent in doc.ents:
            word = ent.text

            # 1. Blacklist
            if word.lower() in BLACKLIST:
                continue

            # 2. Dicionário Fixo (100% verificado)
            if word in QIDS_FIXOS:
                qid = QIDS_FIXOS[word]
                desc = DESCRICOES_QIDS_FIXOS.get(qid, f"{word}.")
                entities_found.append({
                    'word': word, 'qid': qid, 'description': desc, 'source': 'Fixed_Dict',
                    'start_char': ent.start_char, 'end_char': ent.end_char
                })
                stats["fixed_dict"] += 1
                continue

            # 3. Recolher candidatos do DHBB e da Wikidata (PT)
            dhbb_qid, dhbb_desc = search_dhbb(word)
            wiki_qid, wiki_label, wiki_desc_raw = search_wikidata_heuristic_pt(word, ent.label_)
            wiki_desc = f"{wiki_label}: {wiki_desc_raw}." if wiki_qid else None

            # 4. Decisão: se ambos existem, desempatar por similaridade de cosseno
            if dhbb_qid and wiki_qid:
                if sent_emb is None:
                    sent_emb = embedder.encode(sentence, convert_to_tensor=True, show_progress_bar=False)

                cand_embs = embedder.encode([dhbb_desc, wiki_desc], convert_to_tensor=True, show_progress_bar=False)
                sim_dhbb = util.cos_sim(sent_emb, cand_embs[0]).item()
                sim_wiki = util.cos_sim(sent_emb, cand_embs[1]).item()

                desempates_count += 1
                vencedor = "DHBB" if sim_dhbb >= sim_wiki else "Wikidata_Heuristic"

                # Mostra os primeiros 5 e depois de 3 em 3
                if desempates_count <= 5 or desempates_count % 3 == 0:
                    msg = (
                        f"\n--- [Desempate #{desempates_count}] Entidade: '{word}' ---\n"
                        f"Frase:    {sentence}\n"
                        f"DHBB     ({sim_dhbb:.3f}): {dhbb_desc}\n"
                        f"Wikidata ({sim_wiki:.3f}): {wiki_desc}\n"
                        f"-> Vencedor: {vencedor}\n"
                    )
                    tqdm.write(msg)

                if sim_dhbb >= sim_wiki:
                    stats["both_dhbb_won"] += 1
                    entities_found.append({
                        'word': word, 'qid': dhbb_qid, 'description': dhbb_desc,
                        'source': 'DHBB', 'start_char': ent.start_char, 'end_char': ent.end_char
                    })
                else:
                    stats["both_wiki_won"] += 1
                    entities_found.append({
                        'word': word, 'qid': wiki_qid, 'description': wiki_desc,
                        'source': 'Wikidata_Heuristic', 'start_char': ent.start_char, 'end_char': ent.end_char
                    })

            elif dhbb_qid:
                stats["only_dhbb"] += 1
                entities_found.append({
                    'word': word, 'qid': dhbb_qid, 'description': dhbb_desc,
                    'source': 'DHBB', 'start_char': ent.start_char, 'end_char': ent.end_char
                })

            elif wiki_qid:
                stats["only_wiki"] += 1
                entities_found.append({
                    'word': word, 'qid': wiki_qid, 'description': wiki_desc,
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
    print(f"\nTotal de desempates semânticos realizados: {desempates_count}")
    total_both = stats["both_dhbb_won"] + stats["both_wiki_won"]
    print("\n" + "="*50)
    print("RELATÓRIO DE COBERTURA E DESEMPATES (V5)")
    print("="*50)
    print(f"Fixed_Dict (direto):          {stats['fixed_dict']}")
    print(f"Só DHBB tinha candidato:      {stats['only_dhbb']}")
    print(f"Só Wikidata tinha candidato:  {stats['only_wiki']}")
    print(f"Ambas tinham (Desempates):    {total_both}")
    if total_both > 0:
        print(f"  -> DHBB ganhou:             {stats['both_dhbb_won']} ({stats['both_dhbb_won']/total_both*100:.1f}%)")
        print(f"  -> Wikidata ganhou:         {stats['both_wiki_won']} ({stats['both_wiki_won']/total_both*100:.1f}%)")
    return df_results

def show_statistics(df_results):
    print("\n" + "="*50)
    print("DATASET STATISTICS (V5 - 100% PT + DESEMPATE)")
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
    OUTPUT_CSV = "/content/drive/MyDrive/FactNews_Project/entities_hybrid_linked_v5.csv"

    df_results = process_dataset(INPUT_CSV, OUTPUT_CSV)
    show_statistics(df_results)