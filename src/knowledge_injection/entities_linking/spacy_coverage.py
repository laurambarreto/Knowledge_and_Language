'''
==================================================
ESTATÍSTICAS SÓ DO SPACY (sem Wikidata)
==================================================
Total de frases: 6191
Frases com pelo menos 1 entidade (spaCy): 4171 (67.37%)
Total de entidades encontradas pelo spaCy: 9392
'''

import spacy
import pandas as pd
from tqdm import tqdm

# Carregar o modelo
nlp = spacy.load("pt_core_news_lg")

# Carregar o teu CSV original
df = pd.read_csv("data/factnews.csv")

# Contar frases com entidades (só spaCy)
frases_com_spacy = 0
total_entidades_spacy = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="A contar entidades do spaCy"):
    sentence = str(row['sentences'])
    doc = nlp(sentence)
    # Se o spaCy encontrou pelo menos uma entidade
    if len(doc.ents) > 0:
        frases_com_spacy += 1
        total_entidades_spacy += len(doc.ents)

print("\n" + "="*50)
print("ESTATÍSTICAS SÓ DO SPACY (sem Wikidata)")
print("="*50)
print(f"Total de frases: {len(df)}")
print(f"Frases com pelo menos 1 entidade (spaCy): {frases_com_spacy} ({(frases_com_spacy/len(df))*100:.2f}%)")
print(f"Total de entidades encontradas pelo spaCy: {total_entidades_spacy}")

print("\n" + "="*50)
print("AMOSTRA DE FRASES SEM ENTIDADES (spaCy)")
print("="*50)

# Recolher frases sem entidades
frases_sem_entidades = []

for idx, row in df.iterrows():
    sentence = str(row['sentences'])
    doc = nlp(sentence)
    if len(doc.ents) == 0:
        frases_sem_entidades.append(sentence)

print(f"Total de frases sem entidades: {len(frases_sem_entidades)}")
print(f"\nPrimeiras 30 frases sem entidades:\n")

for i, frase in enumerate(frases_sem_entidades[:30], 1):
    print(f"{i}. {frase[:200]}")  # limita a 200 caracteres para não encher o ecrã