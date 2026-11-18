'''
==================================================
AUDITORIA DE PRECISÃO (Amostra das top entidades)
==================================================
Total de entidades verificadas: 1193
Acertos: 345 (28.92%)
Erros: 848 (71.08%)

Exemplos de erros (primeiros 20):
  - Palavra: ONU | Esperado: Q1065 | Obtido: Q25188868 (Onu)
  - Palavra: RDC | Esperado: Q974 | Obtido: Q4697969 (Air Force Systems Command)
  - Palavra: Congo | Esperado: Q974 | Obtido: Q2597556 (Congo)
  - Palavra: Congo | Esperado: Q974 | Obtido: Q33702 (Kongo)
  - Palavra: Europa | Esperado: Q46 | Obtido: Q458 (European Union)
  - Palavra: Congo | Esperado: Q974 | Obtido: Q2597556 (Congo)
  - Palavra: RDC | Esperado: Q974 | Obtido: Q4697969 (Air Force Systems Command)
  - Palavra: ONU | Esperado: Q1065 | Obtido: Q641576 (UN Women)
  - Palavra: RDC | Esperado: Q974 | Obtido: Q4697969 (Air Force Systems Command)
  - Palavra: PT | Esperado: Q657089 | Obtido: Q355 (Facebook)
  - Palavra: Jair Bolsonaro | Esperado: Q10304982 | Obtido: Q57919989 (Jair Bolsonaro é eleito presidente do Brasil)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q58659722 (Bolsonaro anuncia o general Fernando Azevedo e Silva para a Defesa)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q93202317 (Bolsonaro nomeia o novo ministro da Justiça e o diretor geral da Polícia Federal)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q59624508 (Bolsonaro e Mourão são diplomados no Brasil)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q97449707 (Bolsonaro não repassará recursos de multas ambientais para ONGs)
  - Palavra: Lula | Esperado: Q37181 | Obtido: Q1990293 (Luiz Ribeiro Pinto Neto)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q97448602 (Bolsonaro demite diretor da Polícia Federal e desagrada Moro)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q93202317 (Bolsonaro nomeia o novo ministro da Justiça e o diretor geral da Polícia Federal)
  - Palavra: Bolsonaro | Esperado: Q10304982 | Obtido: Q97449707 (Bolsonaro não repassará recursos de multas ambientais para ONGs)
  - Palavra: Lula | Esperado: Q37181 | Obtido: Q2201335 (Lula)
'''
import pandas as pd
import ast

# Carregar o teu CSV final do BERT
df = pd.read_csv("/content/drive/MyDrive/FactNews_Project/entities_bert_disambiguated.csv")

# Dicionário de "QIDs de Ouro" (os corretos para as entidades que sabemos que falharam)
qids_corretos = {
    "Lula": "Q37181",
    "PT": "Q657089", # ou Q1322897 (Partido dos Trabalhadores)
    "PCC": "Q1851018",
    "Congo": "Q974", # República Democrática do Congo (assumindo que é o contexto)
    "RDC": "Q974",
    "CET": "Q18477591", # Companhia de Engenharia de Tráfego
    "Globo": "Q106562841", # ou Q3587 (TV Globo)
    "Bolsonaro": "Q10304982",
    "Jair Bolsonaro": "Q10304982",
    "São Paulo": "Q174",
    "Brasil": "Q155",
    "ONU": "Q1065",
    "União Soviética": "Q15180", # o QID correto é Q15180, mas o teu deu Q99288945 (erro)
    "Europa": "Q46" # o QID correto é Q46, mas o teu deu Q458 (União Europeia)
}

# Inicializar contadores
total_entidades_testadas = 0
acertos = 0
erros = 0
detalhes_erros = []

# Iterar sobre as frases
for idx, row in df.iterrows():
    if isinstance(row['entities'], str):
        # Converter a string do CSV de volta para lista de dicionários
        try:
            entities = ast.literal_eval(row['entities'])
        except:
            continue
    else:
        entities = row['entities']
        
    for ent in entities:
        word = ent['word']
        qid = ent['qid']
        label = ent['wikidata_label']
        
        # Se a palavra estiver no nosso dicionário de ouro, testamos
        if word in qids_corretos:
            total_entidades_testadas += 1
            qid_esperado = qids_corretos[word]
            
            if qid == qid_esperado:
                acertos += 1
            else:
                erros += 1
                detalhes_erros.append(f"Palavra: {word} | Esperado: {qid_esperado} | Obtido: {qid} ({label})")

print("\n" + "="*50)
print("AUDITORIA DE PRECISÃO (Amostra das top entidades)")
print("="*50)
print(f"Total de entidades verificadas: {total_entidades_testadas}")
print(f"Acertos: {acertos} ({(acertos/total_entidades_testadas)*100:.2f}%)")
print(f"Erros: {erros} ({(erros/total_entidades_testadas)*100:.2f}%)")

print("\nExemplos de erros (primeiros 20):")
for err in detalhes_erros[:20]:
    print(f"  - {err}")