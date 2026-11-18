import pandas as pd
import ast
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sentence_transformers import SentenceTransformer, util

# 1. Carregar dados e modelo multilingue
INPUT_CSV = "../entities_linking/data/hybrid_approach/entities_hybrid_linked_v5.csv"
OUTPUT_CSV = "final_extracted_entities_v5.csv"
THRESHOLD = 0.20  # Limiar de corte

df = pd.read_csv(INPUT_CSV)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"A carregar modelo em {device}...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

# 2. Extrair todos os pares (frase, descrição da entidade)
pares = []
for idx, row in df.iterrows():
    sentence = str(row['sentence'])
    try:
        ents = ast.literal_eval(row['entities']) if isinstance(row['entities'], str) else row['entities']
    except:
        continue
    for e in ents:
        pares.append({
            'id_sente': row['id_sente'],
            'classe': row['classe'],
            'sentence': sentence,
            'word': e['word'],
            'qid': e['qid'],
            'source': e['source'],
            'description': e['description'],
            'start_char': e.get('start_char'),
            'end_char': e.get('end_char')
        })

df_scores = pd.DataFrame(pares)
print(f"Total de pares (frase, entidade) a avaliar: {len(df_scores)}")

# 3. Calcular embeddings em batch
sent_embs = model.encode(df_scores['sentence'].tolist(), batch_size=64, convert_to_tensor=True, show_progress_bar=True)
desc_embs = model.encode(df_scores['description'].tolist(), batch_size=64, convert_to_tensor=True, show_progress_bar=True)

scores = util.pairwise_cos_sim(sent_embs, desc_embs).cpu().numpy()
df_scores['similarity'] = scores
df_scores = df_scores.sort_values(by='similarity', ascending=False).reset_index(drop=True)

# 4. Estatísticas da distribuição e impacto do corte
df_filtrado = df_scores[(df_scores['source'] == 'Fixed_Dict') | (df_scores['similarity'] >= THRESHOLD)].copy()

total_frases = df['id_sente'].nunique()
frases_antes = df_scores['id_sente'].nunique()
frases_depois = df_filtrado['id_sente'].nunique()

print("\n" + "="*55)
print(f"IMPACTO DO RELEVANCE SCORING (THRESHOLD >= {THRESHOLD})")
print("="*55)
print(f"Total de frases no dataset:           {total_frases}")
print(f"Frases com entidades ANTES do RS:     {frases_antes} ({frases_antes/total_frases*100:.2f}%)")
print(f"Frases com entidades DEPOIS do RS:    {frases_depois} ({frases_depois/total_frases*100:.2f}%)")
print(f"Entidades mantidas:                   {len(df_filtrado)} de {len(df_scores)} ({len(df_filtrado)/len(df_scores)*100:.1f}%)")
print("\nResumo estatístico da similaridade (todas as entidades):")
print(df_scores['similarity'].describe())

# 5. Gráficos: Barras por Classe em cima + 3 Histogramas Separados em baixo
classes_ordem = [-1, 0, 1]
nomes_classes = ["Citação (-1)", "Facto (0)", "Viés (1)"]

total_real = df.groupby('classe')['id_sente'].nunique().reindex(classes_ordem, fill_value=0)
antes_rs = df_scores.groupby('classe')['id_sente'].nunique().reindex(classes_ordem, fill_value=0)
depois_rs = df_filtrado.groupby('classe')['id_sente'].nunique().reindex(classes_ordem, fill_value=0)

df_barras = pd.DataFrame({
    'Classe': nomes_classes * 3,
    'Contagem': list(total_real.values) + list(antes_rs.values) + list(depois_rs.values),
    'Fase': (['Total Real'] * 3) + (['Antes do RS'] * 3) + ([f'Depois do RS (>= {THRESHOLD})'] * 3)
})

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

# --- TOPO: Barras Agrupadas por Classe ---
ax_top = fig.add_subplot(gs[0, :])
sns.barplot(
    data=df_barras, x='Classe', y='Contagem', hue='Fase',
    palette=['#4c72b0', '#dd8452', '#55a868'], ax=ax_top
)
for container in ax_top.containers:
    ax_top.bar_label(container, fmt='%d', padding=3, fontsize=10)

ax_top.set_title("Frases por Classe: Total Real vs. Com Entidades (Antes e Depois do RS)", fontsize=14, fontweight='bold')
ax_top.set_xlabel("Classe", fontsize=11)
ax_top.set_ylabel("Número de Frases", fontsize=11)
ax_top.legend(title="Etapa")
ax_top.grid(axis='y', alpha=0.3)

# --- FUNDO: 3 Histogramas Separados por Abordagem ---
fontes = [
    ("Fixed_Dict", "#1f77b4", "Fixed_Dict (Dicionário Fixo)"),
    ("DHBB", "#ff7f0e", "DHBB (Dicionário Histórico)"),
    ("Wikidata_Heuristic", "#2ca02c", "Wikidata_Heuristic (PT)")
]

for col_idx, (src, cor, titulo) in enumerate(fontes):
    ax = fig.add_subplot(gs[1, col_idx])
    sub = df_scores[df_scores['source'] == src]

    sns.histplot(data=sub, x='similarity', bins=30, color=cor, alpha=0.85, ax=ax)

    media = sub['similarity'].mean()
    mediana = sub['similarity'].median()
    pct_acima = (sub['similarity'] >= THRESHOLD).mean() * 100

    ax.axvline(media, color='red', linestyle='--', linewidth=1.8, label=f"Média ({media:.2f})")
    ax.axvline(mediana, color='black', linestyle=':', linewidth=1.8, label=f"Mediana ({mediana:.2f})")
    ax.axvline(THRESHOLD, color='purple', linestyle='-', linewidth=1.5, label=f"Corte ({THRESHOLD}) | {pct_acima:.0f}% >= corte")

    ax.set_title(f"{titulo} (n={len(sub)})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Similaridade de Cosseno", fontsize=10)
    ax.set_ylabel("Número de Entidades", fontsize=10)
    ax.set_xlim(-0.2, 0.95)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(alpha=0.3)

plt.savefig("figs/relevance_distribution_v5.png", dpi=150, bbox_inches='tight')
plt.show()

# 6. Top 5 e Bottom 5
print("\n--- TOP 5 MAIS SIMILARES ---")
for _, r in df_scores.head(5).iterrows():
    print(f"[{r['similarity']:.3f}] {r['word']} ({r['source']}) -> {r['description'][:80]}")

print("\n--- BOTTOM 5 MENOS SIMILARES (Removidas) ---")
for _, r in df_scores.tail(5).iterrows():
    print(f"[{r['similarity']:.3f}] {r['word']} ({r['source']}) -> {r['description'][:80]}")

# 7. Guardar o CSV final filtrado (todas as 6.191 frases prontas para o MLP)
entidades_por_frase = {}
for id_s, grupo in df_filtrado.groupby('id_sente'):
    entidades_por_frase[id_s] = [
        {
            'word': r['word'],
            'qid': r['qid'],
            'description': r['description'],
            'source': r['source'],
            'similarity': round(float(r['similarity']), 4),
            'start_char': r['start_char'],
            'end_char': r['end_char']
        }
        for _, r in grupo.iterrows()
    ]

df_final = df[['id_sente', 'classe', 'sentence']].copy()
df_final['entities'] = df_final['id_sente'].map(lambda x: entidades_por_frase.get(x, []))
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\nFicheiro final guardado com sucesso em: {OUTPUT_CSV} ({len(df_final)} frases)")