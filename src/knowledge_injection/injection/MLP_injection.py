'''
demorou 21h20min
=======================================================
RELATÓRIO DE CLASSIFICAÇÃO (BERTimbau + MLP Conhecimento)
=======================================================
              precision    recall  f1-score   support

Citação (-1)       0.94      0.90      0.92       417
   Facto (0)       0.93      0.93      0.93      1273
    Viés (1)       0.56      0.57      0.56       168

    accuracy                           0.90      1858
   macro avg       0.81      0.80      0.81      1858
weighted avg       0.90      0.90      0.90      1858
'''

import ast
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. Configuração e Caminhos ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A correr em: {device}")

# Caminho relativo a partir da pasta 'injection/' (ajusta se correres no Colab)
INPUT_CSV = Path("/content/drive/MyDrive/FactNews_Project/final_extracted_entities_v5.csv")
CACHE_VETORES = Path("vetores_conhecimento_v5.pt")
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# --- 2. Carregar Dados ---
df = pd.read_csv(INPUT_CSV)
print(f"Dataset carregado: {len(df)} frases.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 3. Extrair / Carregar Vetores de Conhecimento (.pt) ---
def extrair_vetores_conhecimento(df_dados):
    extrator = BertModel.from_pretrained(MODEL_NAME).to(device)
    extrator.eval()
    vetores = []

    for celula in tqdm(df_dados["entities"], desc="A extrair vetores de conhecimento"):
        lista_ents = ast.literal_eval(celula) if isinstance(celula, str) else (celula if pd.notna(celula) else [])
        descricoes = [item["description"] for item in lista_ents if item.get("description")]

        if not descricoes:
            vetores.append(torch.zeros(768, dtype=torch.float))
            continue

        texto_conhecimento = " ; ".join(descricoes)
        inputs = tokenizer(
            texto_conhecimento,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = extrator(**inputs)
            vetor_cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        vetores.append(vetor_cls)

    del extrator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vetores

if CACHE_VETORES.exists():
    print(f"A carregar vetores pré-calculados de {CACHE_VETORES}...")
    vetores_conhecimento = torch.load(CACHE_VETORES)
else:
    vetores_conhecimento = extrair_vetores_conhecimento(df)
    torch.save(vetores_conhecimento, CACHE_VETORES)
    print(f"Vetores guardados em {CACHE_VETORES}!")

# --- 4. Divisão Treino / Teste (70/30 Estratificada = 1.858 frases de teste) ---
indices = np.arange(len(df))
idx_train, idx_test = train_test_split(
    indices, test_size=0.30, stratify=df["classe"], random_state=SEED
)

df_train = df.iloc[idx_train].reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)
vet_train = [vetores_conhecimento[i] for i in idx_train]
vet_test = [vetores_conhecimento[i] for i in idx_test]

print(f"Treino: {len(df_train)} frases | Teste: {len(df_test)} frases")

# --- 5. Dataset PyTorch ---
class FactNewsMLPDataset(Dataset):
    def __init__(self, df_sub, vetores, tokenizer, max_len=128):
        self.df = df_sub
        self.vetores = vetores
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mapa_labels = {-1: 0, 0: 1, 1: 2}
        self.labels = [self.mapa_labels[lbl] for lbl in self.df["classe"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frase = str(self.df.iloc[idx]["sentence"])
        inputs = self.tokenizer(
            frase,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "vetor_conhecimento": self.vetores[idx],
            "classe": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- 6. Arquitetura: BERTimbau + Fusão Tardia com MLP ---
class BERTimbauComMLP(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, vetor_conhecimento):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_frase = outputs.last_hidden_state[:, 0, :]
        vetor_fundido = torch.cat((cls_frase, vetor_conhecimento), dim=1)
        return self.mlp(vetor_fundido)

# --- 7. Treino e Avaliação ---
if __name__ == "__main__":
    train_loader = DataLoader(FactNewsMLPDataset(df_train, vet_train, tokenizer), batch_size=16, shuffle=True)
    test_loader = DataLoader(FactNewsMLPDataset(df_test, vet_test, tokenizer), batch_size=32, shuffle=False)

    modelo = BERTimbauComMLP().to(device)
    otimizador = AdamW(modelo.parameters(), lr=2e-5, weight_decay=0.01)
    criterio = nn.CrossEntropyLoss()

    epocas = 4
    for epoca in range(epocas):
        modelo.train()
        loss_total = 0.0
        for lote in tqdm(train_loader, desc=f"Época {epoca + 1}/{epocas}"):
            otimizador.zero_grad()
            logits = modelo(
                lote["input_ids"].to(device),
                lote["attention_mask"].to(device),
                lote["vetor_conhecimento"].to(device)
            )
            loss = criterio(logits, lote["classe"].to(device))
            loss.backward()
            otimizador.step()
            loss_total += loss.item()

        print(f"Época {epoca + 1}/{epocas} | Loss Média: {loss_total / len(train_loader):.4f}")

    # Guardar modelo
    torch.save(modelo.state_dict(), "bertimbau_mlp_factnews_v5.pt")

    # Avaliação no Conjunto de Teste
    modelo.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for lote in test_loader:
            logits = modelo(
                lote["input_ids"].to(device),
                lote["attention_mask"].to(device),
                lote["vetor_conhecimento"].to(device)
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(lote["classe"].numpy())

    # Relatório de Classificação (igual à Meta 2)
    nomes_classes = ["Citação (-1)", "Facto (0)", "Viés (1)"]
    print("\n" + "="*55)
    print("RELATÓRIO DE CLASSIFICAÇÃO (BERTimbau + MLP Conhecimento)")
    print("="*55)
    print(classification_report(y_true, y_pred, target_names=nomes_classes, digits=2))

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["-1", "0", "1"], yticklabels=["-1", "0", "1"]
    )
    plt.title("Matriz de Confusão — BERTimbau + MLP (Com Conhecimento)")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Verdadeira")
    plt.tight_layout()
    plt.savefig("confusion_matrix_mlp_v5.png", dpi=150)
    plt.show()