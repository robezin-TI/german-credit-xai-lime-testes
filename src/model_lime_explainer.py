import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lime.lime_tabular import LimeTabularExplainer
import os

# === Carregar os dados ===
df = pd.read_csv("data/german.data", sep='\s+', header=None)

# === Atribuir nomes às colunas ===
df.columns = [
    "status_checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age", "other_installment_plans",
    "housing", "number_credits", "job", "people_liable", "foreign_worker", "target",
    "unused_col"  # coluna extra que será descartada
]

# === Remover coluna extra se não for útil ===
df = df.drop("unused_col", axis=1)

# === Ajustar o target para 0 (bom pagador) e 1 (mau pagador) ===
df["target"] = df["target"].map({1: 1, 2: 0})

# === Pré-processamento ===
categorical_cols = df.select_dtypes(include=["object"]).columns

# One-hot encoding para variáveis categóricas
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separar features e target
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliação
print("Relatório de Classificação:\n")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === Aplicar LIME ===
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=["Bom Pagador", "Mau Pagador"],
    mode="classification"
)

# Explicar uma instância específica
idx = 0  # pode variar
exp = explainer.explain_instance(X_test.iloc[idx], model.predict_proba, num_features=10)

# === Melhorar visualização com gráfico .png ===
fig, ax = plt.subplots(figsize=(10, 6))

exp_list = exp.as_list()
features = [x[0] for x in exp_list]
weights = [x[1] for x in exp_list]
colors = ['green' if val > 0 else 'red' for val in weights]

# Plot
bars = ax.barh(features, weights, color=colors)
ax.set_title("Explicação Local para a Classe: Mau Pagador", fontsize=14)
ax.set_xlabel("Impacto na Predição", fontsize=12)

# Adicionar valores nas barras
for bar, val in zip(bars, weights):
    ax.text(bar.get_width() + 0.005 * np.sign(bar.get_width()),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va='center', fontsize=9, fontweight='bold')

# Legenda explicativa
legend_text = (
    "\n🟩 Verde: Características que reforçam a decisão de recusar o crédito.\n"
    "🟥 Vermelho: Características que poderiam justificar aprovação."
)
props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
plt.text(1.02, -0.1, legend_text, transform=ax.transAxes,
         fontsize=9, bbox=props, verticalalignment='bottom')

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/lime_explanation_friendly.png")
plt.close()

# === HTML explicativo em PT-BR ===
html_explication = """
<div style="font-family: sans-serif; padding: 10px; background: #f9f9f9;">
  <h2>O que este gráfico mostra?</h2>
  <p>Este gráfico interativo explica por que o modelo classificou este cliente como <strong>\"Mau Pagador\"</strong>.</p>
  <ul>
    <li><span style="color: green; font-weight: bold;">🟩 Verde</span>: características que influenciaram <strong>negativamente</strong> (reforçaram a decisão de <strong>negar</strong> o crédito).</li>
    <li><span style="color: red; font-weight: bold;">🟥 Vermelho</span>: características que influenciaram <strong>positivamente</strong> (indicaram que o cliente <strong>poderia receber</strong> o crédito).</li>
  </ul>
  <p>Esta explicação ajuda clientes, analistas e reguladores a entenderem como a decisão foi tomada, promovendo <strong>transparência</strong> no uso da inteligência artificial.</p>
</div>
"""

with open("images/lime_explanation_friendly.html", "w", encoding="utf-8") as f:
    f.write(html_explication)
    f.write(exp.as_html())

print("Explicação salva como imagem e HTML na pasta 'images/'.")
