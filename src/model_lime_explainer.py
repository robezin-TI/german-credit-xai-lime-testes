import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import lime
import lime.lime_tabular

# === 1. Carregar e nomear colunas ===
colunas = [
    'status_checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status_sex',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'number_existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('data/german.data', sep=' ', header=None)
df.columns = colunas

# === 2. Pré-processar atributos categóricos com LabelEncoder ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# === 3. Separar atributos e alvo ===
X = df.drop('target', axis=1)
y = df['target']

# === 4. Dividir em treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Treinar modelo Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Avaliar desempenho ===
y_pred = model.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# === 7. Criar explicação com LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# Explicar a primeira instância de teste
i = 0
instance = X_test.iloc[i]
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# Mostrar explicação no terminal
print(f"\nExplicação da instância {i}:\n")
print(exp.as_list())

# === 8. Gerar gráfico explicativo em português ===
features, values = zip(*exp.as_list())
colors = ['green' if val > 0 else 'red' for val in values]

plt.figure(figsize=(10, 6))
bars = plt.barh(features, values, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Explicação Local para a Classe: Mau Pagador", fontsize=14)
plt.xlabel("Impacto na decisão do modelo", fontsize=11)

for bar in bars:
    plt.text(
        bar.get_width() + 0.005 if bar.get_width() > 0 else bar.get_width() - 0.06,
        bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.2f}',
        va='center',
        color='black',
        fontsize=9,
        fontweight='bold'
    )

plt.figtext(0.99, 0.01,
    "🟩 Verde: Características que reforçam a decisão de recusar o crédito.\n"
    "🟥 Vermelho: Características que poderiam justificar aprovação.",
    horizontalalignment='right',
    fontsize=9,
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig('images/lime_explanation_friendly.png')
plt.close()

# === 9. Gerar HTML interativo com explicação para leigos ===
explicacao_html_extra = """
<div style="font-family: sans-serif; padding: 10px; border: 1px solid #ccc; background: #f9f9f9; margin-bottom: 10px;">
  <h2>O que este gráfico mostra?</h2>
  <p>Este gráfico interativo explica por que o modelo classificou este cliente como <strong>"Mau Pagador"</strong>.</p>
  <ul>
    <li><span style="color: green; font-weight: bold;">Verde</span>: características que influenciaram <strong>negativamente</strong> (reforçaram a decisão de <strong>negar</strong> o crédito).</li>
    <li><span style="color: red; font-weight: bold;">Vermelho</span>: características que influenciaram <strong>positivamente</strong> (indicaram que o cliente <strong>poderia receber</strong> o crédito).</li>
  </ul>
  <p>Esta explicação ajuda clientes, analistas e reguladores a entenderem como a decisão foi tomada, promovendo <strong>transparência</strong> no uso da inteligência artificial.</p>
</div>
"""

with open("images/lime_explanation_friendly.html", "w", encoding="utf-8") as f:
    f.write(explicacao_html_extra)
    f.write(exp.as_html())

print("\nExplicações salvas na pasta 'images/' com sucesso!")
