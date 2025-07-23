import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime.lime_tabular
import os

# === 1. Carregar o dataset ===
colunas = [
    'status_checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status_sex',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'number_existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('data/german.data', sep='\s+', header=None)
df.columns = colunas

# === 2. Pré-processamento ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# === 3. Separar features e target ===
X = df.drop('target', axis=1)
y = df['target']

# === 4. Dividir entre treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Treinar modelo Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Avaliação ===
y_pred = model.predict(X_test)
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# === 7. Aplicar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

idx = 0
instance = X_test.iloc[idx]
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# === 8. Gráfico PNG Amigável e Informativo ===
fig, ax = plt.subplots(figsize=(10, 6))
exp_list = exp.as_list()
features = [f.replace('_', ' ').title() for f, _ in exp_list]
weights = [w for _, w in exp_list]
colors = ['orange' if w > 0 else 'blue' for w in weights]

bars = ax.barh(features, weights, color=colors)
ax.set_title("Explicação Local para a Classe: Mau Pagador", fontsize=14)
ax.set_xlabel("Contribuição para a Decisão", fontsize=12)

# Rótulo das barras
for bar, w in zip(bars, weights):
    ax.text(bar.get_width() + 0.005 * np.sign(bar.get_width()),
            bar.get_y() + bar.get_height()/2,
            f'{w:.2f}',
            va='center', fontsize=9, fontweight='bold')

# Legenda
legenda = (
    "🟧 Laranja: Características que reforçam a decisão de negar o crédito\n"
    "🟦 Azul: Características que poderiam justificar aprovação"
)
props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
plt.text(1.05, -0.1, legenda, transform=ax.transAxes, fontsize=9, bbox=props, va='bottom')

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/lime_explanation_ptbr.png", bbox_inches='tight')
plt.close()

# === 9. HTML Explicativo Traduzido ===
html_intro = """
<div style="font-family: sans-serif; padding: 10px;">
  <h2>O que este gráfico mostra?</h2>
  <p>Este gráfico interativo explica por que o modelo classificou este cliente como <strong>"Mau Pagador"</strong>.</p>
  <ul>
    <li><span style="color: orange;">🟧 Laranja</span>: características que <strong>reforçaram a recusa</strong> de crédito.</li>
    <li><span style="color: blue;">🟦 Azul</span>: características que <strong>justificariam aprovação</strong> de crédito.</li>
  </ul>
  <p>Essas explicações promovem <strong>transparência</strong> e ajudam clientes, analistas e reguladores a entender as decisões do modelo.</p>
</div>
"""

with open("images/lime_explanation_ptbr.html", "w", encoding="utf-8") as f:
    f.write(html_intro)
    f.write(exp.as_html())

print("\n✅ Explicações salvas em: images/lime_explanation_ptbr.png e .html")
