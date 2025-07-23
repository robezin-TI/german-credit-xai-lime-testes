import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# === 1. Carregar dados ===
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outros_planos',
    'habitação', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone',
    'trabalhador_estrangeiro', 'target'
]

df = pd.read_csv('data/german.data', sep=' ', header=None)
df.columns = colunas

# === 2. Pré-processamento ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Ajustar target para 0 = bom pagador, 1 = mau pagador
df['target'] = df['target'].map({1: 1, 2: 0})

# === 3. Separar features e target ===
X = df.drop('target', axis=1)
y = df['target']

# === 4. Treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Avaliação ===
print("Relatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# === 7. LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

idx = 0
instance = X_test.iloc[idx]
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# === 8. Gráfico aprimorado ===
os.makedirs('images', exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 7))

exp_list = exp.as_list()
features = [x[0] for x in exp_list]
weights = [x[1] for x in exp_list]

colors = ['orange' if val > 0 else 'blue' for val in weights]

bars = ax.barh(features, weights, color=colors)
ax.set_title("Explicação Local: Por que o modelo classificou como 'Mau Pagador'", fontsize=14)
ax.set_xlabel("Contribuição para a decisão", fontsize=12)

for bar, val in zip(bars, weights):
    ax.text(bar.get_width() + 0.005 * np.sign(bar.get_width()),
            bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va='center', fontweight='bold')

# Legenda
legenda = (
    "🟧 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
    "🟦 Azul: Características que sugerem que o crédito poderia ser concedido."
)
props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
plt.text(1.05, -0.1, legenda, transform=ax.transAxes, fontsize=9, bbox=props)

plt.tight_layout()
plt.savefig("images/lime_explanation_ptbr.png", bbox_inches='tight')
plt.close()

# === 9. HTML aprimorado ===
html_intro = """
<div style="font-family: sans-serif; padding: 20px; background: #f9f9f9;">
  <h2 style="color:#111;"><img src="https://img.icons8.com/color/24/graph.png"/> O que este gráfico mostra?</h2>
  <p>Este gráfico explica de forma visual por que o modelo de IA classificou este cliente como <strong>Mau Pagador</strong>.</p>
  <ul>
    <li>🟧 <strong>Laranja</strong>: fatores que <strong>reforçaram a decisão</strong> de negar o crédito.</li>
    <li>🟦 <strong>Azul</strong>: fatores que <strong>apontam possibilidade</strong> de concessão do crédito.</li>
  </ul>
  <p>Esta explicação ajuda clientes, gerentes e reguladores a entenderem como a decisão foi tomada, promovendo <strong>transparência</strong> e responsabilidade no uso da inteligência artificial.</p>
  <hr/>
  <h3>📌 Informações detalhadas:</h3>
</div>
"""

# Explicação LIME em HTML traduzido
lime_html = exp.as_html().replace("Prediction probabilities", "Probabilidades de Classificação")
lime_html = lime_html.replace("Feature", "Variável").replace("Value", "Valor")
lime_html = lime_html.replace("Good", "Bom Pagador").replace("Bad", "Mau Pagador")

html_completo = html_intro + lime_html

with open("images/lime_explanation_ptbr.html", "w", encoding="utf-8") as f:
    f.write(html_completo)

print("✅ Gráfico e HTML salvos em: pasta 'images/'")
