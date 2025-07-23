import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular
import os

# === 1. Carregar e preparar os dados ===
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outras_parcelas',
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone', 'trabalhador_estrangeiro',
    'alvo'
]

df = pd.read_csv('data/german.data', sep=' ', header=None)
df.columns = colunas

# Codificação de variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar variáveis
X = df.drop('alvo', axis=1)
y = df['alvo']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar o modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação
print("Relatório de Classificação:\n")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === 3. Aplicar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# Selecionar uma instância
i = 0
instance = X_test.iloc[i]
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# === 4. Gerar gráfico PNG (com largura estendida e em português) ===
fig = exp.as_pyplot_figure(label=1)
fig.set_size_inches(12, 6)  # largura aumentada

plt.title("Explicação Local: Por que o modelo classificou como 'Mau Pagador'", fontsize=14)
plt.xlabel("Contribuição para a decisão", fontsize=12)

# Corrigir a legenda
legenda = (
    "🟧 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
    "🟦 Azul: Características que sugerem que o crédito poderia ser concedido."
)
plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='gray'))

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/lime_explanation_ptbr.png", bbox_inches='tight')
plt.close()

# === 5. Gerar HTML explicativo em PT-BR com gráfico embutido ===
html_intro = """
<div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f9f9f9;">
  <h2 style="color: #111;"><img src="https://img.icons8.com/color/48/ai.png" style="vertical-align: middle;"> O que este gráfico mostra?</h2>
  <p style="font-size: 16px;">Este gráfico explica de forma visual por que o modelo de IA classificou este cliente como <strong>Mau Pagador</strong>.</p>
  <ul style="font-size: 15px;">
    <li><span style="color: orange; font-weight: bold;">🟧 Laranja</span>: fatores que <strong>reforçaram a decisão</strong> de negar o crédito.</li>
    <li><span style="color: blue; font-weight: bold;">🟦 Azul</span>: fatores que <strong>apontam possibilidade</strong> de concessão do crédito.</li>
  </ul>
  <p style="font-size: 15px;">Esta explicação ajuda clientes, gerentes e reguladores a entenderem como a decisão foi tomada, promovendo <strong>transparência</strong> e responsabilidade no uso da inteligência artificial.</p>
  <hr>
  <h3 style="color: #c2185b;">📌 Informações detalhadas:</h3>
</div>
"""

html_path = "images/lime_explanation_ptbr.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_intro)
    f.write(exp.as_html())  # adiciona o gráfico interativo

print("✅ Gráfico salvo em 'images/lime_explanation_ptbr.png'")
print("✅ HTML completo salvo em 'images/lime_explanation_ptbr.html'")
