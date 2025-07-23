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

# 1. Carregar dados
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outros_planos',
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone', 'trabalhador_estrangeiro',
    'alvo'
]
df = pd.read_csv("data/german.data", sep='\s+', header=None)
df.columns = colunas

# 2. Pré-processamento
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop("alvo", axis=1)
y = df["alvo"]

# 3. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5. Avaliação
print("Relatório de Classificação:\n")
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Aplicar LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# Escolher uma instância para explicação
i = 0
instance = X_test.iloc[i]
exp = explainer.explain_instance(instance.to_numpy(), modelo.predict_proba, num_features=10)

# 7. Gráfico Explicativo com melhorias
fig, ax = plt.subplots(figsize=(12, 7))  # Aumentado
exp_list = exp.as_list()
features = [x[0] for x in exp_list]
weights = [x[1] for x in exp_list]
colors = ['orange' if val > 0 else 'blue' for val in weights]  # Laranja reforça Mau Pagador, Azul sugere Bom Pagador

bars = ax.barh(features, weights, color=colors)
ax.set_title("Explicação Local: Por que o modelo classificou como 'Mau Pagador'", fontsize=14)
ax.set_xlabel("Contribuição para a decisão", fontsize=12)

# Adicionar os valores nas barras
for bar, val in zip(bars, weights):
    ax.text(bar.get_width() + 0.01 * np.sign(bar.get_width()),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va='center', fontsize=9, fontweight='bold')

# Legenda explicativa
legend_text = (
    "🟧 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
    "🟦 Azul: Características que sugerem que o crédito poderia ser concedido."
)
props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
plt.text(1.02, -0.1, legend_text, transform=ax.transAxes,
         fontsize=9, bbox=props, verticalalignment='bottom')

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/lime_explicacao_amigavel.png", bbox_inches='tight')
plt.close()

# 8. HTML traduzido e melhorado
html_explicacao = """
<div style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 20px;">
  <h2>📊 O que este gráfico mostra?</h2>
  <p>Este gráfico explica de forma visual por que o modelo de IA classificou este cliente como <strong>Mau Pagador</strong>.</p>

  <ul>
    <li><span style="color: orange; font-weight: bold;">🟧 Laranja</span>: fatores que <strong>reforçaram a decisão</strong> de negar o crédito.</li>
    <li><span style="color: #1f77b4; font-weight: bold;">🟦 Azul</span>: fatores que <strong>apontam possibilidade</strong> de concessão do crédito.</li>
  </ul>

  <p>Esta explicação ajuda clientes, gerentes e reguladores a entenderem como a decisão foi tomada, promovendo <strong>transparência</strong> e responsabilidade no uso da inteligência artificial.</p>

  <hr style="margin: 20px 0;">

  <h3>📌 Informações detalhadas:</h3>
"""

# Anexar explicação interativa original com tradução
exp_html = exp.as_html().replace("Feature", "Característica").replace("Value", "Valor")
exp_html = exp_html.replace("Prediction probabilities", "Probabilidades de Classificação")
exp_html = exp_html.replace("Good", "Bom Pagador").replace("Bad", "Mau Pagador")
html_explicacao += exp_html
html_explicacao += "</div>"

with open("images/lime_explicacao_amigavel.html", "w", encoding="utf-8") as f:
    f.write(html_explicacao)

print("✅ Explicações salvas em 'images/lime_explicacao_amigavel.png' e 'images/lime_explicacao_amigavel.html'")
