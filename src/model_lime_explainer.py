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
fig.set_size_inches(14, 6)

plt.title("Explicação Local: Por que o modelo classificou como 'Mau Pagador'", fontsize=14)
plt.xlabel("Contribuição para a decisão", fontsize=12)

legenda = (
    "🔵 Azul: Características que reforçaram a decisão de negar o crédito.\n"
    "🟠 Laranja: Características que sugerem que o crédito poderia ser concedido."
)
plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='gray'))

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/lime_explanation_ptbr.png", bbox_inches='tight')
plt.close()

# === 5. Geração de frases explicativas automáticas ===
frases = []
for feature, weight in exp.as_list():
    if weight > 0:
        frases.append(f"O fator '{feature}' aumentou a chance de classificar o cliente como mau pagador.")
    else:
        frases.append(f"O fator '{feature}' ajudou a indicar que o cliente poderia ser um bom pagador.")

# === 6. Criar HTML explicativo com gráfico interativo e simulador ===
html_intro = """
<div style="font-family: Arial, sans-serif; padding: 20px;">
  <h2>📊 Explicação da Decisão do Modelo</h2>
  <p>O modelo classificou este cliente como <strong>'Mau Pagador'</strong>. Abaixo estão os principais fatores que influenciaram essa decisão.</p>
  <ul>
    <li><span style="color: orange;">🟠 Laranja</span>: fatores que sugerem possível aprovação.</li>
    <li><span style="color: blue;">🔵 Azul</span>: fatores que reforçaram a negativa.</li>
  </ul>
  <hr>
  <h3>🧾 Frases Explicativas:</h3>
  <ul>
"""

# Adicionar frases ao HTML
for frase in frases:
    html_intro += f"<li>{frase}</li>\n"

html_intro += """
  </ul>
  <hr>
  <h3>📝 Simule sua solicitação de crédito:</h3>
  <form id="formSimulador">
    <label>Idade: <input type="number" id="idade" required></label><br><br>
    <label>Valor do Crédito: <input type="number" id="valor" required></label><br><br>
    <label>Duração (meses): <input type="number" id="duracao" required></label><br><br>
    <button type="button" onclick="simular()">Ver Resultado</button>
  </form>
  <p id="resultadoSimulacao" style="font-weight: bold;"></p>

  <script>
    function simular() {
      const idade = parseInt(document.getElementById("idade").value);
      const valor = parseInt(document.getElementById("valor").value);
      const duracao = parseInt(document.getElementById("duracao").value);
      let resultado = "✅ Provavelmente o crédito seria APROVADO.";
      if (idade < 25 && valor > 5000 || duracao > 36) {
        resultado = "❌ Provavelmente o crédito seria NEGADO.";
      }
      document.getElementById("resultadoSimulacao").innerText = resultado;
    }
  </script>

  <hr>
  <h3>📈 Gráfico Interativo:</h3>
</div>
"""

# Salvar HTML completo
html_path = "images/lime_explanation_ptbr.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_intro)
    f.write(exp.as_html())  # gráfico interativo incluído

print("✅ Gráfico PNG salvo em: images/lime_explanation_ptbr.png")
print("✅ HTML gerado com gráfico, frases e simulador: images/lime_explanation_ptbr.html")
