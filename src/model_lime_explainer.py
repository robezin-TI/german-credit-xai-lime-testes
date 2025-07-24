import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

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

# Codificar variáveis categóricas
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

# Selecionar instância para explicação
idx = 0
instance = X_test.iloc[idx]
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# === 4. Gráfico PNG (traduzido e largo) ===
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

# === 5. Frases explicativas automáticas ===
frases = []
for feature, weight in exp.as_list():
    if weight > 0:
        frases.append(f"O fator <strong>{feature}</strong> contribuiu para considerar o cliente como <strong>Mau Pagador</strong>.")
    else:
        frases.append(f"O fator <strong>{feature}</strong> indicou características de <strong>Bom Pagador</strong>.")

# === 6. HTML explicativo com gráfico + simulador + frases ===
html_intro = """
<div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f9f9f9;">
  <h2 style="color: #2e7d32;">📊 Explicação da Decisão do Modelo</h2>
  <p>O modelo classificou este cliente como <strong>'Mau Pagador'</strong>. Abaixo estão os principais fatores que influenciaram essa decisão.</p>
  <ul>
    <li><span style="color: orange;">🟠 Laranja</span>: fatores que sugerem possível aprovação.</li>
    <li><span style="color: blue;">🔵 Azul</span>: fatores que reforçaram a negativa.</li>
  </ul>
  <hr>
  <h3>🧾 Frases Explicativas:</h3>
  <ul>
"""

for frase in frases:
    html_intro += f"<li>{frase}</li>\n"

html_intro += """
  </ul>
  <hr>
  <h3>📝 Simule sua solicitação de crédito:</h3>
  <form id="formSimulador">
    <label>Idade: <input type="number" id="idade" required></label><br><br>
    <label>Valor do Crédito (R$): <input type="number" id="valor" required></label><br><br>
    <label>Duração (meses): <input type="number" id="duracao" required></label><br><br>
    <label>Está empregado? 
        <input type="radio" name="empregado" value="sim" checked> Sim
        <input type="radio" name="empregado" value="nao"> Não
    </label><br><br>
    <label>Tempo de emprego (em meses): <input type="number" id="tempo_emprego" required></label><br><br>
    <label>Quantas pessoas moram com você? <input type="number" id="pessoas" required></label><br><br>
    <label>Renda mensal (R$): <input type="number" id="renda" required></label><br><br>
    <button type="button" onclick="simular()">Ver Resultado</button>
  </form>

  <p id="resultadoSimulacao" style="font-weight: bold;"></p>
  <div id="explicacaoDetalhada" style="margin-top: 20px;"></div>

  <script>
    function simular() {
      const idade = parseInt(document.getElementById("idade").value);
      const valor = parseInt(document.getElementById("valor").value);
      const duracao = parseInt(document.getElementById("duracao").value);
      const empregado = document.querySelector('input[name="empregado"]:checked').value;
      const tempo_emprego = parseInt(document.getElementById("tempo_emprego").value);
      const pessoas = parseInt(document.getElementById("pessoas").value);
      const renda = parseInt(document.getElementById("renda").value);

      let aprovado = true;
      let explicacao = "<h4>📌 Fatores que influenciaram:</h4><ul>";

      if (idade < 21) {
        aprovado = false;
        explicacao += "<li>Idade muito baixa pode indicar risco.</li>";
      } else {
        explicacao += "<li>Idade considerada adequada.</li>";
      }

      if (duracao < 3) {
        aprovado = false;
        explicacao += "<li>Duração de pagamento muito curta pode indicar inadimplência.</li>";
      } else {
        explicacao += "<li>Duração dentro do aceitável.</li>";
      }

      if (empregado === "nao") {
        aprovado = false;
        explicacao += "<li>Não estar empregado é um fator negativo.</li>";
      } else {
        explicacao += "<li>Estar empregado é um fator positivo.</li>";
      }

      if (tempo_emprego < 6) {
        aprovado = false;
        explicacao += "<li>Tempo de emprego inferior a 6 meses indica instabilidade.</li>";
      } else {
        explicacao += "<li>Tempo de emprego estável.</li>";
      }

      if (pessoas > 3) {
        aprovado = false;
        explicacao += "<li>Muitas pessoas na residência podem indicar sobrecarga financeira.</li>";
      } else {
        explicacao += "<li>Quantidade de pessoas aceitável.</li>";
      }

      const valor_parcela = valor / duracao;
      if (renda < valor_parcela * 1.5) {
        aprovado = false;
        explicacao += "<li>Renda insuficiente para o valor da parcela.</li>";
      } else {
        explicacao += "<li>Renda compatível com as parcelas.</li>";
      }

      explicacao += "</ul>";

      document.getElementById("resultadoSimulacao").innerText = aprovado
        ? "✅ Provavelmente o crédito seria APROVADO."
        : "❌ Provavelmente o crédito seria NEGADO.";

      document.getElementById("explicacaoDetalhada").innerHTML = explicacao;
    }
  </script>

  <hr>
  <h3>📈 Gráfico Interativo:</h3>
</div>
"""

# Salvar HTML final
html_path = "images/lime_explanation_ptbr.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_intro)
    f.write(exp.as_html())

print("✅ Gráfico PNG salvo em 'images/lime_explanation_ptbr.png'")
print("✅ HTML completo gerado em 'images/lime_explanation_ptbr.html'")
