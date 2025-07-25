import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime.lime_tabular

# Carregar dados
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outras_parcelas',
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone', 'trabalhador_estrangeiro',
    'alvo'
]
df = pd.read_csv("data/german.data", sep=" ", header=None)
df.columns = colunas

# Codificar variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar X e y
X = df.drop("alvo", axis=1)
y = df["alvo"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar
print("Relatório de Classificação:")
print(classification_report(y_test, model.predict(X_test)))

# Aplicar LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=["Bom Pagador", "Mau Pagador"],
    mode="classification"
)

# Selecionar um bom e um mau pagador
bom_idx = y_test[y_test == 1].index[0]
mau_idx = y_test[y_test == 2].index[0]

exemplos = {
    "bom_pagador": (bom_idx, "Bom Pagador"),
    "mau_pagador": (mau_idx, "Mau Pagador")
}

# Criar pastas
os.makedirs("images", exist_ok=True)

# HTML inicial
html = """
<html><head><meta charset="utf-8"><title>Explicação de Crédito</title></head>
<body style="font-family: Arial, sans-serif; background:#f9f9f9; padding: 30px;">
<h2>📌 Explicação da decisão do modelo</h2>
<p>Este modelo de IA foi treinado para prever se um cliente será <strong>Bom Pagador</strong> ou <strong>Mau Pagador</strong>.</p>
<hr>
"""

for nome, (idx, classe) in exemplos.items():
    instance = X_test.loc[idx]
    exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)
    
    # Gráfico PNG
    fig = exp.as_pyplot_figure(label=1 if classe == "Mau Pagador" else 0)
    fig.set_size_inches(14, 6)
    plt.title(f"Explicação Local: {classe}", fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)
    legenda = (
        "🔵 Azul: Fatores que reforçaram a decisão de <strong>negar</strong> o crédito.<br>"
        "🟠 Laranja: Fatores que <strong>sugerem aprovação</strong> do crédito."
    )
    plt.figtext(0.99, 0.01, "Legenda: " + legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))
    plt.tight_layout()
    img_path = f"images/{nome}.png"
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()

    # Frases explicativas
    frases = []
    for feat, weight in exp.as_list():
        if weight > 0:
            frases.append(f"🟠 O fator <strong>{feat}</strong> aumentou a chance de ser classificado como <strong>mau pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feat}</strong> indicou chance de ser <strong>bom pagador</strong>.")

    # Adicionar ao HTML
    html += f"<h3>📊 {classe}</h3>"
    html += exp.as_html()
    html += "<h4>🧾 Explicações em linguagem simples:</h4><ul>"
    for frase in frases:
        html += f"<li>{frase}</li>"
    html += "</ul><hr>"

# Simulador interativo
html += """
<h2>📝 Simule sua solicitação de crédito:</h2>
<form id="formSimulador">
  <label>Idade: <input type="number" id="idade" required></label><br><br>
  <label>Valor do Crédito: <input type="number" id="valor" required></label><br><br>
  <label>Duração (meses): <input type="number" id="duracao" required></label><br><br>
  <label>Está empregado?
    <select id="empregado">
      <option value="sim">Sim</option>
      <option value="nao">Não</option>
    </select>
  </label><br><br>
  <label>Tempo empregado (meses): <input type="number" id="emprego_tempo"></label><br><br>
  <label>Quantas pessoas moram com você? <input type="number" id="moradores"></label><br><br>
  <label>Renda mensal (R$): <input type="number" id="renda"></label><br><br>
  <button type="button" onclick="simular()">Ver Resultado</button>
</form>
<p id="resultadoSimulacao" style="font-weight: bold; font-size: 16px; color: #333;"></p>

<script>
function simular() {
  const idade = +document.getElementById("idade").value;
  const valor = +document.getElementById("valor").value;
  const duracao = +document.getElementById("duracao").value;
  const empregado = document.getElementById("empregado").value;
  const emprego_tempo = +document.getElementById("emprego_tempo").value;
  const moradores = +document.getElementById("moradores").value;
  const renda = +document.getElementById("renda").value;
  
  let msgs = [];
  let aprovado = true;

  if (idade < 21) { msgs.push("❌ Idade abaixo de 21 anos."); aprovado = false; }
  else { msgs.push("✅ Idade aceitável."); }

  if (empregado === "nao") { msgs.push("❌ Desempregado."); aprovado = false; }
  else {
    msgs.push("✅ Está empregado.");
    if (emprego_tempo < 6) { msgs.push("❌ Menos de 6 meses de trabalho."); aprovado = false; }
    else { msgs.push("✅ Tempo de trabalho suficiente."); }
  }

  if (moradores > 3) { msgs.push("❌ Muitas pessoas morando junto."); aprovado = false; }
  else { msgs.push("✅ Quantidade de moradores adequada."); }

  let parcela = valor / duracao;
  if (parcela > renda * 0.5) {
    msgs.push(`❌ Parcela (~R$${parcela.toFixed(2)}) alta em relação à renda.`);
    aprovado = false;
  } else {
    msgs.push("✅ Renda adequada para o valor e prazo.");
  }

  const resultado = aprovado
    ? "✅ Provavelmente o crédito seria APROVADO."
    : "❌ Provavelmente o crédito seria NEGADO.";

  document.getElementById("resultadoSimulacao").innerHTML =
    "<p>" + resultado + "</p><ul><li>" + msgs.join("</li><li>") + "</li></ul>";
}
</script>
</body></html>
"""

# Salvar HTML
with open("images/lime_explanation_ptbr.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Gráficos e HTML gerados com sucesso!")
