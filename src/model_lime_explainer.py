import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular

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

# === 5. Gerar frases explicativas baseadas nos pesos do LIME ===
frases = []
for feature, weight in exp.as_list():
    if weight > 0:
        frases.append(f"🟠 O fator <strong>{feature}</strong> aumentou a chance de classificar o cliente como <strong>mau pagador</strong>.")
    else:
        frases.append(f"🔵 O fator <strong>{feature}</strong> indicou que o cliente pode ser um <strong>bom pagador</strong>.")

# === 6. Criar HTML com todos os blocos ===
html_path = "images/lime_explanation_ptbr.html"
with open(html_path, "w", encoding="utf-8") as f:
    # 1. Explicação da decisão
    f.write("""
    <html><head><meta charset="utf-8"></head><body style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 30px;">
    <h2>📌 Explicação da decisão do modelo</h2>
    <p>O modelo classificou este cliente como <strong>Mau Pagador</strong>. Abaixo estão os principais fatores que influenciaram essa decisão.</p>
    <ul>
        <li><span style="color: blue;">🔵 Azul</span>: Fatores que <strong>reforçaram a decisão de negar</strong> o crédito.</li>
        <li><span style="color: orange;">🟠 Laranja</span>: Fatores que <strong>poderiam indicar aprovação</strong> do crédito.</li>
    </ul>
    <hr>
    """)

    # 2. Gráfico interativo
    f.write("<h3>📊 Gráfico Interativo:</h3>")
    f.write(exp.as_html())
    f.write("<hr>")

    # 3. Frases explicativas
    f.write("<h3>🧾 Explicações em linguagem simples:</h3><ul>")
    for frase in frases:
        f.write(f"<li style='margin-bottom:8px;'>{frase}</li>")
    f.write("</ul><hr>")

    # 4. Simulador de crédito
    f.write("""
    <h3>📝 Simule sua solicitação de crédito:</h3>
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
        <label>Empregado desde quando (em meses): <input type="number" id="emprego_tempo" required></label><br><br>
        <label>Quantas pessoas moram com você? <input type="number" id="moradores" required></label><br><br>
        <label>Renda mensal (R$): <input type="number" id="renda" required></label><br><br>
        <button type="button" onclick="simular()">Ver Resultado</button>
    </form>
    <p id="resultadoSimulacao" style="font-weight: bold; font-size: 16px; color: #333; margin-top: 20px;"></p>

    <script>
    function simular() {
        const idade = parseInt(document.getElementById("idade").value);
        const valor = parseInt(document.getElementById("valor").value);
        const duracao = parseInt(document.getElementById("duracao").value);
        const empregado = document.getElementById("empregado").value;
        const emprego_tempo = parseInt(document.getElementById("emprego_tempo").value);
        const moradores = parseInt(document.getElementById("moradores").value);
        const renda = parseInt(document.getElementById("renda").value);

        let mensagens = [];
        let aprovado = true;

        if (idade < 21) {
            mensagens.push("❌ Idade abaixo de 21 anos pode dificultar a aprovação.");
            aprovado = false;
        } else {
            mensagens.push("✅ Idade adequada.");
        }

        if (empregado === "nao") {
            mensagens.push("❌ Estar desempregado reduz a chance de aprovação.");
            aprovado = false;
        } else {
            mensagens.push("✅ Está empregado.");
            if (emprego_tempo < 6) {
                mensagens.push("❌ Menos de 6 meses de trabalho atual pode ser um fator negativo.");
                aprovado = false;
            } else {
                mensagens.push("✅ Tempo de emprego satisfatório.");
            }
        }

        if (moradores > 3) {
            mensagens.push("❌ Muitas pessoas no domicílio podem indicar maior comprometimento de renda.");
            aprovado = false;
        } else {
            mensagens.push("✅ Número de moradores adequado.");
        }

        let parcela = valor / duracao;
        if (parcela > renda * 0.5) {
            mensagens.push("❌ Renda mensal insuficiente para a parcela estimada (~R$" + parcela.toFixed(2) + ").");
            aprovado = false;
        } else {
            mensagens.push("✅ Renda condizente com o valor e prazo do crédito.");
        }

        const resultado = aprovado ?
          "✅ Provavelmente o crédito seria APROVADO." :
          "❌ Provavelmente o crédito seria NEGADO.";

        document.getElementById("resultadoSimulacao").innerHTML =
          "<p style='font-size:18px;'>" + resultado + "</p><ul><li>" + mensagens.join("</li><li>") + "</li></ul>";
    }
    </script>
    """)

    f.write("</body></html>")

# Mensagens de conclusão
print("✅ Gráfico PNG salvo em: images/lime_explanation_ptbr.png")
print("✅ HTML completo gerado em: images/lime_explanation_ptbr.html")
