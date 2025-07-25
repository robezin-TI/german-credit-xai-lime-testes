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

# Codificar variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar features e alvo
X = df.drop('alvo', axis=1)
y = df['alvo']

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar o modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Relatório
print("Relatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# === 3. Inicializar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# === 4. Função auxiliar para gerar gráfico e explicações ===
def gerar_explicacao(instancia, nome_arquivo, titulo_decisao):
    probs = model.predict_proba([instancia])[0]
    label_predito = np.argmax(probs)

    exp = explainer.explain_instance(instancia.to_numpy(), model.predict_proba, num_features=10)
    
    # PNG
    fig = exp.as_pyplot_figure(label=label_predito)
    fig.set_size_inches(14, 6)
    plt.title(titulo_decisao, fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)
    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
        "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))
    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases
    frases = []
    for feature, peso in exp.as_list():
        if peso > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> contribuiu para classificar como <strong>Mau Pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> indicou tendência de <strong>Bom Pagador</strong>.")

    return exp, frases, img_path

# === 5. Selecionar exemplos reais ===
mau_idx = df[df['alvo'] == 1].index[0]
bom_idx = df[df['alvo'] == 2].index[0]

inst_mau = df.iloc[mau_idx].drop('alvo')
inst_bom = df.iloc[bom_idx].drop('alvo')

exp_mau, frases_mau, img_mau = gerar_explicacao(inst_mau, "lime_explicacao_mau_pagador", "Por que o modelo classificou como 'Mau Pagador'?")
exp_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "lime_explicacao_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")

# === 6. HTML com dois exemplos e simulador ===
html_path = "images/lime_explanation_duplo.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'></head><body style='font-family:Arial; background:#f9f9f9; padding:30px;'>")

    # Seção: Mau Pagador
    f.write("<h2>📌 Exemplo 1: Cliente classificado como <span style='color:red'>Mau Pagador</span></h2>")
    f.write("<h3>📊 Gráfico Interativo</h3>")
    f.write(exp_mau.as_html())
    f.write("<h3>🧾 Explicações</h3><ul>")
    for frase in frases_mau:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # Seção: Bom Pagador
    f.write("<h2>📌 Exemplo 2: Cliente classificado como <span style='color:green'>Bom Pagador</span></h2>")
    f.write("<h3>📊 Gráfico Interativo</h3>")
    f.write(exp_bom.as_html())
    f.write("<h3>🧾 Explicações</h3><ul>")
    for frase in frases_bom:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # Simulador de crédito
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

print("✅ Gráfico Mau Pagador: images/lime_explicacao_mau_pagador.png")
print("✅ Gráfico Bom Pagador: images/lime_explicacao_bom_pagador.png")
print("✅ HTML completo: images/lime_explanation_duplo.html")
