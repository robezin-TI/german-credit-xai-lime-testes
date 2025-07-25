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

# Corrigir rótulos: 1 = mau pagador → 1; 2 = bom pagador → 0
df['alvo'] = df['alvo'].map({1: 1, 2: 0})

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

# === 2. Treinar modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# === 3. Criar explicador LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# === 4. Função para gerar gráfico e frases ===
def gerar_explicacao(instancia, nome_img, titulo):
    exp = explainer.explain_instance(instancia, model.predict_proba, num_features=10)
    predicao = int(model.predict([instancia])[0])

    fig = exp.as_pyplot_figure(label=predicao)
    fig.set_size_inches(14, 6)
    plt.title(titulo, fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)
    plt.figtext(0.99, 0.01,
        "🟠 Características que reforçaram a decisão de negar o crédito.\n🔵 Características que sugerem que o crédito poderia ser concedido.",
        fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='gray'))
    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_img}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, peso in exp.as_list(label=predicao):
        if peso > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> contribuiu para considerar o cliente <strong>mau pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> indicou que o cliente poderia ser <strong>bom pagador</strong>.")
    return exp, frases, img_path

# === 5. Selecionar um bom pagador e um mau pagador reais ===
bom_idx = y_test[y_test == 0].index[0]
mau_idx = y_test[y_test == 1].index[0]
inst_bom = X_test.loc[bom_idx].to_numpy()
inst_mau = X_test.loc[mau_idx].to_numpy()

exp_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
exp_mau, frases_mau, img_mau = gerar_explicacao(inst_mau, "grafico_mau_pagador", "Por que o modelo classificou como 'Mau Pagador'?")

# === 6. Gerar HTML com os dois exemplos ===
html_path = "images/explicacao_credito.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("""
    <html><head><meta charset="utf-8"></head>
    <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 30px;">
    <h1>💳 Análise de Crédito com Interpretação</h1><hr>
    """)

    for tipo, exp, frases, img, titulo in [
        ("Bom Pagador", exp_bom, frases_bom, img_bom, "Por que o modelo classificou como 'Bom Pagador'?"),
        ("Mau Pagador", exp_mau, frases_mau, img_mau, "Por que o modelo classificou como 'Mau Pagador'?")
    ]:
        f.write(f"<h2>📌 Exemplo: {tipo}</h2>")
        f.write(f"<p>{titulo}</p>")
        f.write(f"<img src='{img}' style='max-width:100%; border:1px solid #ccc;'><br><br>")
        f.write("<h4>🧾 Explicações em linguagem simples:</h4><ul>")
        for frase in frases:
            f.write(f"<li>{frase}</li>")
        f.write("</ul><hr>")

    # === 7. Simulador de crédito ===
    f.write("""
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
        <label>Tempo de emprego (meses): <input type="number" id="emprego_tempo" required></label><br><br>
        <label>Quantas pessoas moram com você? <input type="number" id="moradores" required></label><br><br>
        <label>Renda mensal (R$): <input type="number" id="renda" required></label><br><br>
        <button type="button" onclick="simular()">Ver Resultado</button>
    </form>
    <p id="resultadoSimulacao" style="font-weight:bold;font-size:16px;color:#333;margin-top:20px;"></p>

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
                mensagens.push("❌ Tempo de emprego muito curto.");
                aprovado = false;
            } else {
                mensagens.push("✅ Tempo de emprego suficiente.");
            }
        }

        if (moradores > 3) {
            mensagens.push("❌ Muitas pessoas no domicílio podem comprometer a renda.");
            aprovado = false;
        } else {
            mensagens.push("✅ Quantidade de moradores adequada.");
        }

        const parcela = valor / duracao;
        if (parcela > renda * 0.5) {
            mensagens.push("❌ Renda mensal insuficiente para a parcela (~R$" + parcela.toFixed(2) + ").");
            aprovado = false;
        } else {
            mensagens.push("✅ Renda compatível com o crédito.");
        }

        const resultado = aprovado ?
          "✅ Provavelmente o crédito seria APROVADO." :
          "❌ Provavelmente o crédito seria NEGADO.";

        document.getElementById("resultadoSimulacao").innerHTML =
          "<p style='font-size:18px;'>" + resultado + "</p><ul><li>" + mensagens.join("</li><li>") + "</li></ul>";
    }
    </script>
    </body></html>
    """)

print("✅ HTML completo gerado: images/explicacao_credito.html")
print("✅ Gráficos PNG salvos:", img_bom, "e", img_mau)
