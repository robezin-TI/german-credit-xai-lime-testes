import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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

# Ajustar labels: 1 → Mau Pagador, 2 → Bom Pagador → reindexar para 0 e 1
df['alvo'] = df['alvo'].replace({1: 1, 2: 0})  # 1 = Mau, 0 = Bom

X = df.drop('alvo', axis=1)
y = df['alvo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Relatório de Classificação:\n")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === 3. LIME Explainer ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

def gerar_explicacao(instancia, nome_arquivo_png, titulo):
    pred_proba = model.predict_proba([instancia])[0]
    predicao = np.argmax(pred_proba)
    exp = explainer.explain_instance(instancia, model.predict_proba, num_features=10)
    
    # Gráfico
    fig = exp.as_pyplot_figure(label=predicao)
    fig.set_size_inches(12, 6)
    plt.title(titulo)
    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
        "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo_png}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, peso in exp.as_list():
        if peso > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> aumentou a chance de classificar como <strong>mau pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> contribuiu para considerar como <strong>bom pagador</strong>.")
    return exp, frases, img_path

# === 4. Selecionar exemplo real de bom e mau pagador ===
for idx in range(len(X_test)):
    instancia = X_test.iloc[idx].values
    predicao = model.predict([instancia])[0]
    if predicao == 1:
        exp_mau, frases_mau, img_mau = gerar_explicacao(instancia, "grafico_mau_pagador", "Por que o modelo classificou como 'Mau Pagador'?")
        inst_mau_html = exp_mau.as_html()
        break

for idx in range(len(X_test)):
    instancia = X_test.iloc[idx].values
    predicao = model.predict([instancia])[0]
    if predicao == 0:
        exp_bom, frases_bom, img_bom = gerar_explicacao(instancia, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
        inst_bom_html = exp_bom.as_html()
        break

# === 5. Criar HTML final ===
html_path = "images/lime_explicacao_dois_exemplos.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("""
    <html><head><meta charset="utf-8"></head>
    <body style="font-family: Arial; background-color: #f9f9f9; padding: 30px;">
    <h2>📌 Explicações do modelo</h2>
    <p>Abaixo estão dois exemplos reais explicados com o modelo treinado: um <strong>Bom Pagador</strong> e um <strong>Mau Pagador</strong>.</p>
    <hr>
    """)

    # Mau Pagador
    f.write("<h3 style='color:crimson;'>🔴 Exemplo de Mau Pagador</h3>")
    f.write("<h4>📊 Gráfico Interativo:</h4>")
    f.write(inst_mau_html)
    f.write("<h4>🧾 Frases explicativas:</h4><ul>")
    for frase in frases_mau:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # Bom Pagador
    f.write("<h3 style='color:green;'>🟢 Exemplo de Bom Pagador</h3>")
    f.write("<h4>📊 Gráfico Interativo:</h4>")
    f.write(inst_bom_html)
    f.write("<h4>🧾 Frases explicativas:</h4><ul>")
    for frase in frases_bom:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # Simulador
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
                mensagens.push("❌ Menos de 6 meses no emprego atual pode ser um fator negativo.");
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
            mensagens.push("❌ Renda mensal insuficiente para a parcela (~R$" + parcela.toFixed(2) + ").");
            aprovado = false;
        } else {
            mensagens.push("✅ Renda condizente com o valor e o prazo.");
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

print("✅ Gráficos e HTML gerados com sucesso!")
