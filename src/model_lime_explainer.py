# model_lime_explainer.py
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
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone',
    'trabalhador_estrangeiro', 'alvo'
]
df = pd.read_csv('data/german.data', sep=' ', header=None)
df.columns = colunas

# Label encoding
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar dados
X = df.drop('alvo', axis=1)
y = df['alvo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Relatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# Explainer LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# Função para gerar explicação, gráfico e frases
def gerar_explicacao(instancia, nome, classe_real):
    exp = explainer.explain_instance(instancia.to_numpy(), model.predict_proba, num_features=10)
    label_predito = model.predict([instancia])[0]

    # Gráfico
    fig = exp.as_pyplot_figure(label=label_predito - 1)
    fig.set_size_inches(14, 6)
    plt.title(f"Explicação Local: {nome}", fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)
    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
        "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))
    plt.tight_layout()
    img_path = f"images/lime_{nome.lower().replace(' ', '_')}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases
    frases = []
    for feature, weight in exp.as_list(label=label_predito - 1):
        if weight > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> aumentou a chance de classificar como <strong>mau pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> indicou que poderia ser um <strong>bom pagador</strong>.")

    return exp, frases, img_path, label_predito

# Selecionar um bom e um mau pagador
bom_idx = df[df["alvo"] == 2].index[0]
mau_idx = df[df["alvo"] == 1].index[0]
bom_instance = df.loc[bom_idx].drop("alvo")
mau_instance = df.loc[mau_idx].drop("alvo")

exp_bom, frases_bom, img_bom, _ = gerar_explicacao(bom_instance, "Bom Pagador", 2)
exp_mau, frases_mau, img_mau, _ = gerar_explicacao(mau_instance, "Mau Pagador", 1)

# Criar HTML
html_path = "images/lime_explanation_ptbr.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("""
    <html><head><meta charset="utf-8"></head>
    <body style="font-family: Arial; background:#f9f9f9; padding:30px; max-width:900px; margin:auto;">
    <h1>📌 Explicações do Modelo de Crédito</h1>
    """)

    for titulo, exp, frases in [
        ("Exemplo 1 - Bom Pagador", exp_bom, frases_bom),
        ("Exemplo 2 - Mau Pagador", exp_mau, frases_mau)
    ]:
        f.write(f"<h2>{titulo}</h2>")
        f.write("<h3>📊 Gráfico Interativo:</h3>")
        f.write(exp.as_html())
        f.write("<h3>🧾 Fatores que influenciaram a decisão:</h3><ul>")
        for frase in frases:
            f.write(f"<li>{frase}</li>")
        f.write("</ul><hr>")

    # Simulador
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
        <label>Empregado desde quando (em meses): <input type="number" id="emprego_tempo" required></label><br><br>
        <label>Quantas pessoas moram com você? <input type="number" id="moradores" required></label><br><br>
        <label>Renda mensal (R$): <input type="number" id="renda" required></label><br><br>
        <button type="button" onclick="simular()">Ver Resultado</button>
    </form>
    <p id="resultadoSimulacao" style="font-weight:bold; font-size:16px; color:#333; margin-top:20px;"></p>
    <script>
    function simular() {
        const idade = parseInt(document.getElementById("idade").value);
        const valor = parseInt(document.getElementById("valor").value);
        const duracao = parseInt(document.getElementById("duracao").value);
        const empregado = document.getElementById("empregado").value;
        const emprego_tempo = parseInt(document.getElementById("emprego_tempo").value);
        const moradores = parseInt(document.getElementById("moradores").value);
        const renda = parseInt(document.getElementById("renda").value);

        let mensagens = [], aprovado = true;
        if (idade < 21) { mensagens.push("❌ Idade abaixo de 21 anos."); aprovado = false; }
        else { mensagens.push("✅ Idade adequada."); }
        if (empregado === "nao") { mensagens.push("❌ Está desempregado."); aprovado = false; }
        else {
            mensagens.push("✅ Está empregado.");
            if (emprego_tempo < 6) { mensagens.push("❌ Menos de 6 meses no emprego."); aprovado = false; }
            else { mensagens.push("✅ Tempo de emprego satisfatório."); }
        }
        if (moradores > 3) { mensagens.push("❌ Muitos moradores."); aprovado = false; }
        else { mensagens.push("✅ Número de moradores adequado."); }
        let parcela = valor / duracao;
        if (parcela > renda * 0.5) {
            mensagens.push("❌ Parcela R$" + parcela.toFixed(2) + " compromete mais de 50% da renda.");
            aprovado = false;
        } else {
            mensagens.push("✅ Parcela compatível com renda.");
        }
        const resultado = aprovado ? "✅ Crédito provavelmente APROVADO." : "❌ Crédito provavelmente NEGADO.";
        document.getElementById("resultadoSimulacao").innerHTML =
            "<p style='font-size:18px;'>" + resultado + "</p><ul><li>" + mensagens.join("</li><li>") + "</li></ul>";
    }
    </script>
    </body></html>
    """)

print("✅ HTML completo gerado com dois exemplos e simulador em: images/lime_explanation_ptbr.html")
