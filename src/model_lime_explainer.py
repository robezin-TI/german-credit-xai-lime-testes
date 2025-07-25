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

# === 1. Carregar dados ===
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

# Ajustar rótulo: 1 = Mau Pagador → 1, 2 = Bom Pagador → 0
df['alvo'] = df['alvo'].map({1: 1, 2: 0})

# Dividir dados
X = df.drop('alvo', axis=1)
y = df['alvo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# === 3. LIME Explainer ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# === Função para gerar explicações ===
def gerar_explicacao(instancia, nome_arquivo_img, titulo_grafico):
    exp = explainer.explain_instance(instancia.to_numpy(), model.predict_proba, num_features=10)
    predicao = int(model.predict(instancia.to_numpy().reshape(1, -1))[0])
    try:
        fig = exp.as_pyplot_figure(label=predicao)
        fig.set_size_inches(14, 6)
        plt.title(titulo_grafico, fontsize=14)
        plt.xlabel("Contribuição para a decisão", fontsize=12)
        legenda = (
            "🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
            "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
        )
        plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='gray'))
        os.makedirs("images", exist_ok=True)
        caminho = f"images/{nome_arquivo_img}.png"
        plt.savefig(caminho, bbox_inches='tight')
        plt.close()
    except KeyError as e:
        print(f"Erro ao gerar gráfico LIME: {e}")
        caminho = ""
    # Frases explicativas
    frases = []
    for feature, weight in exp.as_list():
        if weight > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> contribuiu para a classificação como <strong>Mau Pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> indicou possível perfil de <strong>Bom Pagador</strong>.")
    return exp, frases, caminho

# === 4. Selecionar exemplos: um bom e um mau pagador ===
bom_idx = y_test[y_test == 0].index[0]
mau_idx = y_test[y_test == 1].index[0]
inst_bom = X.loc[bom_idx]
inst_mau = X.loc[mau_idx]

# === 5. Gerar explicações ===
exp_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "lime_explicacao_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
exp_mau, frases_mau, img_mau = gerar_explicacao(inst_mau, "lime_explicacao_mau_pagador", "Por que o modelo classificou como 'Mau Pagador'?")

# === 6. Gerar HTML ===
html_path = "images/lime_explanation_ptbr.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'></head><body style='font-family:Arial; padding:30px;'>")

    # Exemplo 1 — Mau Pagador
    f.write("<h2>📌 Exemplo 1: Cliente classificado como <span style='color:red;'>Mau Pagador</span></h2>")
    f.write("<h3>📊 Gráfico Interativo</h3>")
    f.write(exp_mau.as_html())
    f.write("<h3>🧾 Explicações</h3><ul>")
    for frase in frases_mau:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # Exemplo 2 — Bom Pagador
    f.write("<h2>📌 Exemplo 2: Cliente classificado como <span style='color:green;'>Bom Pagador</span></h2>")
    f.write("<h3>📊 Gráfico Interativo</h3>")
    f.write(exp_bom.as_html())
    f.write("<h3>🧾 Explicações</h3><ul>")
    for frase in frases_bom:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # Simulador (mantido como estava)
    f.write("<h3>📝 Simule sua solicitação de crédito:</h3>")
    # (Você pode colar aqui o simulador HTML com JS que já estava funcionando.)

    f.write("</body></html>")

# === Conclusão ===
print("✅ Gráfico Mau Pagador:", img_mau)
print("✅ Gráfico Bom Pagador:", img_bom)
print("✅ HTML gerado em:", html_path)
