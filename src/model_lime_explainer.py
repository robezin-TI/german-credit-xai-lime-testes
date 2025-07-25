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

# Corrigir target para começar em 0 (0 = bom, 1 = mau) como o LIME espera
df['alvo'] = df['alvo'] - 1

# Codificação das variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop('alvo', axis=1)
y = df['alvo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# === 3. Aplicar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

def gerar_explicacao(instancia, nome_arquivo, titulo_plot):
    exp = explainer.explain_instance(instancia.to_numpy(), model.predict_proba, num_features=10)
    predicao = model.predict([instancia])[0]
    
    fig = exp.as_pyplot_figure(label=predicao)
    fig.set_size_inches(14, 6)
    plt.title(titulo_plot, fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)

    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
        "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))

    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo}.png"
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, weight in exp.as_list():
        if weight > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> aumentou a chance de ser classificado como <strong>mau pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> ajudou a indicar que o cliente pode ser <strong>bom pagador</strong>.")
    
    return exp, frases, img_path

# Selecionar um bom pagador e um mau pagador do conjunto de testes
inst_bom = X_test[y_test == 0].iloc[0]
inst_mau = X_test[y_test == 1].iloc[0]

# Gerar explicações e gráficos
exp_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "lime_explicacao_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
exp_mau, frases_mau, img_mau = gerar_explicacao(inst_mau, "lime_explicacao_mau_pagador", "Por que o modelo classificou como 'Mau Pagador'?")

# === 4. Gerar HTML final ===
html_path = "images/lime_explanation_completo.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'></head><body style='font-family: Arial, sans-serif; background:#f9f9f9; padding: 30px;'>")

    # --- MAU PAGADOR ---
    f.write("<h2>📌 Explicação da decisão para Mau Pagador</h2>")
    f.write("<p>O cliente foi classificado como <strong>Mau Pagador</strong>. Abaixo estão os principais fatores:</p>")
    f.write("<h3>📊 Gráfico Interativo:</h3>")
    f.write(exp_mau.as_html())
    f.write("<h3>🧾 Frases Explicativas:</h3><ul>")
    for frase in frases_mau:
        f.write(f"<li>{frase}</li>")
    f.write("</ul><hr>")

    # --- BOM PAGADOR ---
    f.write("<h2>📌 Explicação da decisão para Bom Pagador</h2>")
    f.write("<p>Este cliente foi classificado como <strong>Bom Pagador</strong> pelo modelo.</p>")
    f.write("<h3>📊 Gráfico Interativo:</h3>")
    f.write(exp_bom.as_html())
    f.write("<h3>🧾 Frases Explicativas:</h3><ul>")
    for frase in frases_bom:
        f.write(f"<li>{frase}</li>")
    f.write("</ul>")

    f.write("</body></html>")

print("✅ HTML salvo em: images/lime_explanation_completo.html")
print("✅ Gráfico bom pagador salvo:", img_bom)
print("✅ Gráfico mau pagador salvo:", img_mau)
