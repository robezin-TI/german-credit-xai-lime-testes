import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular

# === 1. Carregar os dados ===
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outras_parcelas',
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone',
    'trabalhador_estrangeiro', 'alvo'
]

caminhos_possiveis = ["data/german.data", "../data/german.data"]
for caminho in caminhos_possiveis:
    if os.path.exists(caminho):
        df = pd.read_csv(caminho, sep=' ', header=None, names=colunas)
        break
else:
    raise FileNotFoundError("Arquivo 'german.data' não encontrado.")

# Codificar variáveis categóricas
label_encoders = {}
for coluna in df.columns:
    if df[coluna].dtype == 'object':
        le = LabelEncoder()
        df[coluna] = le.fit_transform(df[coluna])
        label_encoders[coluna] = le

# Separar atributos e alvo
X = df.drop("alvo", axis=1)
y = df["alvo"]

# Dividir dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, modelo.predict(X_test)))

# === 3. Criar explicador LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=["Bom Pagador", "Mal Pagador"],
    mode="classification"
)

# === 4. Função para gerar explicações ===
def gerar_explicacao(instancia, nome_arquivo, titulo):
    predicao = modelo.predict([instancia])[0]

    exp = explainer.explain_instance(
        data_row=instancia,
        predict_fn=modelo.predict_proba,
        num_features=10
    )

    fig = exp.as_pyplot_figure(label=predicao)
    fig.set_size_inches(14, 6)
    plt.title(titulo, fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)

    legenda = (
        "🟠 Laranja: Características que aumentam a chance de ser classificado como 'Mal Pagador'.\n"
        "🔵 Azul: Características que aumentam a chance de ser classificado como 'Bom Pagador'."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))

    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    frases = []
    for feature, peso in exp.as_list(label=predicao):
        if peso > 0:
            frases.append(f"🟠 A característica <strong>{feature}</strong> contribuiu para classificar como <strong>Mal Pagador</strong>.")
        else:
            frases.append(f"🔵 A característica <strong>{feature}</strong> contribuiu para classificar como <strong>Bom Pagador</strong>.")

    return exp, frases, img_path, predicao

# === 5. Selecionar exemplos claros ===
idx_bom = next((i for i in y_test.index if y_test[i] == 1), None)
idx_mal = next((i for i in y_test.index if y_test[i] == 2), None)

if idx_bom is None or idx_mal is None:
    raise ValueError("Não foi possível encontrar exemplos de bom e mal pagador nos dados de teste.")

inst_bom = X_test.loc[idx_bom]
inst_mal = X_test.loc[idx_mal]

# === 6. Gerar explicações ===
exp_bom, frases_bom, img_bom, classe_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
exp_mal, frases_mal, img_mal, classe_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# === 7. Exibir frases ===
print("\n✅ Explicações para Bom Pagador:")
for frase in frases_bom:
    print("-", frase.replace("<strong>", "").replace("</strong>", ""))

print("\n✅ Explicações para Mal Pagador:")
for frase in frases_mal:
    print("-", frase.replace("<strong>", "").replace("</strong>", ""))

print(f"\n✅ Gráficos salvos em:\n  • {img_bom}\n  • {img_mal}")
