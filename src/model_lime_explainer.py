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
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone', 'trabalhador_estrangeiro',
    'alvo'
]

df = pd.read_csv("data/german.data", sep=' ', header=None)
df.columns = colunas

# Codificação de variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Ajustar classe alvo: 1 → Bom pagador, 2 → Mau pagador → para 0 e 1
df['alvo'] = df['alvo'].map({1: 0, 2: 1})  # 0 = bom, 1 = mau

# Separar variáveis
X = df.drop('alvo', axis=1)
y = df['alvo']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar o modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação do modelo
print("=== Relatório de Classificação ===")
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# === 3. Aplicar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# === Função para gerar explicações LIME ===
def gerar_explicacao(instancia, nome_arquivo, titulo_plot):
    exp = explainer.explain_instance(instancia.values, modelo.predict_proba, num_features=10)
    
    # Geração do gráfico PNG
    fig = exp.as_pyplot_figure()
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
    path = f"images/{nome_arquivo}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, peso in exp.as_list():
        if peso > 0:
            frases.append(f"🟠 O fator **{feature}** aumentou a chance de classificar como **mau pagador**.")
        else:
            frases.append(f"🔵 O fator **{feature}** indicou chance maior de ser **bom pagador**.")
    
    return exp, frases, path

# === 4. Selecionar uma instância de cada classe ===
bom_idx = y_test[y_test == 0].index[0]
mau_idx = y_test[y_test == 1].index[0]

inst_bom = X_test.loc[bom_idx]
inst_mau = X_test.loc[mau_idx]

# === 5. Gerar explicações para bom e mau pagador ===
exp_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", 
                                                "Por que o modelo classificou como 'Bom Pagador'?")

exp_mau, frases_mau, img_mau = gerar_explicacao(inst_mau, "grafico_mau_pagador", 
                                                "Por que o modelo classificou como 'Mau Pagador'?")

# === 6. Mostrar resultados no terminal ===
print("\n=== Explicações para BOM PAGADOR ===")
for frase in frases_bom:
    print(frase)

print("\n=== Explicações para MAU PAGADOR ===")
for frase in frases_mau:
    print(frase)

print("\n✅ Gráficos PNG salvos em:")
print(f"   → {img_bom}")
print(f"   → {img_mau}")
