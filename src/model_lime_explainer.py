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

# === Etapa 1: Leitura dos dados com fallback ===
colunas = [
    'status_conta', 'duracao', 'historico_credito', 'proposito', 'valor_credito',
    'conta_poupanca', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residencia', 'propriedade', 'idade', 'outras_parcelas',
    'moradia', 'numero_empregos', 'profissao', 'responsaveis', 'telefone',
    'trabalhador_estrangeiro', 'alvo'
]

caminhos_possiveis = ["data/german.data", "../data/german.data"]
df = None
for caminho in caminhos_possiveis:
    if os.path.exists(caminho):
        df = pd.read_csv(caminho, sep=' ', header=None, names=colunas)
        print(f"✅ Arquivo carregado de: {caminho}")
        break

if df is None:
    raise FileNotFoundError("❌ Arquivo german.data não encontrado nos caminhos esperados.")

# === Etapa 2: Pré-processamento ===
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("alvo", axis=1)
y = df["alvo"].map({1: 1, 2: 0})  # 1 = Bom Pagador, 0 = Mal Pagador

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Etapa 3: Treinamento do modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# === Etapa 4: Avaliação ===
y_pred = modelo.predict(X_test)
print("\n🎯 Relatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=["Mal Pagador", "Bom Pagador"]))

# === Etapa 5: Instâncias para explicação ===
bom_idx = y_test[y_test == 1].index[0]
mal_idx = y_test[y_test == 0].index[0]

inst_bom = X_test.loc[bom_idx]
inst_mal = X_test.loc[mal_idx]

# === Etapa 6: Inicializar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode="classification"
)

# === Etapa 7: Função para gerar explicação ===
def gerar_explicacao(instancia, nome_arquivo, titulo):
    exp = explainer.explain_instance(instancia.values, modelo.predict_proba, num_features=10)
    predicao = int(modelo.predict([instancia.values])[0])

    # Gráfico PNG
    fig = exp.as_pyplot_figure(label=predicao)
    fig.set_size_inches(14, 6)
    plt.title(titulo, fontsize=14)
    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão do modelo.\n"
        "🔵 Azul: Características que indicaram tendência contrária à decisão."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='gray'))
    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, weight in exp.as_list(label=predicao):
        if weight > 0:
            frases.append(f"🟠 O fator **{feature}** contribuiu para classificar como **{['Mal Pagador', 'Bom Pagador'][predicao]}**.")
        else:
            frases.append(f"🔵 O fator **{feature}** indicou tendência oposta à classificação como **{['Mal Pagador', 'Bom Pagador'][predicao]}**.")
    return exp, frases, img_path

# === Etapa 8: Gerar explicações ===
exp_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
exp_mal, frases_mal, img_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# === Etapa 9: Imprimir resumo no terminal ===
print("\n✅ Gráficos gerados:")
print(f"📊 Bom Pagador: {img_bom}")
print(f"📊 Mal Pagador: {img_mal}\n")

print("🧾 Explicações do Bom Pagador:")
for frase in frases_bom:
    print("-", frase)

print("\n🧾 Explicações do Mal Pagador:")
for frase in frases_mal:
    print("-", frase)
