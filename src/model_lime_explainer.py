import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import os

# -------------------------------
# Carregar e preparar os dados
# -------------------------------
colunas = [
    "status_conta", "duração", "histórico_crédito", "propósito", "valor_crédito",
    "conta_poupança", "emprego_desde", "taxa_parcelamento", "sexo_estado_civil", "outros_devedores",
    "residência", "idade", "outro_plano", "moradia", "número_créditos", "trabalhador_estrangeiro",
    "empregado", "propriedade", "telefone", "profissão", "classe"
]

df = pd.read_csv("german.data", sep=' ', header=None, names=colunas)

# Ajuste do target
df["classe"] = df["classe"].map({1: 1, 2: 0})  # 1 = Bom pagador, 0 = Mal pagador

# Separar features e target
X = df.drop("classe", axis=1)
y = df["classe"]

# Transformar categóricas com OneHotEncoder
X = pd.get_dummies(X)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------------
# Treinar modelo
# -------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Avaliação
y_pred = clf.predict(X_test)
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Função explicativa com LIME
# -------------------------------
feature_names = list(X.columns)
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["Mal Pagador", "Bom Pagador"],
    discretize_continuous=True,
    verbose=False,
    random_state=42
)

def gerar_explicacao(instancia, nome_arquivo, titulo):
    exp = explainer.explain_instance(instancia, clf.predict_proba, num_features=10)
    predicao = int(clf.predict(instancia.reshape(1, -1))[0])

    # Gerar gráfico
    fig = exp.as_pyplot_figure(label=predicao)
    plt.title(titulo)
    legenda_texto = (
        "\n🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
        "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
    )
    plt.figtext(0.5, -0.05, legenda_texto, wrap=True, horizontalalignment='center', fontsize=9)
    plt.savefig(f"{nome_arquivo}.png", bbox_inches='tight')
    plt.close()

    # Frases explicativas simples
    explicacoes = exp.as_list(label=predicao)
    frases = []
    for texto, peso in explicacoes:
        direcao = "reforçou a decisão de negar" if peso < 0 else "sugeriu que o crédito poderia ser concedido"
        cor = "🟠" if peso < 0 else "🔵"
        frases.append(f"{cor} O fator {texto} {direcao}.")
    return predicao, frases, f"{nome_arquivo}.png"

# -------------------------------
# Seleção automática das instâncias
# -------------------------------
def selecionar_instancia_por_classe(X, y, classe=1):
    for i in range(len(X)):
        if y.iloc[i] == classe:
            return X[i], y.iloc[i]
    raise ValueError("Não foi possível encontrar uma instância com a classe especificada.")

# Selecionar exemplos
inst_bom, _ = selecionar_instancia_por_classe(X_test, y_test, classe=1)
inst_mal, _ = selecionar_instancia_por_classe(X_test, y_test, classe=0)

# Gerar explicações
pred_bom, frases_bom, img_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
pred_mal, frases_mal, img_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Mostrar explicações no console
print("\n✅ Explicações - Bom Pagador:")
for f in frases_bom:
    print(f)

print("\n✅ Explicações - Mal Pagador:")
for f in frases_mal:
    print(f)
