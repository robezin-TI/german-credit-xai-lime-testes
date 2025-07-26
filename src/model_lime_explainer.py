import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lime.lime_tabular import LimeTabularExplainer

# Definir nomes das colunas
colunas = [
    'status_conta', 'duracao', 'historico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'garantidor', 'duracao_residencia', 'idade', 'outras_contas', 'moradia',
    'número_créditos', 'trabalhador_autônomo', 'trabalhador_estrangeiro',
    'emprego_telefone', 'propriedade', 'profissão', 'classe'
]

# Carregar o dataset (caminho relativo correto)
df = pd.read_csv("../data/german.data", sep=' ', header=None, names=colunas)

# Ajustar a variável alvo
df['classe'] = df['classe'].map({1: 1, 2: 0})

# Separar variáveis independentes e alvo
X = df.drop('classe', axis=1)
y = df['classe']

# One-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# Instanciador do LIME
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode='classification',
    verbose=True,
    discretize_continuous=True
)

# Função auxiliar para gerar explicações
def gerar_explicacao(instancia, nome_arquivo, titulo_grafico):
    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)
    predicao = int(modelo.predict([instancia])[0])

    # Gerar gráfico
    fig = exp.as_pyplot_figure(label=predicao)
    plt.title(titulo_grafico)
    plt.xlabel("Contribuição para a decisão")
    colors = {'positive': 'green', 'negative': 'red'}
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=6, label='🔶 Laranja: Características que reforçaram a decisão de negar o crédito.'),
        plt.Line2D([0], [0], color='green', lw=6, label='🔷 Azul: Características que sugerem que o crédito poderia ser concedido.')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    fig.savefig(nome_arquivo + ".png", bbox_inches='tight')
    plt.close(fig)

    # Frases explicativas
    explicacoes = exp.as_list(label=predicao)
    frases = []
    for nome, valor in explicacoes:
        direcao = "reforçou a decisão" if valor < 0 else "sugeriu que o crédito poderia ser concedido"
        frases.append(f"🔹 O fator {nome} {direcao}.")
    return exp, frases

# Selecionar instância de bom pagador
for i in range(len(X_test)):
    if modelo.predict([X_test.iloc[i]])[0] == 1:
        inst_bom = X_test.iloc[i].values
        break

# Selecionar instância de mal pagador
for i in range(len(X_test)):
    if modelo.predict([X_test.iloc[i]])[0] == 0:
        inst_mal = X_test.iloc[i].values
        break

# Gerar explicações e gráficos
_, frases_bom = gerar_explicacao(inst_bom, "../data/grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
_, frases_mal = gerar_explicacao(inst_mal, "../data/grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Exibir frases explicativas no terminal
print("\n📘 Explicações - Bom Pagador:")
for frase in frases_bom:
    print(frase)

print("\n📕 Explicações - Mal Pagador:")
for frase in frases_mal:
    print(frase)
