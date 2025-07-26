import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Caminho do dataset
caminho_dataset = os.path.join(os.path.dirname(__file__), '../data/german.data')

# Nomes das colunas com tradução
colunas = [
    "status_conta", "duração", "histórico_crédito", "propósito", "valor_crédito",
    "conta_poupança", "emprego_desde", "taxa_parcelamento", "sexo_estado_civil", "garantia",
    "residência_anos", "propriedade", "idade", "outros_planos", "habitação",
    "número_créditos", "trabalho", "trabalhador_estrangeiro", "telefone", "classe"
]

# Carregamento do dataset
df = pd.read_csv(caminho_dataset, sep=' ', header=None, names=colunas)

# Separação entre atributos e rótulos
X = df.drop("classe", axis=1)
y = df["classe"]

# Divisão em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
print("Relatório de Classificação:\n")
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Preparação para o LIME
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns,
    class_names=['Bom Pagador', 'Mal Pagador'],
    discretize_continuous=True,
    mode="classification"
)

# Seleção de exemplos
X_test_np = X_test.to_numpy()
y_pred = modelo.predict(X_test)

idx_bom = np.where(y_pred == 1)[0][0]
idx_mal = np.where(y_pred == 2)[0][0]
inst_bom = X_test_np[idx_bom]
inst_mal = X_test_np[idx_mal]

# Função para gerar explicações LIME + PNG
def gerar_explicacao(instancia, nome_arquivo, titulo):
    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure(label=exp.available_labels()[0])
    
    # Customizações
    plt.title(titulo)
    plt.xlabel("Contribuição para a decisão")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(
            handles=handles,
            labels=["🟠 Laranja: Características que aumentam a chance de ser classificado como 'Mal Pagador'.",
                    "🔵 Azul: Características que aumentam a chance de ser classificado como 'Bom Pagador'."],
            loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=8
        )

    caminho_img = os.path.join(os.path.dirname(__file__), f"../img/{nome_arquivo}.png")
    plt.savefig(caminho_img, bbox_inches="tight")
    plt.close()

    explicacoes = []
    for feature, peso in exp.as_list(label=exp.available_labels()[0]):
        cor = "🟠" if peso > 0 else "🔵"
        direcao = "Mal Pagador" if peso > 0 else "Bom Pagador"
        explicacoes.append(f"{cor} A característica {feature} contribuiu para classificar como {direcao}.")

    return explicacoes

# Geração dos gráficos e frases explicativas
frases_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
frases_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Impressão das explicações
print("\n✅ Explicações para Bom Pagador:")
for frase in frases_bom:
    print("-", frase)

print("\n✅ Explicações para Mal Pagador:")
for frase in frases_mal:
    print("-", frase)

# Explicação final geral
print("\n📘 Definições gerais utilizadas pelo modelo:")

print("""
✅ Bom Pagador
Um cliente é classificado como Bom Pagador quando possui um perfil que sugere baixo risco de inadimplência. Entre os principais fatores que influenciam positivamente estão:

- Ter uma conta bancária ativa com bom histórico de movimentação.
- Apresentar um bom histórico de crédito (pagamentos anteriores em dia).
- Ter uma idade mais avançada, geralmente acima dos 30 anos, o que indica maior estabilidade.
- Solicitar valores de crédito mais baixos ou proporcionais à renda.
- Estar empregado há mais tempo, demonstrando estabilidade profissional.
- Ter objetivos de crédito claros e seguros, como aquisição de bens essenciais.
- Possuir bens no nome (como carro ou imóvel).
- Ter telefone ativo, o que sugere maior rastreabilidade e transparência.

❌ Mal Pagador
Um cliente é classificado como Mal Pagador quando o modelo identifica um conjunto de características associadas a maior risco de inadimplência. Entre os fatores mais comuns estão:

- Ausência de conta bancária ativa ou movimentações suspeitas.
- Histórico de crédito ruim ou inexistente.
- Idade muito baixa, indicando pouca experiência financeira.
- Solicitação de valores elevados de crédito, desproporcionais à estabilidade demonstrada.
- Pouco tempo no emprego atual.
- Falta de reserva financeira (como conta poupança ou investimentos).
- Motivações de crédito mais arriscadas, como empréstimos para consumo não essencial.
- Ausência de patrimônio registrado ou garantias.
""")
