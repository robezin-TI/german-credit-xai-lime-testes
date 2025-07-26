import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular

# Caminho robusto para o arquivo
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '..', 'data', 'german.data')

# Nomes das colunas conforme documentação
colunas = [
    "status_conta", "duração", "histórico_crédito", "propósito", "valor_crédito",
    "conta_poupança", "residência_atual", "emprego_desde", "taxa_parcelamento",
    "sexo_estado_civil", "garantia", "residência", "idade", "outros_planos",
    "habitação", "n_creditos", "trabalhador_estrangeiro", "profissão",
    "telefone", "status", "classe"
]

# Leitura do dataset
df = pd.read_csv(data_path, sep=' ', header=None, names=colunas)

# Separar X e y
X = df.drop("classe", axis=1)
y = df["classe"].copy()

# Ajustar rótulos: 1 = bom pagador, 2 = mal pagador
y = y.map({1: 1, 2: 0})  # 1=bom, 0=mal

# Codificar variáveis categóricas
X_encoded = pd.get_dummies(X)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# Preparar diretório de saída
output_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)

# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode="classification"
)

# Função de explicação com salvamento e frases
def gerar_explicacao(instancia, nome_arquivo, titulo):
    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure(label=int(modelo.predict([instancia])[0]))
    plt.title(titulo)
    img_path = os.path.join(output_dir, f"{nome_arquivo}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    explicacoes = exp.as_list(label=int(modelo.predict([instancia])[0]))
    frases = []
    for feat, val in explicacoes:
        direcao = "🟢" if val > 0 else "🔴"
        classe = "Bom Pagador" if val > 0 else "Mal Pagador"
        frases.append(f"- {direcao} A característica {feat} contribuiu para classificar como {classe}.")
    return frases

# Selecionar instâncias reais
inst_bom = X_test[y_test == 1].iloc[0]
inst_mal = X_test[y_test == 0].iloc[0]

# Gerar explicações
frases_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
frases_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Apresentar explicações
print("\n✅ Explicações para Bom Pagador:")
for frase in frases_bom:
    print(frase)

print("\n❌ Explicações para Mal Pagador:")
for frase in frases_mal:
    print(frase)

# Explicações gerais para o relatório
print("\n📘 O que define um Bom ou Mal Pagador:\n")

print("✅ Bom Pagador")
print("Um cliente é classificado como Bom Pagador quando possui um perfil que sugere baixo risco de inadimplência. Entre os principais fatores estão:")
print("- Ter uma conta bancária ativa com bom histórico de movimentação.")
print("- Apresentar um bom histórico de crédito (pagamentos em dia).")
print("- Idade mais avançada (acima dos 30 anos).")
print("- Solicitar valores proporcionais à renda.")
print("- Estabilidade no emprego atual.")
print("- Objetivos de crédito claros (ex: aquisição de bens).")
print("- Possuir bens no nome (carro, imóvel).")
print("- Ter telefone ativo (transparência e rastreabilidade).")

print("\n❌ Mal Pagador")
print("Um cliente é classificado como Mal Pagador quando o modelo identifica fatores associados a maior risco de inadimplência. Entre os principais estão:")
print("- Ausência de conta bancária ativa ou movimentações suspeitas.")
print("- Histórico de crédito ruim ou inexistente.")
print("- Idade muito baixa (pouca experiência).")
print("- Solicitação de valores elevados e desproporcionais.")
print("- Pouco tempo no emprego atual.")
print("- Falta de reserva financeira ou garantias.")
print("- Motivações de crédito mais arriscadas.")
