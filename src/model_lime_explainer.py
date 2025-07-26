import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime.lime_tabular

# === 1. Carregar e preparar os dados ===
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outras_parcelas',
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone', 'trabalhador_estrangeiro',
    'alvo'
]

# Caminho seguro para leitura
caminhos_possiveis = ["data/german.data", "../data/german.data", "./german.data"]
for caminho in caminhos_possiveis:
    if os.path.exists(caminho):
        df = pd.read_csv(caminho, sep=' ', header=None, names=colunas)
        break
else:
    raise FileNotFoundError("Arquivo 'german.data' não encontrado em nenhum dos caminhos esperados.")

# Codificação de variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar variáveis
X = df.drop('alvo', axis=1)
y = df['alvo']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Treinar o modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação
print("Relatório de Classificação:\n")
print(classification_report(y_test, model.predict(X_test)))

# === 3. Configurar o LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mal Pagador'],  # <-- corrigido
    mode='classification'
)

# === 4. Função para gerar gráfico LIME e explicações ===
def gerar_explicacao(instancia, nome_arquivo, titulo):
    predicao = int(model.predict([instancia])[0])
    exp = explainer.explain_instance(instancia.to_numpy(), model.predict_proba, num_features=10)
    
    label = 0 if predicao == 1 else 1  # LIME usa 0/1

    fig = exp.as_pyplot_figure(label=label)
    fig.set_size_inches(14, 6)
    plt.title(titulo, fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)
    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão negativa (Mal Pagador).\n"
        "🔵 Azul: Características que sugerem um perfil positivo (Bom Pagador)."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray'))
    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo}.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, weight in exp.as_list(label=label):
        if weight > 0:
            frases.append(f"🟠 O fator <strong>{feature}</strong> aumentou a chance de ser classificado como <strong>mal pagador</strong>.")
        else:
            frases.append(f"🔵 O fator <strong>{feature}</strong> indicou um perfil de <strong>bom pagador</strong>.")
    
    return exp, frases, img_path, predicao

# === 5. Selecionar automaticamente um exemplo de cada classe ===
def selecionar_instancia_por_classe(X_test, y_test, classe_alvo):
    for i in range(len(X_test)):
        instancia = X_test.iloc[i]
        pred = int(model.predict([instancia])[0])
        if pred == classe_alvo:
            return instancia
    raise ValueError(f"Nenhuma instância da classe {classe_alvo} encontrada.")

# Gerar instâncias e explicações
inst_bom = selecionar_instancia_por_classe(X_test, y_test, 1)
exp_bom, frases_bom, img_bom, classe_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")

inst_mal = selecionar_instancia_por_classe(X_test, y_test, 2)
exp_mal, frases_mal, img_mal, classe_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Mostrar saída no console
print(f"\n✅ Gráfico gerado: {img_bom} - Classificação: {'Bom Pagador' if classe_bom == 1 else 'Mal Pagador'}")
print(f"Explicações do bom pagador:")
for frase in frases_bom:
    print("-", frase)

print(f"\n✅ Gráfico gerado: {img_mal} - Classificação: {'Bom Pagador' if classe_mal == 1 else 'Mal Pagador'}")
print(f"Explicações do mal pagador:")
for frase in frases_mal:
    print("-", frase)
