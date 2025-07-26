import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime.lime_tabular

# ============================
# 1. Carregar e preparar dados
# ============================
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'emprego_desde', 'taxa_parcelamento', 'sexo_estado_civil',
    'outros_devedores', 'tempo_residência', 'propriedade', 'idade', 'outras_parcelas',
    'moradia', 'número_empréstimos', 'profissão', 'responsáveis', 'telefone', 'trabalhador_estrangeiro',
    'alvo'
]

# Caminho fixo para evitar erros
df = pd.read_csv("data/german.data", sep=' ', header=None, names=colunas)

# Codificar variáveis categóricas
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar dados
X = df.drop("alvo", axis=1)
y = df["alvo"]

# Dividir base de treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================
# 2. Treinar o modelo
# ====================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar desempenho
print("\nRelatório de Classificação:\n")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ============================
# 3. Inicializar o LIME
# ============================
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=["Bom Pagador", "Mal Pagador"],
    mode='classification'
)

# Função para gerar explicação e gráfico
def gerar_explicacao(instancia, nome_arquivo, titulo):
    predicao = model.predict([instancia])[0]
    classe = "Bom Pagador" if predicao == 1 else "Mal Pagador"
    
    exp = explainer.explain_instance(
        data_row=instancia,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    fig = exp.as_pyplot_figure(label=int(predicao - 1))
    fig.set_size_inches(13, 5)
    plt.title(f"{titulo} (Classe: {classe})", fontsize=14)
    plt.xlabel("Contribuição para a decisão", fontsize=12)
    
    legenda = (
        "🟠 Laranja: Características que reforçaram a decisão de negar o crédito.\n"
        "🔵 Azul: Características que sugerem que o crédito poderia ser concedido."
    )
    plt.figtext(0.99, 0.01, legenda, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='gray'))
    
    os.makedirs("images", exist_ok=True)
    img_path = f"images/{nome_arquivo}.png"
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Frases explicativas
    frases = []
    for feature, weight in exp.as_list(label=int(predicao - 1)):
        if weight > 0:
            frases.append(f"🟠 O fator '{feature}' aumentou a chance de ser classificado como mal pagador.")
        else:
            frases.append(f"🔵 O fator '{feature}' indicou que o cliente poderia ser um bom pagador.")
    
    return exp, frases, img_path, classe

# ================================
# 4. Selecionar 1 bom e 1 mal pagador
# ================================
def encontrar_instancia(classe_desejada):
    for i in range(len(X_test)):
        instancia = X_test.iloc[i]
        pred = model.predict([instancia])[0]
        if pred == classe_desejada:
            return instancia
    return None

inst_bom = encontrar_instancia(1)  # Bom pagador
inst_mal = encontrar_instancia(2)  # Mal pagador

# ================================
# 5. Gerar gráficos e frases explicativas
# ================================
exp_bom, frases_bom, img_bom, classe_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
exp_mal, frases_mal, img_mal, classe_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# ================================
# 6. Mostrar frases no terminal
# ================================
print("\nExplicação do BOM PAGADOR:\n")
for frase in frases_bom:
    print(frase)

print("\nExplicação do MAL PAGADOR:\n")
for frase in frases_mal:
    print(frase)

# ================================
# 7. Mensagens finais
# ================================
print("\n✅ Gráficos gerados:")
print(f"- Bom Pagador: {img_bom}")
print(f"- Mal Pagador: {img_mal}")
