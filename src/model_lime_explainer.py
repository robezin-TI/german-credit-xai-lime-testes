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
print("Relatório de Classificação:\n")
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
    probas = modelo.predict_proba([instancia])
    
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
    plt.close
