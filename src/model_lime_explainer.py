import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# 1. Carregamento do dataset
# -----------------------------
colunas = [
    'status_conta', 'duração', 'histórico_crédito', 'propósito', 'valor_crédito',
    'conta_poupança', 'tempo_emprego', 'taxa_parcelamento', 'status_pessoal_sexo',
    'fiadores', 'duração_residência', 'propriedade', 'idade', 'outros_planos',
    'habitação', 'n_cred_pendentes', 'emprego_desde', 'trabalhador_estrangeiro',
    'telefone', 'profissão', 'classe'
]

# Caminho robusto para acesso ao dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
data_path = os.path.join(base_path, 'german.data')

# Carregamento
df = pd.read_csv(data_path, sep=' ', header=None, names=colunas)

# -----------------------------
# 2. Pré-processamento
# -----------------------------
df['classe'] = df['classe'].map({1: 1, 2: 0})  # 1: bom pagador, 0: mal pagador
X = df.drop(columns=['classe'])
y = df['classe']

# One-hot encoding automático para variáveis categóricas
X_encoded = pd.get_dummies(X)

# -----------------------------
# 3. Treinamento do modelo
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, modelo.predict(X_test)))

# -----------------------------
# 4. Explicador LIME
# -----------------------------
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_encoded.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode="classification",
    discretize_continuous=True
)

# -----------------------------
# 5. Função de explicação com LIME
# -----------------------------
def gerar_explicacao(instancia, nome_arquivo, titulo):
    pred = modelo.predict([instancia])[0]
    probas = modelo.predict_proba([instancia])[0]
    label = int(pred)

    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)

    fig = exp.as_pyplot_figure(label=label)
    plt.title(titulo, fontsize=12, fontweight='bold')
    plt.tight_layout()
    img_path = os.path.join("..", "data", f"{nome_arquivo}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

    frases = []
    for atributo, peso in exp.as_list(label=label):
        cor = "🔵" if peso > 0 else "🟠"
        direcao = "Bom Pagador" if peso > 0 else "Mal Pagador"
        frases.append(f"- {cor} A característica {atributo} contribuiu para classificar como {direcao}.")

    return exp, frases, img_path, label

# -----------------------------
# 6. Seleção automática de instâncias
# -----------------------------
inst_bom = X_test[y_test == 1].iloc[0]
inst_mal = X_test[y_test == 0].iloc[0]

# -----------------------------
# 7. Gerar explicações
# -----------------------------
_, frases_bom, img_bom, _ = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
_, frases_mal, img_mal, _ = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

print("\n✅ Explicações para Bom Pagador:")
print("\n".join(frases_bom))

print("\n✅ Explicações para Mal Pagador:")
print("\n".join(frases_mal))

# -----------------------------
# 8. Explicação interpretativa final (para relatório)
# -----------------------------
print("\n📘 Definição interpretativa:")

print("\n✅ Bom Pagador")
print("Um cliente é classificado como Bom Pagador quando possui um perfil que sugere baixo risco de inadimplência. Entre os principais fatores que influenciam positivamente estão:\n")
print("- Ter uma conta bancária ativa com bom histórico de movimentação.")
print("- Apresentar um bom histórico de crédito (pagamentos anteriores em dia).")
print("- Ter uma idade mais avançada, geralmente acima dos 30 anos, o que indica maior estabilidade.")
print("- Solicitar valores de crédito mais baixos ou proporcionais à renda.")
print("- Estar empregado há mais tempo, demonstrando estabilidade profissional.")
print("- Ter objetivos de crédito claros e seguros, como aquisição de bens essenciais.")
print("- Possuir bens no nome (como carro ou imóvel).")
print("- Ter telefone ativo, o que sugere maior rastreabilidade e transparência.")

print("\n❌ Mal Pagador")
print("Um cliente é classificado como Mal Pagador quando o modelo identifica um conjunto de características associadas a maior risco de inadimplência. Entre os fatores mais comuns estão:\n")
print("- Ausência de conta bancária ativa ou movimentações suspeitas.")
print("- Histórico de crédito ruim ou inexistente.")
print("- Idade muito baixa, indicando pouca experiência financeira.")
print("- Solicitação de valores elevados de crédito, desproporcionais à estabilidade demonstrada.")
print("- Pouco tempo no emprego atual.")
print("- Falta de reserva financeira (como conta poupança ou investimentos).")
print("- Motivações de crédito mais arriscadas, como empréstimos para consumo não essencial.")
print("- Ausência de patrimônio registrado ou garantias.")
