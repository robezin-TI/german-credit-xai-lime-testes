import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular

# === 1. Caminho robusto para arquivo ===
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '..', 'data', 'german.data')
output_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)

# === 2. Leitura e preparação dos dados ===
colunas = [
    "status_conta", "duração", "histórico_crédito", "propósito", "valor_crédito",
    "conta_poupança", "residência_atual", "emprego_desde", "taxa_parcelamento",
    "sexo_estado_civil", "garantia", "residência", "idade", "outros_planos",
    "habitação", "n_creditos", "trabalhador_estrangeiro", "profissão",
    "telefone", "status", "classe"
]
df = pd.read_csv(data_path, sep=' ', header=None, names=colunas)

X = df.drop("classe", axis=1)
y = df["classe"].map({1: 1, 2: 0})  # 1 = bom pagador, 0 = mal pagador
X_encoded = pd.get_dummies(X)

# Dividir conjunto
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# === 3. Treinamento do modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
print("Relatório de Classificação:\n")
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# === 4. Configurar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode="classification"
)

# === 5. Função de explicação + gráfico ===
def gerar_explicacao(instancia, nome_base, titulo):
    predicao = int(modelo.predict([instancia])[0])
    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)

    # Gerar gráfico
    fig = exp.as_pyplot_figure(label=predicao)
    plt.title(titulo)
    img_path = os.path.join(output_dir, f"{nome_base}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # Gerar frases
    frases = []
    for feat, val in exp.as_list(label=predicao):
        direcao = "🟢" if val > 0 else "🔴"
        classe = "Bom Pagador" if val > 0 else "Mal Pagador"
        frases.append(f"- {direcao} A característica {feat} contribuiu para classificar como {classe}.")

    return img_path, frases

# === 6. Instâncias para explicação ===
inst_bom = X_test[y_test == 1].iloc[0]
inst_mal = X_test[y_test == 0].iloc[0]

# Gerar explicações
grafico_bom, frases_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
grafico_mal, frases_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Explicações gerais
explicacao_geral_bom = [
    "✅ Bom Pagador",
    "Um cliente é classificado como Bom Pagador quando possui um perfil que sugere baixo risco de inadimplência:",
    "- Conta bancária ativa com bom histórico.",
    "- Histórico de crédito positivo (pagamentos em dia).",
    "- Idade mais avançada (acima de 30 anos).",
    "- Solicita valores proporcionais à renda.",
    "- Estável no emprego.",
    "- Finalidade do crédito é segura (ex: aquisição de bens).",
    "- Possui bens no nome (imóvel, carro).",
    "- Telefone ativo (rastreabilidade)."
]

explicacao_geral_mal = [
    "❌ Mal Pagador",
    "Um cliente é classificado como Mal Pagador quando o perfil sugere alto risco de inadimplência:",
    "- Ausência de conta ativa ou movimentações suspeitas.",
    "- Histórico de crédito ruim ou inexistente.",
    "- Idade muito baixa (inexperiente financeiramente).",
    "- Solicita valores elevados ou desproporcionais.",
    "- Pouco tempo de trabalho.",
    "- Falta de reserva ou garantia.",
    "- Motivações arriscadas (ex: consumo supérfluo)."
]

# === 7. Gerar PDF Consolidado ===
def gerar_pdf_consolidado():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "📄 Relatório Explicativo do Modelo de Crédito", ln=True, align="C")
    pdf.ln(8)

    def bloco_cliente(titulo, imagem, frases, explicacao):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, titulo, ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", '', 11)
        for f in frases:
            pdf.multi_cell(0, 8, f)
        pdf.ln(3)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "📘 O que define esse perfil:", ln=True)
        pdf.set_font("Arial", '', 11)
        for linha in explicacao:
            pdf.multi_cell(0, 7, linha)
        pdf.ln(5)

        if os.path.exists(imagem):
            pdf.image(imagem, x=10, w=180)
        pdf.ln(10)

    bloco_cliente("✅ Exemplo de Bom Pagador", grafico_bom, frases_bom, explicacao_geral_bom)
    bloco_cliente("❌ Exemplo de Mal Pagador", grafico_mal, frases_mal, explicacao_geral_mal)

    caminho_pdf = os.path.join(output_dir, "relatorio_explicativo.pdf")
    pdf.output(caminho_pdf)
    print(f"✅ PDF consolidado gerado: {caminho_pdf}")

# Gerar o relatório
gerar_pdf_consolidado()
