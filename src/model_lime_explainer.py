import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular
from fpdf import FPDF

# Caminho robusto
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '..', 'data', 'german.data')

colunas = [
    "status_conta", "duração", "histórico_crédito", "propósito", "valor_crédito",
    "conta_poupança", "residência_atual", "emprego_desde", "taxa_parcelamento",
    "sexo_estado_civil", "garantia", "residência", "idade", "outros_planos",
    "habitação", "n_creditos", "trabalhador_estrangeiro", "profissão",
    "telefone", "status", "classe"
]

# Leitura
df = pd.read_csv(data_path, sep=' ', header=None, names=colunas)
X = df.drop("classe", axis=1)
y = df["classe"].map({1: 1, 2: 0})  # 1=bom, 0=mal
X_encoded = pd.get_dummies(X)

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
print("Relatório de Classificação:\n")
print(classification_report(y_test, modelo.predict(X_test)))

# Diretório de saída
output_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)

# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode="classification"
)

def gerar_explicacao(instancia, nome_arquivo, titulo):
    predicao = int(modelo.predict([instancia])[0])
    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)
    
    fig = exp.as_pyplot_figure(label=predicao)
    plt.title(titulo)
    img_path = os.path.join(output_dir, f"{nome_arquivo}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    explicacoes = exp.as_list(label=predicao)
    frases_terminal = []
    frases_pdf = []
    for feat, val in explicacoes:
        if val > 0:
            frases_terminal.append(f"🟢 A característica {feat} contribuiu para classificar como Bom Pagador.")
            frases_pdf.append(f"A característica {feat} contribuiu para classificar como Bom Pagador.")
        else:
            frases_terminal.append(f"🔴 A característica {feat} contribuiu para classificar como Mal Pagador.")
            frases_pdf.append(f"A característica {feat} contribuiu para classificar como Mal Pagador.")
    return frases_terminal, frases_pdf, img_path

# Selecionar exemplos
inst_bom = X_test[y_test == 1].iloc[0]
inst_mal = X_test[y_test == 0].iloc[0]

# Explicações
frases_bom_terminal, frases_bom_pdf, img_bom = gerar_explicacao(inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?")
frases_mal_terminal, frases_mal_pdf, img_mal = gerar_explicacao(inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?")

# Terminal
print("\n✅ Explicações para Bom Pagador:")
for f in frases_bom_terminal:
    print("- " + f)

print("\n❌ Explicações para Mal Pagador:")
for f in frases_mal_terminal:
    print("- " + f)

# Explicação geral (terminal)
print("\n📘 O que define um Bom ou Mal Pagador:\n")
print("✅ Bom Pagador")
print("- Conta ativa, bom histórico de crédito, idade acima de 30 anos.")
print("- Crédito proporcional à renda, emprego estável, telefone ativo.")
print("❌ Mal Pagador")
print("- Sem conta ativa, histórico ruim, idade baixa, valores altos, instabilidade.")

# PDF
def gerar_pdf_consolidado():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_title("Explicações LIME - Bom e Mal Pagador")

    def adicionar_secao(titulo, frases, imagem_path):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, titulo, ln=True)
        pdf.set_font("Arial", '', 12)
        for frase in frases:
            pdf.multi_cell(0, 8, "- " + frase)
        pdf.ln(4)
        if os.path.exists(imagem_path):
            pdf.image(imagem_path, w=180)
        pdf.ln(10)

    # Seção 1
    adicionar_secao("Explicações - Bom Pagador", frases_bom_pdf, img_bom)
    # Seção 2
    adicionar_secao("Explicações - Mal Pagador", frases_mal_pdf, img_mal)
    # Seção 3 - explicação geral
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "O que define um Bom ou Mal Pagador", ln=True)
    pdf.set_font("Arial", '', 12)

    bom_txt = """Bom Pagador:
Um cliente é classificado como Bom Pagador quando possui um perfil que sugere baixo risco de inadimplência. Fatores positivos incluem:
- Ter conta bancária ativa com bom histórico.
- Bom histórico de crédito (pagamentos em dia).
- Idade acima de 30 anos.
- Valor do crédito proporcional à renda.
- Estabilidade no emprego.
- Objetivos claros (como aquisição de bens).
- Ter bens no nome (como imóvel ou veículo).
- Ter telefone ativo para contato.
"""
    mal_txt = """Mal Pagador:
Um cliente é classificado como Mal Pagador quando o modelo identifica risco de inadimplência. Fatores incluem:
- Ausência de conta ativa.
- Histórico de crédito ruim ou inexistente.
- Idade muito baixa.
- Valor do crédito desproporcional.
- Pouco tempo no emprego atual.
- Falta de garantias ou bens.
- Motivações de crédito arriscadas.
"""

    pdf.multi_cell(0, 8, bom_txt)
    pdf.multi_cell(0, 8, mal_txt)

    caminho_pdf = os.path.join(output_dir, "explicacoes_lime.pdf")
    pdf.output(caminho_pdf)
    print(f"\n📄 PDF gerado com sucesso: {caminho_pdf}")

gerar_pdf_consolidado()
