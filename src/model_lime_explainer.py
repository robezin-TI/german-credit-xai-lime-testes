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

# === 1. Caminhos e colunas ===
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '..', 'data', 'german.data')
output_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)

colunas = [
    "status_conta", "duração", "histórico_crédito", "propósito", "valor_crédito",
    "conta_poupança", "residência_atual", "emprego_desde", "taxa_parcelamento",
    "sexo_estado_civil", "garantia", "residência", "idade", "outros_planos",
    "habitação", "n_creditos", "trabalhador_estrangeiro", "profissão",
    "telefone", "status", "classe"
]

# === 2. Leitura e pré-processamento ===
df = pd.read_csv(data_path, sep=' ', header=None, names=colunas)
X = df.drop("classe", axis=1)
y = df["classe"].map({1: 1, 2: 0})  # 1 = bom, 0 = mal pagador
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# === 3. Treinamento ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

print("Relatório de Classificação:\n")
print(classification_report(y_test, modelo.predict(X_test)))

# === 4. Explicador LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["Mal Pagador", "Bom Pagador"],
    mode="classification"
)

# === 5. Função para gerar gráfico + explicações ===
def gerar_explicacao(instancia, nome_arquivo, titulo):
    pred = int(modelo.predict([instancia])[0])
    exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=10)
    
    fig = exp.as_pyplot_figure(label=pred)
    plt.title(titulo)
    img_path = os.path.join(output_dir, f"{nome_arquivo}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    
    explicacoes = exp.as_list(label=pred)
    frases_txt = []
    frases_pdf = []

    for feat, val in explicacoes:
        emoji = "🟢" if val > 0 else "🔴"
        classe_txt = "Bom Pagador" if val > 0 else "Mal Pagador"
        frases_txt.append(f"- {emoji} A característica {feat} contribuiu para classificar como {classe_txt}.")
        frases_pdf.append(f"- A característica {feat} contribuiu para classificar como {classe_txt}.")
    
    return frases_txt, frases_pdf, img_path

# === 6. Selecionar exemplos ===
inst_bom = X_test[y_test == 1].iloc[0]
inst_mal = X_test[y_test == 0].iloc[0]

frases_bom_txt, frases_bom_pdf, img_bom = gerar_explicacao(
    inst_bom, "grafico_bom_pagador", "Por que o modelo classificou como 'Bom Pagador'?"
)
frases_mal_txt, frases_mal_pdf, img_mal = gerar_explicacao(
    inst_mal, "grafico_mal_pagador", "Por que o modelo classificou como 'Mal Pagador'?"
)

# === 7. Mostrar no terminal ===
print("\n✅ Explicações para Bom Pagador:")
for frase in frases_bom_txt:
    print(frase)

print("\n❌ Explicações para Mal Pagador:")
for frase in frases_mal_txt:
    print(frase)

# === 8. PDF consolidado ===
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Explicações das Classificações com LIME", ln=True)

def adicionar_secao(titulo, img_path, explicacoes):
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, titulo, ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for frase in explicacoes:
        pdf.multi_cell(0, 8, frase)
    pdf.ln(3)
    if os.path.exists(img_path):
        pdf.image(img_path, w=170)
    else:
        pdf.cell(0, 10, "[Gráfico não encontrado]", ln=True)

adicionar_secao("✅ Cliente Bom Pagador", img_bom, frases_bom_pdf)
adicionar_secao("❌ Cliente Mal Pagador", img_mal, frases_mal_pdf)

# === 9. Explicações gerais ===
pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "📘 O que define um Bom ou Mal Pagador:", ln=True)

pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "✅ Bom Pagador", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 7, """Um cliente é classificado como Bom Pagador quando possui um perfil que sugere baixo risco de inadimplência. Entre os principais fatores que influenciam positivamente estão:

- Ter uma conta bancária ativa com bom histórico de movimentação.
- Apresentar um bom histórico de crédito (pagamentos anteriores em dia).
- Ter uma idade mais avançada, geralmente acima dos 30 anos.
- Solicitar valores de crédito mais baixos ou proporcionais à renda.
- Estar empregado há mais tempo, demonstrando estabilidade profissional.
- Ter objetivos de crédito claros e seguros.
- Possuir bens no nome (como carro ou imóvel).
- Ter telefone ativo.""")

pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "❌ Mal Pagador", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 7, """Um cliente é classificado como Mal Pagador quando o modelo identifica características associadas a maior risco de inadimplência. Entre os principais fatores estão:

- Ausência de conta bancária ativa ou movimentações suspeitas.
- Histórico de crédito ruim ou inexistente.
- Idade muito baixa, indicando pouca experiência financeira.
- Solicitação de valores elevados de crédito, desproporcionais à estabilidade demonstrada.
- Pouco tempo no emprego atual.
- Falta de reserva financeira.
- Motivações de crédito mais arriscadas.
- Ausência de patrimônio registrado ou garantias.""")

# === 10. Salvar PDF ===
pdf.output(os.path.join(output_dir, "explicacoes_lime.pdf"))
print("\n📄 PDF gerado em: data/explicacoes_lime.pdf")
