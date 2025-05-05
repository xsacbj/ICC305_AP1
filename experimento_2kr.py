import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os

# Criar pasta de resultados, se não existir
os.makedirs("results", exist_ok=True)

# Passo 1: entrada do número de fatores k e replicacoes r
k = int(input("Digite o número de fatores (2 a 5): "))
r = int(input("Digite o número de replicações (1 a 3): "))

assert 2 <= k <= 5, "k deve estar entre 2 e 5"
assert 1 <= r <= 3, "r deve estar entre 1 e 3"

# Passo 2: gerar tabela de sinais
fatores = [f"F{i+1}" for i in range(k)]
sinais = list(itertools.product([-1, 1], repeat=k))
tabela = pd.DataFrame(sinais, columns=fatores)
tabela.insert(0, 'I', 1)

# Gerar colunas de interações
for i in range(k):
    for j in range(i+1, k):
        tabela[f"{fatores[i]}{fatores[j]}"] = tabela[fatores[i]] * tabela[fatores[j]]

if k >= 3:
    for i in range(k):
        for j in range(i+1, k):
            for l in range(j+1, k):
                tabela[f"{fatores[i]}{fatores[j]}{fatores[l]}"] = tabela[fatores[i]] * tabela[fatores[j]] * tabela[fatores[l]]

# Coletar respostas do usuário
y = []
print("\nDigite os valores de y para cada linha da tabela, separados por espaço (ex: 12 13 11):")
for idx in range(len(tabela)):
    entrada = input(f"{tabela.iloc[idx].to_dict()}: ").split()
    valores = list(map(float, entrada))
    assert len(valores) == r, "Deve fornecer exatamente r valores para cada experimento"
    y.append(valores)

# Calcular respostas médias por experimento
y = np.array(y)
y_medias = y.mean(axis=1)

# Adicionar colunas R1, R2, ..., Rr e Média
tabela_respostas = pd.DataFrame(y, columns=[f"R{i+1}" for i in range(r)])
tabela_respostas['Média'] = y_medias

# Estimar os efeitos
tabela_full = pd.concat([tabela, tabela_respostas], axis=1)
efeitos = {}
for coluna in tabela.columns:
    efeitos[coluna] = np.sum(tabela_full[coluna] * tabela_full['Média']) / len(tabela_full)

# Calcular modelo estimado para cada experimento
y_estimado = np.zeros_like(y_medias)
for i in range(len(tabela)):
    soma = 0
    for coluna in tabela.columns:
        soma += efeitos[coluna] * tabela.loc[i, coluna]
    y_estimado[i] = soma

# Calcular erros experimentais
erros = y - y_estimado[:, None]

# Adicionar colunas de erro E1, E2, ..., Er
tabela_erros = pd.DataFrame(erros, columns=[f"E{i+1}" for i in range(r)])
tabela_full = pd.concat([tabela_full, tabela_erros], axis=1)

# Adicionar somatório ponderado apenas para colunas de sinais (I, F1, ..., interações)
soma_ponderada = {}
media_ponderada = {}
for coluna in tabela.columns:
    soma_ponderada[coluna] = np.sum(tabela_full[coluna] * tabela_full['Média'])
    media_ponderada[coluna] = soma_ponderada[coluna] / len(tabela_full)

# Inserir linhas personalizadas no final da tabela
linha_soma = {col: soma_ponderada.get(col, '') for col in tabela_full.columns}
linha_media = {col: media_ponderada.get(col, '') for col in tabela_full.columns}
tabela_full.loc['Soma ponderada'] = linha_soma
tabela_full.loc['Média ponderada'] = linha_media

# Soma dos quadrados total (SST)
y_flat = y.flatten()
y_global = y_flat.mean()
SST = np.sum((y_flat - y_global)**2)

# Calcular soma dos quadrados dos efeitos
SS_efeitos = {nome: (val**2)*r*(2**k) for nome, val in efeitos.items() if nome != 'I'}
SSE = np.sum(erros**2)

# Tabela final com efeitos e proporção da variação explicada
resultado = pd.DataFrame({
    'Efeito': efeitos,
    'SS': {k: v for k, v in SS_efeitos.items()}
}).fillna(0)
resultado['% Var Explicada'] = 100 * resultado['SS'] / SST

# Adicionar SSE
resultado.loc['Erro Experimental'] = [0, SSE, 100 * SSE / SST]

# Exibir tabelas no terminal
print("\nTabela de Sinais")
print(tabela_full.round(2))

print("\nEfeitos e Porção da Variação Explicada:")
print(resultado.round(2))

# # Gerar gráfico de pizza da variação explicada
# resultado_plot = resultado[resultado['% Var Explicada'] > 0]
# plt.figure(figsize=(8, 8))
# plt.pie(resultado_plot['% Var Explicada'], labels=resultado_plot.index, autopct='%1.1f%%', startangle=90)
# plt.title('Distribuição da Variação Explicada (%)')
# plt.axis('equal')
# plt.tight_layout()
# plt.savefig('results/variacao_explicada_pizza.png')
