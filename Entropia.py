import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

dados = pd.read_csv('dados_covid.csv')

# Seleção das colunas desejadas
colunas_dados = ['Casos']  # professor, substitua pela coluna de dados que mais lhe interessar, para saber as possibilidades, abra o arquivo em csv pra ver as colunas.
dados = dados[colunas_dados]

#probabilidade
quantidade_elementos = dados.size
probabilidades = dados.groupby(colunas_dados).size() / quantidade_elementos

#entropia
entropia = -np.sum(probabilidades * np.log2(probabilidades))

#entropia máxima
entropia_maxima = np.log2(len(probabilidades))

# usando o k-NN para cálculo da entropia
matriz_dados = dados.values
num_linhas = len(matriz_dados)

k = min(5, num_linhas - 1)

knn = NearestNeighbors(n_neighbors=k + 1)
knn.fit(matriz_dados)

distancias, indices_vizinhos = knn.kneighbors(matriz_dados)
distancias = distancias[:, 1:]
indices_vizinhos = indices_vizinhos[:, 1:]

entropias = []
for i in range(num_linhas):
    vizinhos = indices_vizinhos[i]
    dists = distancias[i]
    dists[dists == 0] = 1e-10
    proporcoes = 1 / dists
    proporcoes /= np.sum(proporcoes)
    entropia_local = -np.sum(proporcoes * np.log2(proporcoes))
    entropias.append(entropia_local)

entropia_media = np.mean(entropias)

print("Entropia aproximada:", entropia)
print("Entropia máxima aproximada:", entropia_maxima)
print("Entropia média aproximada (k-NN):", entropia_media)
