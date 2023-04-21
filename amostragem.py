import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def amostragem_aleatoria_simples(dataset, amostras, seed=1):
    return dataset.sample(n=amostras, random_state=seed)


def amostragem_sistematica(dataset, amostras, seed=1):
    intervalo = len(dataset) // amostras
    random.seed(seed)
    inicio = random.randint(0, intervalo)
    indices = np.arange(inicio, len(dataset), step=intervalo)
    amostra_sistematica = dataset.iloc[indices]

    return amostra_sistematica


def amostragem_agrupamento(dataset, numero_grupos, seed=1):
    intervalo = len(dataset) // numero_grupos

    grupos = []
    id_grupo = 0
    contagem = 0

    for _ in dataset.iterrows():
        grupos.append(id_grupo)
        contagem += 1

        if contagem > intervalo:
            contagem = 0
            id_grupo += 1

    dataset['grupo'] = grupos
    random.seed(seed)
    grupo_selecionado = random.randint(0, numero_grupos)

    return dataset[dataset['grupo'] == grupo_selecionado]


def amostragem_estratificada(dataset, amostras, seed=1):
    percentual = amostras / len(dataset)
    split = StratifiedShuffleSplit(test_size=percentual, random_state=seed)

    _, test_index = next(split.split(dataset, dataset['income']))

    return dataset.iloc[test_index]


if __name__ == '__main__':
    ds_census = pd.read_csv('datasets/census.csv')

    print('\nAmostragem aleatória simples')
    df_amostra_aleatoria_simples = amostragem_aleatoria_simples(ds_census, 100)
    print(df_amostra_aleatoria_simples.shape)
    print(df_amostra_aleatoria_simples)

    print('\nAmostragem sistemática')
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100)
    print(df_amostra_sistematica.shape)
    print(df_amostra_sistematica)

    print('\nAmostragem por grupos')
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100)
    print(df_amostra_agrupamento.shape)
    print(df_amostra_agrupamento['grupo'].value_counts())
    print(df_amostra_agrupamento)

    print('\nAmostragem estratificada')
    df_amostra_estratificada = amostragem_estratificada(ds_census, 100)
    print(df_amostra_estratificada.shape)
    print(df_amostra_estratificada['income'].value_counts())
    print(df_amostra_estratificada)
