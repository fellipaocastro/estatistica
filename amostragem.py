import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def amostragem_aleatoria_simples(dataset, amostras, seed=None):
    return dataset.sample(n=amostras, random_state=seed)


def amostragem_sistematica(dataset, amostras, seed=None):
    intervalo = len(dataset) // amostras
    random.seed(seed)
    inicio = random.randint(0, intervalo)
    indices = np.arange(inicio, len(dataset), step=intervalo)
    amostra_sistematica = dataset.iloc[indices]

    return amostra_sistematica


def amostragem_agrupamento(dataset, amostras, seed=None):
    grupos = []
    id_grupo = 0
    contagem = 0

    for _ in dataset.iterrows():
        grupos.append(id_grupo)
        contagem += 1

        if contagem >= amostras:
            contagem = 0
            id_grupo += 1

    dataset['grupo'] = grupos
    numero_grupos = len(dataset) // amostras
    random.seed(seed)
    grupo_selecionado = random.randint(0, numero_grupos - 1)

    return dataset[dataset['grupo'] == grupo_selecionado]


def amostragem_estratificada(dataset, amostras, seed=None):
    percentual = amostras / len(dataset)
    split = StratifiedShuffleSplit(test_size=percentual, random_state=seed)

    _, test_index = next(split.split(dataset, dataset['income']))

    return dataset.iloc[test_index]


def amostragem_reservatorio(dataset, amostras, seed=None):
    tamanho = len(dataset)
    stream = np.arange(tamanho)

    reservatorio = np.zeros(amostras, dtype=np.int32)

    i = 0

    for i in range(amostras):
        reservatorio[i] = stream[i]

    random.seed(seed)

    while i < tamanho:
        j = random.randrange(i + 1)
        if j < amostras:
            reservatorio[j] = stream[i]
        i += 1

    return dataset.iloc[reservatorio]


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

    print('\nAmostragem de reservatório')
    df_amostra_reservatorio = amostragem_reservatorio(ds_census, 100)
    print(df_amostra_reservatorio.shape)
    print(df_amostra_reservatorio)

    print('\nComparativo dos resultados')
    print(f"len(ds_census): {len(ds_census)}")
    print(f"ds_census['age'].mean(): {ds_census['age'].mean()}")
    print(f"df_amostra_aleatoria_simples['age'].mean(): {df_amostra_aleatoria_simples['age'].mean()}")
    print(f"df_amostra_sistematica['age'].mean(): {df_amostra_sistematica['age'].mean()}")
    print(f"df_amostra_agrupamento['age'].mean(): {df_amostra_agrupamento['age'].mean()}")
    print(f"df_amostra_estratificada['age'].mean(): {df_amostra_estratificada['age'].mean()}")
    print(f"df_amostragem_reservatorio['age'].mean(): {df_amostra_reservatorio['age'].mean()}")
