import pandas as pd

from amostragem import amostragem_aleatoria_simples, amostragem_sistematica, amostragem_agrupamento


ds_census = pd.read_csv('datasets/census.csv')
SEED = 1


def test_amostragem_aleatoria_simples_shape():
    df_amostra_aleatoria_simples = amostragem_aleatoria_simples(ds_census, 100, SEED)

    assert df_amostra_aleatoria_simples.shape == (100, 15)


def test_amostragem_aleatoria_simples_start():
    df_amostra_aleatoria_simples = amostragem_aleatoria_simples(ds_census, 100, SEED)

    assert df_amostra_aleatoria_simples.index[0] == 9646


def test_amostragem_aleatoria_simples_end():
    df_amostra_aleatoria_simples = amostragem_aleatoria_simples(ds_census, 100, SEED)

    assert df_amostra_aleatoria_simples.index[-1] == 551


def test_amostragem_sistematica_shape():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica.shape == (100, 15)


def test_amostragem_sistematica_start():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica.index[0] == 68


def test_amostragem_sistematica_end():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica.index[-1] == 32243


def test_amostragem_agrupamento_shape():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_agrupamento.shape == (326, 16)


def test_amostragem_agrupamento_start():
    df_amostra_sistematica = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_sistematica.index[0] == 5542


def test_amostragem_agrupamento_end():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_agrupamento.index[-1] == 5867
