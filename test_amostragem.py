import pandas as pd
from pandas import testing as tm

from amostragem import amostragem_aleatoria_simples, amostragem_sistematica, amostragem_agrupamento,\
    amostragem_estratificada, amostragem_reservatorio


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


def test_amostragem_aleatoria_simples_mean_age():
    df_amostra_aleatoria_simples = amostragem_aleatoria_simples(ds_census, 100, SEED)

    assert df_amostra_aleatoria_simples['age'].mean() == 39.41


def test_amostragem_sistematica_shape():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica.shape == (100, 15)


def test_amostragem_sistematica_start():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica.index[0] == 68


def test_amostragem_sistematica_end():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica.index[-1] == 32243


def test_amostragem_sistematica_mean_age():
    df_amostra_sistematica = amostragem_sistematica(ds_census, 100, SEED)

    assert df_amostra_sistematica['age'].mean() == 37.57


def test_amostragem_agrupamento_shape():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_agrupamento.shape == (326, 16)


def test_amostragem_agrupamento_value_counts():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    tm.assert_series_equal(
        df_amostra_agrupamento['grupo'].value_counts(),
        pd.Series(data={17: 326}, name='count', index=pd.Index([17], dtype='int64', name='grupo')))


def test_amostragem_agrupamento_start():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_agrupamento.index[0] == 5542


def test_amostragem_agrupamento_end():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_agrupamento.index[-1] == 5867


def test_amostragem_agrupamento_mean_age():
    df_amostra_agrupamento = amostragem_agrupamento(ds_census, 100, SEED)

    assert df_amostra_agrupamento['age'].mean() == 39.23312883435583


def test_amostragem_estratificada_shape():
    df_amostra_estratificada = amostragem_estratificada(ds_census, 100, SEED)

    assert df_amostra_estratificada.shape == (100, 16)


def test_amostragem_estratificada_value_counts():
    df_amostra_estratificada = amostragem_estratificada(ds_census, 100, SEED)

    tm.assert_series_equal(
        df_amostra_estratificada['income'].value_counts(),
        pd.Series(data={' <=50K': 76, ' >50K': 24}, name='count',
                  index=pd.Index([' <=50K', ' >50K'], dtype='object', name='income')))


def test_amostragem_estratificada_start():
    df_amostra_estratificada = amostragem_estratificada(ds_census, 100, SEED)

    assert df_amostra_estratificada.index[0] == 5611


def test_amostragem_estratificada_end():
    df_amostra_estratificada = amostragem_estratificada(ds_census, 100, SEED)

    assert df_amostra_estratificada.index[-1] == 23215


def test_amostragem_estratificada_mean_age():
    df_amostra_estratificada = amostragem_estratificada(ds_census, 100, SEED)

    assert df_amostra_estratificada['age'].mean() == 36.9


def test_amostragem_reservatorio_shape():
    df_amostra_reservatorio = amostragem_reservatorio(ds_census, 100, SEED)

    assert df_amostra_reservatorio.shape == (100, 16)


def test_amostragem_reservatorio_start():
    df_amostra_reservatorio = amostragem_reservatorio(ds_census, 100, SEED)

    assert df_amostra_reservatorio.index[0] == 29608


def test_amostragem_reservatorio_end():
    df_amostra_reservatorio = amostragem_reservatorio(ds_census, 100, SEED)

    assert df_amostra_reservatorio.index[-1] == 25870


def test_amostragem_reservatorio_mean_age():
    df_amostra_reservatorio = amostragem_reservatorio(ds_census, 100, SEED)

    assert df_amostra_reservatorio['age'].mean() == 37.85
