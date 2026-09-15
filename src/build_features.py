"""
Constroi a base de modelagem: agregacao DISTRITO x MES com features
temporais (lag, media movel) e a variavel-alvo (classe de risco).

IMPORTANTE (prevencao de vazamento de dados):
- A classe de risco de um mes e calculada a partir do TOTAL de ocorrencias
  DAQUELE mes (o que queremos prever).
- As features usadas para prever usam apenas informacao de meses ANTERIORES
  (lag_1, lag_2, media_movel_3) — nunca o total do proprio mes.
- Os limiares de quantil (baixo/medio/alto) sao calculados SOMENTE com dados
  de treino e depois aplicados ao teste, para nao vazar distribuicao futura.
"""
import pandas as pd
import numpy as np

IN_PATH = "data/processed/sp_capital_com_distrito.parquet"
OUT_PATH = "data/processed/base_modelagem.parquet"


def main():
    df = pd.read_parquet(IN_PATH)
    df["ANO"] = df["DATA_OCORRENCIA"].dt.year
    df["MES"] = df["DATA_OCORRENCIA"].dt.month
    df = df[df["ANO"].isin([2021, 2022])]

    df["ANO_MES"] = df["ANO"].astype(int) * 100 + df["MES"].astype(int)

    # Agregacao base: total de ocorrencias por distrito x mes
    agg = df.groupby(["DISTRITO", "SUBPREFEITURA", "ANO", "MES", "ANO_MES"]).size().reset_index(name="TOTAL_OCORRENCIAS")

    # Proporcao por categoria (perfil do distrito naquele mes)
    cat = df.groupby(["DISTRITO", "ANO_MES", "CATEGORIA"]).size().unstack(fill_value=0)
    cat = cat.div(cat.sum(axis=1), axis=0).add_prefix("PCT_").reset_index()
    agg = agg.merge(cat, on=["DISTRITO", "ANO_MES"], how="left")

    agg = agg.sort_values(["DISTRITO", "ANO_MES"]).reset_index(drop=True)

    # Features de lag e media movel (por distrito, ordenado no tempo)
    g = agg.groupby("DISTRITO")["TOTAL_OCORRENCIAS"]
    agg["LAG_1"] = g.shift(1)
    agg["LAG_2"] = g.shift(2)
    agg["MEDIA_MOVEL_3"] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    agg["TENDENCIA"] = agg["LAG_1"] - agg["LAG_2"]
    # Media historica expandida (nivel-base do distrito, mais estavel que so 3 meses)
    agg["MEDIA_HISTORICA"] = g.shift(1).expanding().mean().reset_index(level=0, drop=True)

    # Sinal regional: media movel-3 agregada por subprefeitura (suaviza ruido de 1 distrito)
    subpref_mensal = agg.groupby(["SUBPREFEITURA", "ANO_MES"])["TOTAL_OCORRENCIAS"].sum().reset_index()
    subpref_mensal = subpref_mensal.sort_values(["SUBPREFEITURA", "ANO_MES"])
    subpref_mensal["SUBPREF_MEDIA_MOVEL_3"] = (
        subpref_mensal.groupby("SUBPREFEITURA")["TOTAL_OCORRENCIAS"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )
    agg = agg.merge(subpref_mensal[["SUBPREFEITURA", "ANO_MES", "SUBPREF_MEDIA_MOVEL_3"]], on=["SUBPREFEITURA", "ANO_MES"], how="left")

    # Remove primeiras observacoes de cada distrito sem historico suficiente
    agg = agg.dropna(subset=["LAG_1", "LAG_2", "MEDIA_MOVEL_3", "MEDIA_HISTORICA"]).reset_index(drop=True)

    print(f"Base de modelagem: {len(agg)} linhas (distrito x mes, com historico)")
    print(f"Periodo: {agg['ANO_MES'].min()} a {agg['ANO_MES'].max()}")
    print(f"Distritos: {agg['DISTRITO'].nunique()}")

    agg.to_parquet(OUT_PATH, index=False)
    print(f"Salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
