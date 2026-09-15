"""
Join espacial: atribui cada ocorrencia criminal (lat/long) ao distrito
e subprefeitura correspondente do municipio de Sao Paulo.

Pontos fora dos poligonos mas a menos de ~300m da borda sao recuperados
via nearest-join (erro de precisao de coordenada). Os demais (fora do
municipio de Sao Paulo, em cidades vizinhas da Grande SP) sao descartados.

Entrada: data/processed/sp_capital_2021_2022.parquet
         data/external/distritos-sp.geojson
Saida:   data/processed/sp_capital_com_distrito.parquet
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

IN_OCORRENCIAS = "data/processed/sp_capital_2021_2022.parquet"
IN_DISTRITOS = "data/external/distritos-sp.geojson"
OUT = "data/processed/sp_capital_com_distrito.parquet"
LIMIAR_BORDA_GRAUS = 0.003  # ~300m


def main():
    df = pd.read_parquet(IN_OCORRENCIAS)
    print(f"Ocorrencias carregadas (bounding box): {len(df)}")

    distritos = gpd.read_file(IN_DISTRITOS)[["ds_nome", "ds_subpref", "geometry"]]
    distritos = distritos.rename(columns={"ds_nome": "DISTRITO", "ds_subpref": "SUBPREFEITURA"})

    geometry = [Point(xy) for xy in zip(df["LONGITUDE"], df["LATITUDE"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # 1. Join exato (within)
    exato = gpd.sjoin(gdf, distritos, how="left", predicate="within").drop(columns=["index_right"])

    sem_distrito = exato["DISTRITO"].isna()
    print(f"Sem distrito no join exato: {sem_distrito.sum()} ({sem_distrito.mean()*100:.2f}%)")

    # 2. Nearest-join para os que ficaram de fora, recuperando so os proximos da borda
    faltantes = exato.loc[sem_distrito, ["geometry"]].copy()
    proximo = gpd.sjoin_nearest(faltantes, distritos, how="left", distance_col="dist_borda")
    recuperados = proximo[proximo["dist_borda"] <= LIMIAR_BORDA_GRAUS]
    print(f"Recuperados por proximidade de borda (<=~300m): {len(recuperados)}")

    exato.loc[recuperados.index, "DISTRITO"] = recuperados["DISTRITO"]
    exato.loc[recuperados.index, "SUBPREFEITURA"] = recuperados["SUBPREFEITURA"]

    # 3. Descarta o que continua sem distrito (fora do municipio de SP)
    antes = len(exato)
    resultado = exato[exato["DISTRITO"].notna()].drop(columns=["geometry"]).copy()
    print(f"Descartados por estarem fora do municipio de SP: {antes - len(resultado)}")
    print(f"Total final (dentro de SP capital): {len(resultado)}")

    print("\nTop 10 distritos por volume de ocorrencias:")
    print(resultado["DISTRITO"].value_counts().head(10))

    resultado.to_parquet(OUT, index=False)
    print(f"\nSalvo em: {OUT}")


if __name__ == "__main__":
    main()

