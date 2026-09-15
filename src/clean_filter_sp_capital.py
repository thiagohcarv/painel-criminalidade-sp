"""
Limpeza e filtragem dos dados brutos SPSafe para o municipio de Sao Paulo (capital).

Entrada: arquivos CSV brutos do SPSafe (um por ano), separador ';'.
Saida: um unico arquivo parquet/csv com apenas registros geolocalizados
       dentro do bounding box do municipio de Sao Paulo.

Uso:
    python clean_filter_sp_capital.py <arquivo1.csv> <arquivo2.csv> ... --out data/processed/sp_capital.parquet
"""
import argparse
import pandas as pd

# Bounding box aproximado do municipio de Sao Paulo (capital)
LAT_MIN, LAT_MAX = -24.05, -23.35
LON_MIN, LON_MAX = -46.83, -46.35

COLS_USO = [
    "NUM_BO", "ANO_BO", "NATUREZA_APURADA", "DATA_OCORRENCIA", "HORA_OCORRENCIA",
    "PERIODO_OCORRENCIA", "LOGRADOURO", "BAIRRO", "TIPO_LOCAL",
    "LATITUDE", "LONGITUDE", "FLAG_VITIMA_FATAL", "STATUS",
]


def processa_arquivo(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";", dtype=str, low_memory=False, usecols=COLS_USO)

    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    dentro_sp = lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX)
    df_sp = df.loc[dentro_sp].copy()
    df_sp["LATITUDE"] = lat.loc[dentro_sp]
    df_sp["LONGITUDE"] = lon.loc[dentro_sp]

    print(f"{caminho}: {len(df)} linhas brutas -> {len(df_sp)} em SP capital ({len(df_sp)/len(df)*100:.1f}%)")
    return df_sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivos", nargs="+", help="CSVs brutos do SPSafe")
    parser.add_argument("--out", required=True, help="Caminho de saida (.parquet ou .csv)")
    args = parser.parse_args()

    partes = [processa_arquivo(f) for f in args.arquivos]
    resultado = pd.concat(partes, ignore_index=True)

    # Parse de datas
    resultado["DATA_OCORRENCIA"] = pd.to_datetime(resultado["DATA_OCORRENCIA"], errors="coerce")

    print(f"\nTotal combinado: {len(resultado)} registros")
    print(f"Periodo: {resultado['DATA_OCORRENCIA'].min()} a {resultado['DATA_OCORRENCIA'].max()}")

    if args.out.endswith(".parquet"):
        resultado.to_parquet(args.out, index=False)
    else:
        resultado.to_csv(args.out, index=False, sep=";")
    print(f"Salvo em: {args.out}")


if __name__ == "__main__":
    main()
