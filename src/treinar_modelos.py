"""
Treina e compara Arvore de Decisao vs XGBoost para prever a classe de
risco (Baixo/Medio/Alto) de um distrito em um dado mes, usando apenas
features defasadas (sem vazamento de dados).

Split temporal: treino = ate 2022-08, teste = 2022-09 a 2022-12.
Cross-validation: TimeSeriesSplit dentro do treino (respeita a ordem
temporal, ao contrario de um KFold aleatorio).
Otimizacao de hiperparametros: GridSearchCV sobre o TimeSeriesSplit.
"""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

IN_PATH = "data/processed/base_modelagem.parquet"
CORTE_TESTE = 202209  # treino: < 202209 | teste: >= 202209

FEATURES = [
    "MES", "LAG_1", "LAG_2", "MEDIA_MOVEL_3", "TENDENCIA",
    "PCT_FURTO", "PCT_ROUBO", "PCT_OUTROS", "SUBPREF_ENC",
]


def definir_classe_risco(serie_treino, serie_completa):
    """Define limiares de tercil usando SOMENTE o treino, aplica em tudo."""
    q1, q2 = serie_treino.quantile([1 / 3, 2 / 3])

    def classifica(x):
        if x <= q1:
            return "BAIXO"
        elif x <= q2:
            return "MEDIO"
        else:
            return "ALTO"

    return serie_completa.apply(classifica), (q1, q2)


def main():
    df = pd.read_parquet(IN_PATH)
    df = df.fillna({"PCT_FURTO": 0, "PCT_ROUBO": 0, "PCT_OUTROS": 0, "PCT_MORTE/HOMICIDIO": 0})

    le_subpref = LabelEncoder()
    df["SUBPREF_ENC"] = le_subpref.fit_transform(df["SUBPREFEITURA"])

    treino_mask = df["ANO_MES"] < CORTE_TESTE
    teste_mask = ~treino_mask

    # Limiares de risco calculados so no treino (evita vazamento da distribuicao futura)
    classe, (q1, q2) = definir_classe_risco(df.loc[treino_mask, "TOTAL_OCORRENCIAS"], df["TOTAL_OCORRENCIAS"])
    df["CLASSE_RISCO"] = classe
    print(f"Limiares de risco (definidos so com dados de treino): Q1={q1:.1f} | Q2={q2:.1f}")
    print(df["CLASSE_RISCO"].value_counts())
    print()

    le_target = LabelEncoder()
    df["CLASSE_RISCO_ENC"] = le_target.fit_transform(df["CLASSE_RISCO"])

    X_train, y_train = df.loc[treino_mask, FEATURES], df.loc[treino_mask, "CLASSE_RISCO_ENC"]
    X_test, y_test = df.loc[teste_mask, FEATURES], df.loc[teste_mask, "CLASSE_RISCO_ENC"]
    print(f"Treino: {len(X_train)} linhas (ate {CORTE_TESTE-1}) | Teste: {len(X_test)} linhas (a partir de {CORTE_TESTE})")
    print()

    tscv = TimeSeriesSplit(n_splits=4)
    resultados = {}

    # --- Arvore de Decisao ---
    grid_arvore = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={"max_depth": [3, 5, 7, 10], "min_samples_leaf": [5, 10, 20]},
        cv=tscv, scoring="f1_macro", n_jobs=-1,
    )
    grid_arvore.fit(X_train, y_train)
    melhor_arvore = grid_arvore.best_estimator_
    pred_arvore = melhor_arvore.predict(X_test)

    resultados["Arvore de Decisao"] = {
        "melhores_params": grid_arvore.best_params_,
        "cv_f1_macro": grid_arvore.best_score_,
        "test_accuracy": accuracy_score(y_test, pred_arvore),
        "test_f1_macro": f1_score(y_test, pred_arvore, average="macro"),
    }

    # --- XGBoost ---
    grid_xgb = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="mlogloss"),
        param_grid={"max_depth": [3, 5], "n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        cv=tscv, scoring="f1_macro", n_jobs=-1,
    )
    grid_xgb.fit(X_train, y_train)
    melhor_xgb = grid_xgb.best_estimator_
    pred_xgb = melhor_xgb.predict(X_test)

    resultados["XGBoost"] = {
        "melhores_params": grid_xgb.best_params_,
        "cv_f1_macro": grid_xgb.best_score_,
        "test_accuracy": accuracy_score(y_test, pred_xgb),
        "test_f1_macro": f1_score(y_test, pred_xgb, average="macro"),
    }

    print("=== RESULTADOS DO DUELO ===")
    for nome, r in resultados.items():
        print(f"\n{nome}:")
        print(f"  Melhores hiperparametros: {r['melhores_params']}")
        print(f"  F1-macro (cross-validation, treino): {r['cv_f1_macro']:.3f}")
        print(f"  Acuracia (teste, holdout temporal): {r['test_accuracy']:.3f}")
        print(f"  F1-macro (teste, holdout temporal): {r['test_f1_macro']:.3f}")

    vencedor = "XGBoost" if resultados["XGBoost"]["test_f1_macro"] >= resultados["Arvore de Decisao"]["test_f1_macro"] else "Arvore de Decisao"
    print(f"\n>>> Modelo vencedor (maior F1-macro no teste): {vencedor}")

    modelo_final = melhor_xgb if vencedor == "XGBoost" else melhor_arvore
    pred_final = pred_xgb if vencedor == "XGBoost" else pred_arvore

    print("\nClassification report (modelo vencedor, teste):")
    print(classification_report(y_test, pred_final, target_names=le_target.classes_))

    matriz = confusion_matrix(y_test, pred_final)
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print(le_target.classes_)
    print(matriz)

    # Salva resultados e artefatos para o relatorio final
    with open("reports/resultados_modelagem.json", "w") as f:
        json.dump({
            "resultados": resultados,
            "vencedor": vencedor,
            "limiares_risco": {"q1": float(q1), "q2": float(q2)},
            "classes": list(le_target.classes_),
            "matriz_confusao": matriz.tolist(),
            "corte_treino_teste": CORTE_TESTE,
        }, f, indent=2, ensure_ascii=False)

    # Feature importance do vencedor
    if vencedor == "XGBoost":
        importancias = pd.Series(modelo_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    else:
        importancias = pd.Series(modelo_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nImportancia das features (modelo vencedor):")
    print(importancias)
    importancias.to_csv("reports/feature_importance.csv")

    # Predicoes finais para o mapa (distrito, ano_mes, real, previsto)
    saida = df.loc[teste_mask, ["DISTRITO", "SUBPREFEITURA", "ANO_MES", "TOTAL_OCORRENCIAS", "CLASSE_RISCO"]].copy()
    saida["CLASSE_PREVISTA"] = le_target.inverse_transform(pred_final)
    saida.to_parquet("data/processed/predicoes_teste.parquet", index=False)
    print("\nPredicoes de teste salvas em data/processed/predicoes_teste.parquet")


if __name__ == "__main__":
    main()
