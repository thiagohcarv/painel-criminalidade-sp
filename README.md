# Painel de Criminalidade — São Paulo

Projeto de Parceria | Semantix × EBAC — Profissão: Cientista de Dados

## Objetivo

Construir um painel interativo (mapa + filtros por tipo de ocorrência + linha do tempo) para o município de São Paulo, com uma camada de modelagem preditiva/classificação de risco, usando exclusivamente dados públicos.

## Fonte de dados

**Fonte principal: SPSafe** (Freitas, Clarindo & Aguiar, 2023 — DSW/SBC) — dataset acadêmico padronizado a partir dos boletins de ocorrência da SSP/SP, 2003-2022, com ETL completo (limpeza, padronização, enriquecimento).
- Repositório: https://github.com/julianabfreitas/SPSafe (dados hospedados em https://zenodo.org/records/16739645)
- Campos-chave: `COD_IBGE` (código do município padronizado), `PONTO_CRIME` (coordenada geográfica em WKT — já georreferenciado, sem necessidade de geocodificação manual), `CIDADE`, `DATA_OCORRENCIA`, `HORA_OCORRENCIA`.
- **Vantagem sobre o portal oficial da SSP-SP:** o portal oficial não oferece API nem download direto estruturado (é uma ferramenta de consulta interativa) e o campo de bairro no dado bruto é texto livre não normalizado. O SPSafe já resolve isso via join espacial com limites municipais do IBGE.
- Licença: MIT.
- Citação obrigatória: Freitas, J. B., Clarindo, J. P., & Aguiar, C. D. (2023). SPSafe: um dataset sobre dados de criminalidade no estado de São Paulo. DSW 2023, 48-57. DOI: 10.5753/dsw.2023.233945

**Fonte complementar: SSP-SP "Dados Mensais"** — série agregada estadual por natureza/mês (sem granularidade espacial), usada para contexto e série temporal do Estado.

## Estrutura do repositório

```
├── data/
│   ├── raw/          # dados brutos, como baixados da fonte (não versionar arquivos grandes)
│   └── processed/    # dados limpos/tratados, prontos para análise
├── notebooks/        # notebooks de EDA, modelagem e visualização
├── src/              # scripts reutilizáveis (limpeza, geocodificação, features)
├── reports/
│   └── figures/      # gráficos e exports para o relatório final
└── README.md
```

## Roteiro do projeto

1. [ ] Ingestão e inspeção da base bruta (colunas, período, qualidade)
2. [ ] Limpeza e normalização (especialmente campo de bairro)
3. [ ] Geocodificação (bairro/município → lat/long)
4. [ ] Análise exploratória (EDA)
5. [ ] Construção do painel/mapa interativo
6. [ ] Modelagem (a definir: índice de risco / forecasting / clusterização)
7. [ ] Avaliação do modelo
8. [ ] Documentação final e storytelling

## Status atual

🟢 Ocorrências atribuídas a distritos reais da capital. Pronto para EDA e modelagem.

### Progresso

- Dados brutos SPSafe (2021 e 2022) filtrados por bounding box geográfico → 743.261 registros.
- **Refinamento com malha de distritos oficial (GeoSampa/Prefeitura de SP, via `codigourbano/distritos-sp`):** join espacial (point-in-polygon) atribuiu cada ocorrência ao distrito e subprefeitura corretos.
  - 27,3% dos registros do bounding box caíam em **municípios vizinhos** (Guarulhos, Osasco, Diadema, etc.) — o retângulo geográfico não respeita os limites reais da cidade. Foram descartados corretamente.
  - 13.803 registros muito próximos da borda (~300m) foram recuperados via nearest-join, tratando imprecisão de coordenada.
  - **Total final validado: 554.383 ocorrências**, distribuídas em 96 distritos e 32 subprefeituras.
  - Distritos com maior volume: República, Sé, Capão Redondo, Consolação, São Mateus.
- Dado salvo em `data/processed/sp_capital_com_distrito.parquet`.
- Malha de distritos salva em `data/external/distritos-sp.geojson` (fonte: GeoSampa/Prefeitura de SP, via repositório `codigourbano/distritos-sp`).

### Decisão de modelagem

**Abordagem escolhida: Índice de Risco por Região (classificação).** Cada distrito+mês será classificado em Baixo/Médio/Alto risco por tipo de ocorrência, usando Árvore de Decisão vs. XGBoost (alinhado ao Módulo 41 do curso) com validação cruzada (Módulo 37).

### Próximos passos

- [ ] EDA completa: distribuição por tipo de ocorrência, sazonalidade, dia da semana/hora, comparação 2021 vs 2022
- [ ] Construir a variável-alvo (classe de risco por distrito/mês)
- [ ] Feature engineering (histórico, densidade populacional IBGE, tipo predominante de crime)
- [ ] Treinar e comparar Árvore de Decisão vs. XGBoost com cross-validation
- [ ] Construir painel com filtros por tipo de ocorrência e camada de risco

---
*Projeto em desenvolvimento contínuo, documentado a cada etapa.*
