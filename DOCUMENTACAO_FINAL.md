# Painel de Risco Criminal — São Paulo
### Projeto de Parceria | Semantix × EBAC — Profissão: Cientista de Dados

---

## 1. Problema de negócio

A percepção de risco criminal na cidade de São Paulo é fragmentada e pouco acessível: os dados públicos existem, mas estão dispersos em portais instáveis, sem padronização geográfica confiável. Ferramentas fechadas (como sistemas municipais de câmeras) resolvem isso via vigilância, mas são caras, não auditáveis e não replicáveis.

**Por que dados e Machine Learning ajudam a resolver isso:** transformar boletins de ocorrência brutos em um **índice de risco por região, atualizado mensalmente e comparável entre distritos**, permite que qualquer pessoa — gestor público, seguradora, morador — enxergue padrões que não são visíveis boletim a boletim. Machine Learning entra porque queremos **prever** o nível de risco do mês corrente usando apenas o histórico disponível até o mês anterior — útil para antecipação, não só descrição do passado.

---

## 2. Coleta de dados

### Fonte principal: SPSafe
Dataset acadêmico (Freitas, Clarindo & Aguiar, 2023 — DSW/SBC) construído a partir dos boletins de ocorrência oficiais da Secretaria de Segurança Pública de São Paulo (SSP-SP), com ETL, padronização e geolocalização já aplicados.

- **Repositório:** https://github.com/julianabfreitas/SPSafe (dados em https://zenodo.org/records/16739645)
- **Licença:** MIT
- **Citação:** Freitas, J. B., Clarindo, J. P., & Aguiar, C. D. (2023). *SPSafe: um dataset sobre dados de criminalidade no estado de São Paulo*. DSW 2023, 48-57.
- **Período utilizado:** 2021 e 2022 (arquivos anuais brutos, ~547 mil e ~720 mil registros respectivamente, todo o Estado de SP)
- **Campos principais usados:** `NATUREZA_APURADA`, `DATA_OCORRENCIA`, `LATITUDE`, `LONGITUDE`

### Fonte complementar: malha de distritos de São Paulo
- **Repositório:** https://github.com/codigourbano/distritos-sp (fonte primária: GeoSampa, Secretaria Municipal de Desenvolvimento Urbano)
- 96 distritos administrativos, agrupados em 32 subprefeituras.

### Fonte adicional (v2): população por distrito
- **Repositório:** Fundação SEADE, projeção de população por distrito do MSP (dadosabertos.des.sp.gov.br / repositorio.seade.gov.br)
- Dados quinquenais (2000, 2005, ..., 2050); usamos o ano **2020** (mais próximo do período de análise 2021-2022, já que o dado exato de 2022 não está nessa série).
- Usada para normalizar as ocorrências por 100 mil habitantes, evitando que o índice de risco reflita apenas fluxo de circulação.

### Por que SPSafe, e não o portal oficial diretamente
O portal oficial (ssp.sp.gov.br/estatistica) não oferece API nem download estruturado — funciona como ferramenta de consulta interativa, e esteve instável/inacessível durante o desenvolvimento do projeto (confirmado inclusive por artigo acadêmico independente que documentou a mesma limitação). O SPSafe resolve isso oferecendo os mesmos dados já tratados e hospedados de forma estável (Zenodo).

### Tratamento e limpeza aplicados
1. **Filtro geográfico municipal:** as coordenadas foram inicialmente filtradas por bounding box e, em seguida, por **join espacial (point-in-polygon)** contra os polígonos reais dos distritos de SP — o bounding box sozinho incluía indevidamente ~27% de registros de municípios vizinhos (Guarulhos, Osasco, Diadema), que foram corretamente descartados.
2. **Recuperação de borda:** 13.803 registros a até ~300m da fronteira do município foram recuperados via nearest-join (erro de precisão de coordenada, não erro de localização real).
3. **Descoberta de qualidade de dado:** os campos `CIDADE` (98% nulo) e `DEPARTAMENTO_CIRCUNSCRICAO`/`DEPARTAMENTO_ELABORACAO` (96% nulo) mostraram-se não confiáveis como filtro de município — descoberta feita empiricamente, documentada para não repetir o erro.
4. **Datas inválidas:** <0,3% dos registros tinham erro de digitação no ano da ocorrência (ex: 1887, 1921) — mantidos na base (não afetam a agregação mensal do período de interesse) mas excluídos de qualquer agregação temporal fora de 2021-2022.

**Resultado final:** 554.383 ocorrências válidas, geolocalizadas e atribuídas a um distrito real da capital paulista.

---

## 3. Análise exploratória (EDA)

- **Crescimento 2021→2022:** 246.690 → 302.708 ocorrências (+22,7%), consistente com a narrativa de retomada de mobilidade pós-pandemia. Ver `reports/figures/01_evolucao_mensal.png`.
- **Composição:** Roubo e Furto somados representam a maioria esmagadora das ocorrências: destaque para "Roubo - Outros" (103.867), "Furto - Outros" (100.419), "Furto de Veículo" (77.093) e "Roubo de Veículo" (72.288). Ver `reports/figures/02_por_categoria.png`.
- **Padrão temporal:** ocorrências concentram-se no período **noturno** (174.381) e no meio da semana (quinta-feira é o pico). Ver `reports/figures/03_heatmap_dia_periodo.png`.
- **Padrão espacial (volume absoluto):** os distritos com maior volume são República, Sé e Consolação (região central, alto fluxo comercial/turístico) e Capão Redondo, São Mateus, Jardim Ângela (periferia). Ver `reports/figures/04_top_distritos.png`.
- **Padrão espacial (normalizado por população — achado central da v2):** ao dividir pelo número de residentes de cada distrito (dados SEADE), o ranking muda drasticamente. **Sé lidera com ~47.800 ocorrências por 100 mil habitantes** — mais que o triplo do 2º colocado — seguido por Barra Funda, República, Brás e Pari: todos distritos centrais de baixíssima população residente e altíssimo fluxo de pessoas (comércio, transporte, trabalho). **Capão Redondo, São Mateus, Jardim Ângela, Sapopemba e Ipiranga — que apareciam no top 10 por volume — desaparecem completamente do top 10 por taxa.** Isso confirma empiricamente a limitação identificada na v1: ranking por volume bruto mede fluxo de circulação, não risco real por morador. Ver `reports/figures/06_volume_vs_taxa_percapita.png`.

---

## 4. Modelagem

### Definição do problema
**Classificação supervisionada:** prever a classe de risco (Baixo / Médio / Alto) de cada distrito em cada mês, usando **apenas informação disponível até o mês anterior** (nenhuma variável do próprio mês é usada como feature, para evitar vazamento de dados).

### Variável-alvo (v2 — normalizada por população)
**Taxa de ocorrências por 100 mil habitantes** por distrito/mês (não mais o volume bruto), discretizada em tercis, usando população por distrito (Fundação SEADE, ano-base 2020). **Os limiares de corte foram calculados exclusivamente com dados de treino** e depois aplicados ao teste — evita que a distribuição do futuro vaze para a definição das classes. A normalização corrige o viés identificado na EDA, em que distritos de alto fluxo comercial (Sé, República, Barra Funda) dominavam o ranking bruto sem necessariamente serem os de maior risco por morador.

### Features utilizadas
`MES` (sazonalidade), `LAG_1`, `LAG_2` (taxa dos 2 meses anteriores), `MEDIA_MOVEL_3` (média móvel de 3 meses da taxa), `TENDENCIA` (diferença entre lags), `PCT_ROUBO`/`PCT_FURTO`/`PCT_OUTROS` (perfil de composição criminal do mês anterior), `SUBPREFEITURA` (codificada).

*Duas features adicionais foram testadas (média histórica expandida e sinal agregado da subprefeitura) e descartadas por piorarem levemente o resultado no teste — mantidas fora do modelo final por simplicidade (navalha de Occam).*

### Split e validação
- **Split temporal** (não aleatório): treino = abril/2021 a agosto/2022 (1.631 linhas), teste = setembro a dezembro/2022 (384 linhas) — holdout estritamente posterior ao treino.
- **Cross-validation:** `TimeSeriesSplit` (4 folds) dentro do treino, respeitando a ordem cronológica — um `KFold` aleatório teria vazamento (treinaria com meses futuros para prever meses passados).
- **Otimização de hiperparâmetros:** `GridSearchCV` sobre o TimeSeriesSplit.

### O duelo: Árvore de Decisão vs. XGBoost (v2 — variável normalizada)

| Modelo | Melhores hiperparâmetros | F1-macro (CV, treino) | F1-macro (teste, holdout) |
|---|---|---|---|
| Árvore de Decisão | `max_depth=3, min_samples_leaf=20` | 0,759 | **0,730** |
| XGBoost | `max_depth=3, learning_rate=0.1, n_estimators=50` | 0,779 | **0,755** |

**Vencedor: XGBoost**, com precisão de 0,91 para a classe ALTO risco (recall 0,84) — o caso de maior interesse prático.

### Comparação com baselines

| Modelo | F1-macro (teste) |
|---|---|
| Baseline ingênuo (classe majoritária) | 0,104 |
| Árvore de Decisão | 0,730 |
| Baseline de persistência (repete a classe do mês anterior) | 0,726 |
| **XGBoost** | **0,755** |

**Diferente da v1 (variável não normalizada), aqui o XGBoost supera o baseline de persistência.** Isso é consistente com a hipótese de que a taxa per capita captura sinal real de risco (menos ruidoso que o volume bruto, que mistura população residente com fluxo transitório) — o modelo tem algo genuíno para aprender além de "repetir o mês passado". Esse é o principal ganho metodológico da v2 sobre a v1: não só o ranking ficou mais correto, o modelo também ficou mais útil.

*Nota histórica: a v1 deste projeto (sem normalização por população) obteve F1-macro de 0,666 para o XGBoost, sem superar o baseline de persistência (0,687) — documentado para registrar a evolução do raciocínio.*

Ver `reports/figures/05_comparacao_modelos_baseline.png` e `reports/figures/06_volume_vs_taxa_percapita.png`.

---

## 5. Visualização

O entregável visual principal é um **mapa coroplético interativo** dos 96 distritos de São Paulo, colorido pela classe de risco prevista pelo modelo (Verde = Baixo, Amarelo = Médio, Vermelho = Alto), com tooltip mostrando risco previsto vs. risco real e o total de ocorrências no mês.

Arquivo: `reports/figures/mapa_risco_previsto_dez2022.html`

---

## 6. Conclusões

1. **O pipeline de dados foi o maior desafio técnico, não a modelagem em si.** A maior parte do esforço do projeto foi validar e corrigir a qualidade espacial dos dados (campos nulos, bounding box vazando para cidades vizinhas, coordenadas corrompidas) — uma etapa frequentemente subestimada, mas que determina se qualquer modelo posterior é confiável.
2. **Normalizar por população mudou o resultado em dois níveis: o ranking e o modelo.** No ranking, distritos centrais de baixa população residente e alto fluxo (Sé, Barra Funda, República) saltaram para o topo, enquanto distritos periféricos populosos (Capão Redondo, São Mateus, Jardim Ângela) saíram do top 10 — o volume bruto media circulação, não risco por morador. No modelo, essa mesma correção fez o **XGBoost passar a superar o baseline de persistência** (F1-macro 0,755 vs. 0,726), algo que não acontecia com a variável não normalizada (0,666 vs. 0,687 na v1). A taxa per capita carrega sinal real que o modelo consegue aprender; o volume bruto era, em boa parte, ruído de composição populacional.
3. **O modelo XGBoost venceu a Árvore de Decisão** em ambas as versões, com boa precisão para a classe de maior interesse prático (ALTO risco: precisão 0,91 na v2).
4. **Valor prático do modelo:** além de superar a persistência, o XGBoost oferece probabilidades de classe (úteis para priorização por confiança) e incorpora explicitamente sazonalidade e composição do tipo de crime.
5. **Limitações reconhecidas:**
   - Apenas 2 anos de dado mensal (24 pontos temporais por distrito) — pouco para capturar sazonalidade robusta.
   - População por distrito é de 2020 (ano mais próximo disponível na série quinquenal SEADE), não exatamente 2021-2022 — aproximação razoável, mas não exata.
   - Ausência de variáveis socioeconômicas adicionais (renda, infraestrutura urbana) que poderiam explicar mais variação entre distritos.
6. **Próximos passos recomendados:** estender a série histórica além de 2021-2022, testar modelos de série temporal dedicados (ex: Prophet, ARIMA) como comparação adicional às árvores, incorporar features de autocorrelação espacial (efeito de transbordamento entre distritos vizinhos), e usar renda/densidade (SEADE) para explicar variação residual entre distritos.

---

## Estrutura do repositório

```
├── data/
│   ├── external/       # malha de distritos (GeoSampa)
│   ├── processed/       # dados tratados (parquet)
├── src/                  # scripts de limpeza, join espacial, features e modelagem
├── reports/
│   ├── figures/          # gráficos da EDA, mapa de risco, comparação de modelos
│   └── resultados_modelagem.json
├── README.md              # log de desenvolvimento do projeto
└── DOCUMENTACAO_FINAL.md  # este arquivo
```

## Como reproduzir

```bash
pip install -r requirements.txt
python src/clean_filter_sp_capital.py <SPSafe_2021.csv> <SPSafe_2022.csv> --out data/processed/sp_capital_2021_2022.parquet
python src/join_distrito.py
python src/build_features.py
python src/treinar_modelos.py
```
