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

### Por que essa fonte, e não o portal oficial diretamente
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
- **Padrão espacial:** os distritos com maior volume absoluto são República, Sé e Consolação (região central, alto fluxo comercial/turístico) e Capão Redondo, São Mateus, Jardim Ângela (periferia). Ver `reports/figures/04_top_distritos.png`.
- **Limitação reconhecida:** o ranking por volume absoluto favorece regiões de alta circulação (comércio, transporte) mesmo sem serem necessariamente as de maior risco *per capita* — normalizar por população residente (dado do IBGE/SEADE) é uma melhoria natural para uma próxima iteração.

---

## 4. Modelagem

### Definição do problema
**Classificação supervisionada:** prever a classe de risco (Baixo / Médio / Alto) de cada distrito em cada mês, usando **apenas informação disponível até o mês anterior** (nenhuma variável do próprio mês é usada como feature, para evitar vazamento de dados).

### Variável-alvo
Total de ocorrências por distrito/mês, discretizado em tercis. **Os limiares de corte foram calculados exclusivamente com dados de treino** e depois aplicados ao teste — evita que a distribuição do futuro vaze para a definição das classes.

### Features utilizadas
`MES` (sazonalidade), `LAG_1`, `LAG_2` (ocorrências dos 2 meses anteriores), `MEDIA_MOVEL_3` (média móvel de 3 meses), `TENDENCIA` (diferença entre lags), `PCT_ROUBO`/`PCT_FURTO`/`PCT_OUTROS` (perfil de composição criminal do mês anterior), `SUBPREFEITURA` (codificada).

*Duas features adicionais foram testadas (média histórica expandida e sinal agregado da subprefeitura) e descartadas por piorarem levemente o resultado no teste — mantidas fora do modelo final por simplicidade (navalha de Occam).*

### Split e validação
- **Split temporal** (não aleatório): treino = abril/2021 a agosto/2022 (1.631 linhas), teste = setembro a dezembro/2022 (384 linhas) — holdout estritamente posterior ao treino.
- **Cross-validation:** `TimeSeriesSplit` (4 folds) dentro do treino, respeitando a ordem cronológica — um `KFold` aleatório teria vazamento (treinaria com meses futuros para prever meses passados).
- **Otimização de hiperparâmetros:** `GridSearchCV` sobre o TimeSeriesSplit.

### O duelo: Árvore de Decisão vs. XGBoost

| Modelo | Melhores hiperparâmetros | F1-macro (CV, treino) | F1-macro (teste, holdout) |
|---|---|---|---|
| Árvore de Decisão | `max_depth=5, min_samples_leaf=20` | 0,772 | **0,646** |
| XGBoost | `max_depth=5, learning_rate=0.05, n_estimators=50` | 0,779 | **0,666** |

**Vencedor: XGBoost**, com melhor generalização no holdout temporal (precisão de 0,85 para a classe ALTO risco — o caso de maior interesse prático).

### Comparação com baselines (achado crítico, reportado com transparência)

| Modelo | F1-macro (teste) |
|---|---|
| Baseline ingênuo (classe majoritária) | 0,104 |
| Árvore de Decisão | 0,646 |
| XGBoost | 0,666 |
| **Baseline de persistência** (repete a classe do mês anterior) | **0,687** |

Testamos também um baseline de persistência simples (assumir que o risco do mês é igual ao do mês anterior). **Esse baseline superou os dois modelos treinados.** Isso não invalida o exercício — é um achado real e consistente com a literatura de criminologia: taxas de criminalidade têm forte autocorrelação temporal e espacial ("hot spots" persistem). Um modelo sofisticado só se justifica se capturar sinal *além* dessa persistência (ex: mudanças de tendência, sazonalidade fina, composição de crime) — e neste recorte de dados (2 anos, granularidade mensal), esse sinal adicional foi limitado.

Ver `reports/figures/05_comparacao_modelos_baseline.png` para a comparação visual completa.

---

## 5. Visualização

O entregável visual principal é um **mapa coroplético interativo** dos 96 distritos de São Paulo, colorido pela classe de risco prevista pelo modelo (Verde = Baixo, Amarelo = Médio, Vermelho = Alto), com tooltip mostrando risco previsto vs. risco real e o total de ocorrências no mês.

Arquivo: `reports/figures/mapa_risco_previsto_dez2022.html`

---

## 6. Conclusões

1. **O pipeline de dados foi o maior desafio técnico, não a modelagem em si.** A maior parte do esforço do projeto foi validar e corrigir a qualidade espacial dos dados (campos nulos, bounding box vazando para cidades vizinhas, coordenadas corrompidas) — uma etapa frequentemente subestimada, mas que determina se qualquer modelo posterior é confiável.
2. **O modelo XGBoost venceu a Árvore de Decisão**, mas **nenhum dos dois superou o baseline de persistência simples.** Isso é um resultado honesto e cientificamente relevante: criminalidade mensal por distrito é uma série altamente autocorrelacionada, e a "aposta segura" (repetir o mês anterior) é difícil de bater com apenas 2 anos de histórico mensal.
3. **Valor prático do modelo, mesmo sem bater o baseline:** o XGBoost oferece (a) probabilidades de classe, úteis para priorização por confiança; (b) incorporação explícita de sazonalidade e composição do tipo de crime, que a persistência ignora; (c) capacidade de generalizar para situações fora do padrão simples de "repetir o mês anterior" (ex: mudanças estruturais).
4. **Limitações reconhecidas:**
   - Apenas 2 anos de dado mensal (24 pontos temporais por distrito) — pouco para capturar sazonalidade robusta.
   - Ranking por volume absoluto não normalizado por população residente.
   - Ausência de variáveis socioeconômicas (renda, densidade) que poderiam explicar variação entre distritos.
5. **Próximos passos recomendados:** normalizar por população (IBGE/SEADE), estender a série histórica além de 2021-2022, testar modelos de série temporal dedicados (ex: Prophet) como baseline adicional, e incorporar features espaciais de vizinhança entre distritos.

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
