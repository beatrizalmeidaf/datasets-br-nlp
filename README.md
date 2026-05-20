# 🇧🇷 Datasets de NLP em Português Brasileiro

Esse repositório reúne uma coleção curada de **datasets para Processamento de Linguagem Natural (PLN)** focados no **português brasileiro**.
O objetivo é centralizar recursos, facilitar o acesso e disponibilizar versões processadas para **cenários de Few-shot Learning**, com folds padronizados para experimentação reprodutível.

## Estrutura do Repositório

Os datasets estão organizados por categoria de tarefa:

```
/
├── raw_data/
│   ├── ... (arquivos originais e códigos utilizados para pré-processamento)
│
├── reviews/
│   ├── B2WReviewsCorpus/
│   │   └── few_shot/
│   ├── BrandsCorpus/
│   │   └── few_shot/
│   ├── BuscapeCorpus/
│   │   └── few_shot/
│   ├── KaggleTweetsCorpus/
│   │   └── few_shot/
│   ├── OlistCorpus/
│   │   └── few_shot/
│   ├── ReProCorpus/
│   │   └── few_shot/
│   └── UTLCorpus/
│       └── few_shot/
│
├── intent/
│   ├── IntentPTCorpus/
│   │   └── few_shot/
│   └── CourtDecisionCorpus/
│       └── few_shot/
│
├── hate/
│   ├── HateBRCorpus/
│   │   └── few_shot/  
│   └── TuPyCorpus/
│       └── few_shot/     
│
├── category/
│   ├── EniacCorpus/
│   │   └── few_shot/ 
│   ├── MMLU_PTBR_Corpus/
│   │   └── few_shot/  
│   ├── RecognasummCorpus/
│   │   └── few_shot/   
│   └── RulingBRCorpus/
│       └── few_shot/  
```

## Sobre a pasta `raw_data`

A pasta `raw_data/` contém todos os datasets em seus formatos brutos, exatamente como foram extraídos das fontes originais.
Dentro dela também estão incluídos todos os scripts utilizados para limpeza, normalização e transformação dos dados até chegarem às versões padronizadas disponibilizadas nas demais pastas do repositório.

Isso garante transparência total e permite que qualquer pessoa:

* Reproduza o pré-processamento;
* Adapte os scripts para suas próprias pesquisas;
* Verifique a integridade dos dados originais.


## Datasets Disponíveis

### Avaliações, Reviews e Análise de Sentimentos

Datasets contendo textos avaliativos ou opiniões rotulados com **polaridade** (positivo/negativo).
A maioria possui versões few-shot com 5 folds.

### B2W Reviews Corpus

* **Descrição:** Avaliações de produtos de e-commerces brasileiros (Americanas, Submarino, Shoptime).
* **Localização:** `./reviews/B2WReviewsCorpus/`

### Brands Corpus

* **Descrição:** Avaliações focadas em marcas específicas.
* **Localização:** `./reviews/BrandsCorpus/`

### Buscape Corpus

* **Descrição:** Reviews coletados da plataforma Buscapé, com notas e avaliações textuais.
* **Localização:** `./reviews/BuscapeCorpus/`

### Kaggle Tweets Corpus

* **Descrição:** Tweets rotulados com polaridade positiva/negativa, versão adaptada para PT-BR.
* **Localização:** `./reviews/KaggleTweetsCorpus/`

### Olist Corpus

* **Descrição:** Avaliações de clientes da base pública da Olist.
* **Localização:** `./reviews/OlistCorpus/`

### RePro Corpus

* **Descrição:** Reviews com foco em elogios e problemas relatados durante a experiência de compra.
* **Localização:** `./reviews/ReProCorpus/`

### UTL Corpus

* **Descrição:** Dataset de polaridade textual PT-BR amplamente usado em pesquisas.
* **Localização:** `./reviews/UTLCorpus/`



## Classificação de Intenção

### IntentPTCorpus

* **Descrição:** Corpus de intenções em PT-BR baseado no conjunto de dados da Amazon Alexa.
* **Tarefas:** Identificação de intenções (como comprar, solicitar, perguntar, elogiar).
* **Localização:** `./intent/IntentPTCorpus/`

### CourtDecisionCorpus

* **Descrição:** Corpus jurídico com classificações de intenção e decisão judicial.
* **Tarefas:** Intenção/propósito de petições e documentos.
* **Localização:** `./intent/CourtDecisionCorpus/`



## Detecção de Discurso de Ódio

### HateBRCorpus

* **Descrição:** Corpus brasileiro focado em discurso de ódio e linguagem ofensiva.
* **Localização:** `./hate/HateBRCorpus/`

### TuPyCorpus

* **Descrição:** Corpus brasileiro focado em discurso de ódio e linguagem ofensiva.
* **Localização:** `./hate/TuPy/`



## Classificação Geral por Categorias

### EniacCorpus

* **Descrição:** Dataset de classificação com base em avaliações de lugares em PT-BR.
* **Localização:** `./category/EniacCorpus/`
   
### MMLU_PTBR_Corpus

* **Descrição:** Versão em português brasileiro do benchmark MMLU, cobrindo diversas áreas do conhecimento.
* **Localização:** `./category/MMLU_PTBR_Corpus/`

### RecognasummCorpus

* **Descrição:** Dataset de classificação geral envolvendo múltiplas categorias temáticas.
* **Localização:** `./category/RecognasummCorpus/`

### RulingBRCorpus

* **Descrição:** Conjunto de decisões judiciais brasileiras estruturadas, adequado para tarefas de classificação jurídica supervisionada.
* **Localização:** `./category/RulingBRCorpus/`



## Como Usar
Para executar todos os scripts, utilize o comando:
`bash run_all_pipelines.sh`

Todos os datasets em `few_shot/` seguem o mesmo padrão:

* Formato: **JSON**
* Estrutura:

  * `fold_1/`, `fold_2/`, ..., `fold_5/`
  * Cada fold contém os mesmos exemplos embaralhados em diferentes divisões de treino/validação/teste.

### Uso no Dataloader

Os datasets em `few_shot/` contêm o conjunto completo de exemplos.
A escolha do número de amostras (k-shot) deve ser feita no código de carregamento, garantindo:

* Reprodutibilidade
* Menos redundância
* Maior compatibilidade entre benchmarks
