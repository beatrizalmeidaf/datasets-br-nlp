# 🇧🇷 Datasets de NLP em Português Brasileiro

Esse repositório é uma coleção curada de datasets para Processamento de Linguagem Natural (PLN) focados no português brasileiro. O objetivo é centralizar e facilitar o acesso a recursos para treinamento e avaliação de modelos de NLP em nosso idioma, principalmente para o cenário few shot.

## Estrutura do Repositório

O repositório está organizado em pastas, onde cada uma representa uma categoria de tarefa de NLP:

```
/
├── reviews/
│   ├── B2WReviewsCorpus/
│   │   └── few_shot/
│   ├── BrandsCorpus/
│   │   └── few_shot/
│   └── ReProCorpus/
│       └── few_shot/
│
├── analise_de_sentimentos/
│   └── (em breve...)
│
├── classificacao_de_intencao/
│   └── (em breve...)

```

## Datasets Disponíveis

### Avaliações de Produtos e Serviços

Datasets que contêm textos de avaliações de usuários sobre produtos e serviços, geralmente acompanhados de uma nota (ex: 1 a 5 estrelas). Ideal para tarefas de regressão ou classificação de sentimentos baseada em notas.

  * **B2W Reviews Corpus**

      * **Descrição:** Um grande conjunto de avaliações de produtos do e-commerce brasileiro (Americanas, Submarino, etc.), extraído do dataset público da B2W. Contém o texto da avaliação, título e nota de 1 a 5 estrelas.
      * **Formato:** Versões processadas em formato `few-shot` com 5 folds de validação cruzada.
      * **Localização:** `[./reviews/B2WReviewsCorpus/](./reviews/B2WReviewsCorpus/)`

  * **Brands Corpus**

      * **Descrição:** Dataset com avaliações de produtos focadas em marcas específicas. Inclui texto da avaliação, título e nota de 1 a 5 estrelas.
      * **Formato:** Versões processadas em `few-shot` com 5 folds.
      * **Localização:** `[./reviews/BrandsCorpus/](./reviews/BrandsCorpus/)`

  * **RePro Corpus**

      * **Descrição:** Dataset de avaliações de produtos com foco em reviews que mencionam problemas ou elogios específicos sobre a experiência de compra.
      * **Formato:** Versões processadas em `few-shot` com 5 folds.
      * **Localização:** `[./reviews/ReProCorpus/](./reviews/ReProCorpus/)`


### Análise de Sentimentos

Essa categoria inclui datasets focados na classificação de polaridade (positivo, negativo, neutro) de textos diversos, que não se limitam a avaliações de produtos.

  * *(Em breve...)*


### Classificação de Intenção

Datasets para identificar a intenção do usuário em uma frase ou diálogo (ex: perguntar, comprar, reclamar, elogiar).

  * *(Em breve...)*


## Como Usar

Os datasets na pasta `few_shot` estão em formato JSON, prontos para serem carregados para treinamento e avaliação em frameworks de machine learning.

