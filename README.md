# 🇧🇷 Datasets de NLP em Português Brasileiro

Esse repositório reúne uma coleção curada de **datasets para Processamento de Linguagem Natural (PLN)** focados no **português brasileiro**.
O objetivo é centralizar recursos, facilitar o acesso e disponibilizar versões processadas para **cenários de Few-shot Learning**, com folds padronizados para experimentação reprodutível.


## Estrutura do Repositório

Os datasets estão organizados por categoria de tarefa:

```
/
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
```


## Datasets Disponíveis


## **Avaliações, Reviews e Análise de Sentimentos**

Datasets contendo textos avaliativos ou opiniões rotulados com **polaridade** (positivo/negativo). A maioria possui versões few-shot com 5 folds.

### **B2W Reviews Corpus**

* **Descrição:** Avaliações de produtos de e-commerces brasileiros (Americanas, Submarino, Shoptime).
* **Localização:** `./reviews/B2WReviewsCorpus/`

### **Brands Corpus**

* **Descrição:** Avaliações focadas em marcas específicas.
* **Localização:** `./reviews/BrandsCorpus/`

### **Buscape Corpus**

* **Descrição:** Reviews coletados da plataforma Buscapé, com notas e avaliações textuais.
* **Localização:** `./reviews/BuscapeCorpus/`

### **Kaggle Tweets Corpus**

* **Descrição:** Tweets rotulados com polaridade positiva/negativa, versão adaptada para PT-BR.
* **Localização:** `./reviews/KaggleTweetsCorpus/`

### **Olist Corpus**

* **Descrição:** Avaliações de clientes da base pública da Olist.
* **Localização:** `./reviews/OlistCorpus/`

### **RePro Corpus**

* **Descrição:** Reviews com foco em elogios e problemas relatados durante a experiência de compra.
* **Localização:** `./reviews/ReProCorpus/`

### **UTL Corpus**

* **Descrição:** Dataset de polaridade textual PT-BR amplamente usado em pesquisas.
* **Localização:** `./reviews/UTLCorpus/`


## **Classificação de Intenção**

Datasets para identificar a intenção do usuário em frases, diálogos ou documentos.

### **IntentPTCorpus**

* **Descrição:** Corpus de intenções em PT-BR baseado no conjunto de dados da Amazon Alexa.
* **Tarefas:** Identificação de intenções (ex.: comprar, solicitar, perguntar, elogiar).
* **Localização:** `./intent/IntentPTCorpus/`

### **CourtDecisionCorpus**

* **Descrição:** Corpus jurídico com classificações de intenção e decisão judicial.
* **Tarefas:** Intenção/propósito de petições e documentos.
* **Localização:** `./intent/CourtDecisionCorpus/`


## Como Usar

Todos os datasets em `few_shot/` seguem o mesmo padrão:

* Formato: **JSON**
* Estrutura:

  * `fold_1/`, `fold_2/`, ..., `fold_5/`
  * Arquivos de treino, validação e teste padronizados
