#!/bin/bash

# Este script encontra e executa todos os scripts de processamento na pasta raw_data
# E também limpa os arquivos .json e .jsonl antigos.

echo "Iniciando limpeza e processamento de datasets..."

ROOT_DIR=$(pwd)

echo "Limpando todos os arquivos .json e .jsonl antigos..."
# Encontra todas as pastas data dentro de raw_data e remove json/jsonl
find "$ROOT_DIR/raw_data" -type f \( -name "*.json" -o -name "*.jsonl" \) -delete 2>/dev/null
for d in "data" "category" "hate" "intent" "reviews"; do
    find "$ROOT_DIR/$d" -type f \( -name "*.json" -o -name "*.jsonl" \) -delete 2>/dev/null
done

echo "=========================================================="
echo "Instalando dependências necessárias..."
pip install tqdm

echo "=========================================================="
echo "Verificando e baixando datasets pendentes..."

# MMLU
cd "$ROOT_DIR/raw_data/category/mmlu" || exit
if [ ! -f "mmlu_PT-BR.csv" ] || [ ! -s "mmlu_PT-BR.csv" ] || grep -q "Entry not found" "mmlu_PT-BR.csv"; then
    echo "Baixando MMLU..."
    curl -L -o mmlu_PT-BR.csv "https://huggingface.co/datasets/openai/MMMLU/resolve/main/test/mmlu_PT-BR.csv"
fi

cd "$ROOT_DIR" || exit

echo "=========================================================="
echo "Executando scripts de processamento..."

find "$ROOT_DIR/raw_data" -type f -name "process*.py" | while read script; do
    script_dir=$(dirname "$script")
    script_name=$(basename "$script")
    
    echo "=========================================================="
    echo "Acessando diretório: $script_dir"
    cd "$script_dir" || continue
    
    echo "Executando $script_name..."
    python "$script_name"
    
    cd "$ROOT_DIR"
done

echo "=========================================================="
echo "Todos os scripts foram executados com sucesso!"
