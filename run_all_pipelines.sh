#!/bin/bash

echo "Iniciando processamento de datasets..."

ROOT_DIR=$(pwd)


# Detectar comando Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Erro: Python não encontrado. Instale o Python para rodar os pipelines."
    exit 1
fi

# Detectar comando Pip
if command -v pip3 &>/dev/null; then
    PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD=""
fi

if [ -n "$PIP_CMD" ]; then
    echo "=========================================================="
    echo "Instalando dependências necessárias..."
    $PIP_CMD install --break-system-packages tqdm pandas numpy || $PIP_CMD install tqdm pandas numpy || echo "Aviso: Falha ao instalar dependências com pip."
fi

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

find "$ROOT_DIR/raw_data" -type f \( -name "process*.py" -o -name "processar*.py" \) | while read script; do
    script_dir=$(dirname "$script")
    script_name=$(basename "$script")
    
    echo "=========================================================="
    echo "Acessando diretório: $script_dir"
    cd "$script_dir" || continue
    
    echo "Executando $script_name..."
    "$PYTHON_CMD" "$script_name"
    
    cd "$ROOT_DIR"
done

echo "=========================================================="
echo "Todos os scripts foram executados com sucesso!"
