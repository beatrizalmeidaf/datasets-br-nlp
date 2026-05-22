#!/bin/bash

echo "Iniciando processamento de datasets..."

ROOT_DIR=$(pwd)


# ---------------------------------------------------------------------------
# Detectar Python real (evita alias do Windows Store em Git Bash)
# ---------------------------------------------------------------------------
find_python() {
    # 1. Caminhos fixos comuns no Windows
    local WIN_PATHS=(
        "/c/Python314/python.exe"
        "/c/Python313/python.exe"
        "/c/Python312/python.exe"
        "/c/Python311/python.exe"
        "/c/Python310/python.exe"
        "/c/Python39/python.exe"
        "/c/Program Files/Python314/python.exe"
        "/c/Program Files/Python313/python.exe"
        "/c/Program Files/Python312/python.exe"
        "/c/Program Files/Python311/python.exe"
        "/c/Users/$USER/AppData/Local/Programs/Python/Python314/python.exe"
        "/c/Users/$USER/AppData/Local/Programs/Python/Python313/python.exe"
        "/c/Users/$USER/AppData/Local/Programs/Python/Python312/python.exe"
        "/c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe"
    )
    for p in "${WIN_PATHS[@]}"; do
        if [ -f "$p" ]; then
            if "$p" --version &>/dev/null 2>&1; then
                echo "$p"
                return 0
            fi
        fi
    done

    # 2. Fallback: python3 / python via PATH — valida que imprime "Python 3.x"
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" --version 2>&1)
            if echo "$ver" | grep -q "Python 3"; then
                echo "$cmd"
                return 0
            fi
        fi
    done

    return 1
}

PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "Erro: Python 3 não encontrado."
    echo "Instale o Python em https://www.python.org/downloads/ e certifique-se de marcá-lo no PATH."
    exit 1
fi

echo "Python encontrado: $PYTHON_CMD ($("$PYTHON_CMD" --version 2>&1))"


# ---------------------------------------------------------------------------
# Instalar dependências usando o pip do mesmo Python encontrado
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "Instalando dependências necessárias..."
"$PYTHON_CMD" -m pip install --quiet tqdm pandas numpy scikit-learn openpyxl \
    || echo "Aviso: Falha ao instalar algumas dependências."


# ---------------------------------------------------------------------------
# Verificar / baixar datasets pendentes
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "Verificando e baixando datasets pendentes..."

# MMLU
cd "$ROOT_DIR/raw_data/category/mmlu" || exit
if [ ! -f "mmlu_PT-BR.csv" ] || [ ! -s "mmlu_PT-BR.csv" ] || grep -q "Entry not found" "mmlu_PT-BR.csv"; then
    echo "Baixando MMLU..."
    curl -L -o mmlu_PT-BR.csv "https://huggingface.co/datasets/openai/MMMLU/resolve/main/test/mmlu_PT-BR.csv"
fi

cd "$ROOT_DIR" || exit


# ---------------------------------------------------------------------------
# Executar scripts de processamento
# Nota: usamos um array para evitar problemas de subshell com while+pipe
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "Executando scripts de processamento..."

# Coleta scripts em array (evita subshell do while|pipe)
mapfile -t SCRIPTS < <(find "$ROOT_DIR/raw_data" -type f \( -name "process*.py" -o -name "processar*.py" \) | sort)

FAILED=0

for script in "${SCRIPTS[@]}"; do
    script_dir=$(dirname "$script")
    script_name=$(basename "$script")

    echo "=========================================================="
    echo "Acessando diretório: $script_dir"
    cd "$script_dir" || { echo "ERRO: Não foi possível acessar $script_dir"; FAILED=$((FAILED + 1)); continue; }

    echo "Executando $script_name..."
    if "$PYTHON_CMD" "$script_name"; then
        echo "OK: $script_name concluído com sucesso."
    else
        echo "ERRO: $script_name falhou com código $?."
        FAILED=$((FAILED + 1))
    fi

    cd "$ROOT_DIR" || exit
done

echo "=========================================================="
if [ "$FAILED" -eq 0 ]; then
    echo "Todos os scripts foram executados com sucesso!"
else
    echo "Atenção: $FAILED script(s) falharam. Verifique as mensagens acima."
    exit 1
fi
