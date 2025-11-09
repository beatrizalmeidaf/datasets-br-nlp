import pandas as pd
import json
import shutil
from pathlib import Path
import numpy as np

INPUT_FILE_PATH = "dataset-eniac-2023.csv"
INPUT_TEXT_COLUMN = 'sentenca'
INPUT_LABEL_COLUMN = 'categoria'

FINAL_TEXT_COLUMN = 'sentence'
FINAL_LABEL_COLUMN = 'label'

OUTPUT_BASE_DIR = "data"
DATASET_NAME = "EniacCorpus"
NUM_FOLDS = 5
RANDOM_SEED = 42


def load_data_from_csv(file_path):
    """
    Carrega, extrai a primeira label e limpa os dados.
    """
    print(f"Carregando dados do arquivo CSV: '{file_path}'...")
    try:
        df = pd.read_csv(file_path, sep=',')
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo CSV: {e}")
        return None

    print(f"Total de {len(df)} amostras carregadas. Limpando e preparando os dados...")

    if INPUT_TEXT_COLUMN not in df.columns or INPUT_LABEL_COLUMN not in df.columns:
        print(f"ERRO: Colunas esperadas '{INPUT_TEXT_COLUMN}' ou '{INPUT_LABEL_COLUMN}' não encontradas.")
        return None

    initial_rows = len(df)
    
    df.dropna(subset=[INPUT_TEXT_COLUMN, INPUT_LABEL_COLUMN], inplace=True)
    
    print("Extraindo a primeira label e renomeando colunas...")
    
    df.rename(columns={INPUT_TEXT_COLUMN: FINAL_TEXT_COLUMN}, inplace=True)
    
    df[FINAL_LABEL_COLUMN] = df[INPUT_LABEL_COLUMN].astype(str).apply(lambda x: x.split(',')[0].strip())

    clean_df = df[[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN]].copy()
    
    clean_df.dropna(subset=[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN], inplace=True)
    clean_df = clean_df[
        (clean_df[FINAL_LABEL_COLUMN] != '') & (clean_df[FINAL_TEXT_COLUMN] != '')
    ]

    print(f"{initial_rows - len(clean_df)} linhas com dados inválidos ou labels vazias foram removidas. Total final: {len(clean_df)} amostras.")
    
    return clean_df

def save_json_pool(dataframe, file_path):
    """
    Salva o DataFrame como um pool de dados em formato JSONL (JSON Lines).
    O dataframe já deve conter as colunas 'sentence' e 'label'.
    """
    data_list = dataframe.to_dict('records')
    
    print(f"Salvando {len(data_list)} registros em JSONL em: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')

def main():
    """
    Função principal que orquestra a criação dos folds de validação cruzada.
    """
    
    full_df = load_data_from_csv(INPUT_FILE_PATH)
    if full_df is None or len(full_df) == 0:
        print("ERRO: Nenhum dado processado. Encerrando.")
        return

    # deduplicação
    print(f"\nTotal de amostras antes da deduplicação: {len(full_df)}")
    df_deduplicated = full_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    removidas = len(full_df) - len(df_deduplicated)
    print(f"Total de amostras únicas (após deduplicação): {len(df_deduplicated)} (removidas {removidas} duplicatas)")

    # embaralhar
    print(f"\nEmbaralhando o dataset único com a semente {RANDOM_SEED}...")
    df_shuffled = df_deduplicated.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # dividir em Folds
    print(f"Dividindo os dados únicos em {NUM_FOLDS} folds...")
    folds = np.array_split(df_shuffled, NUM_FOLDS)

    output_root = Path(OUTPUT_BASE_DIR) / DATASET_NAME / "few_shot"

    for i in range(NUM_FOLDS):
        fold_name = f"{i+1:02d}"
        print(f"\nProcessando Fold {fold_name}/{NUM_FOLDS}")
        
        output_path = output_root / fold_name
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        df_test = folds[i].reset_index(drop=True)
        df_valid = folds[(i + 1) % NUM_FOLDS].reset_index(drop=True)
        train_folds_indices = [j for j in range(NUM_FOLDS) if j != i and j != (i + 1) % NUM_FOLDS]
        df_train = pd.concat([folds[j] for j in train_folds_indices]).reset_index(drop=True)

        print(f"Tamanhos para o Fold {fold_name}: Treino={len(df_train)}, Validação={len(df_valid)}, Teste={len(df_test)}")

        print("Salvando pools de dados (.jsonl)...")
        save_json_pool(df_train, output_path / "train.json")
        save_json_pool(df_valid, output_path / "valid.json")
        save_json_pool(df_test, output_path / "test.json")

    print(f"\nPROCESSO CONCLUÍDO! {NUM_FOLDS} folds de validação cruzada foram criados em '{output_root}'")

if __name__ == "__main__":
    main()