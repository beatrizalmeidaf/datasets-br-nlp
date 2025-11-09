import pandas as pd
import json
import shutil
from pathlib import Path
import numpy as np

INPUT_FILE_PATH = "brandsBr.xlsx"
INPUT_TEXT_COLUMN = 'review_text'
INPUT_TITLE_COLUMN = 'review_title'
INPUT_LABEL_COLUMN = 'overall_rating'

FINAL_TEXT_COLUMN = 'sentence'

OUTPUT_BASE_DIR = "data"
DATASET_NAME = "BrandsCorpus"
NUM_FOLDS = 5
RANDOM_SEED = 42

LABEL_MAP = {
    1: "Negativo",
    2: "Negativo",
    4: "Positivo",
    5: "Positivo"
}

VALID_LABELS = list(LABEL_MAP.keys()) # [1, 2, 4, 5]


def load_data_from_excel(file_path):
    """
    Carrega, combina, limpa e traduz os dados do arquivo Excel BrandsCorpus.
    """
    print(f"Carregando dados do arquivo Excel: '{file_path}'...")
    try:
        # pip install openpyxl
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
        return None
    except ImportError:
        print("ERRO: Pacote 'openpyxl' não encontrado. Instale com: pip install openpyxl")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo Excel: {e}")
        return None

    print(f"Total de {len(df)} amostras carregadas. Limpando e preparando os dados...")

    df[FINAL_TEXT_COLUMN] = df[INPUT_TITLE_COLUMN].astype(str).fillna('') + ' ' + df[INPUT_TEXT_COLUMN].astype(str).fillna('')

    df[INPUT_LABEL_COLUMN] = pd.to_numeric(df[INPUT_LABEL_COLUMN], errors='coerce')
    
    initial_rows = len(df)
    
    df.dropna(subset=[FINAL_TEXT_COLUMN, INPUT_LABEL_COLUMN], inplace=True)
    
    df[INPUT_LABEL_COLUMN] = df[INPUT_LABEL_COLUMN].astype(int)

    df = df[df[INPUT_LABEL_COLUMN].isin(VALID_LABELS)].copy()
    print(f"{initial_rows - len(df)} linhas com dados inválidos ou labels indesejadas (como '3') foram removidas.")


    print("Traduzindo labels para o formato descritivo...")
    df[INPUT_LABEL_COLUMN] = df[INPUT_LABEL_COLUMN].map(LABEL_MAP)

    clean_df = df[[FINAL_TEXT_COLUMN, INPUT_LABEL_COLUMN]].copy()

    print(f"Total final: {len(clean_df)} amostras limpas e traduzidas.")
    
    return clean_df


def save_json_pool(dataframe, file_path):
    """
    Salva o DataFrame como um pool de dados em formato JSONL (JSON Lines).
    """

    data_list = dataframe.rename(columns={INPUT_LABEL_COLUMN: 'label'}).to_dict('records')
    
    print(f"Salvando {len(data_list)} registros em JSONL em: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')


def main():
    """
    Função principal que orquestra a criação dos folds de validação cruzada.
    """
    full_df = load_data_from_excel(INPUT_FILE_PATH)
    if full_df is None:
        return

    print(f"\nTotal de amostras antes da deduplicação: {len(full_df)}")
    df_deduplicated = full_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    removidas = len(full_df) - len(df_deduplicated)
    print(f"Total de amostras únicas (após deduplicação): {len(df_deduplicated)} (removidas {removidas} duplicatas)")

    print(f"\nEmbaralhando o dataset único com a semente {RANDOM_SEED}...")
    
    df_shuffled = df_deduplicated.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

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