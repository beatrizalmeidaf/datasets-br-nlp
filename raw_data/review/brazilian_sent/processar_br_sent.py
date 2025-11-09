import pandas as pd
import json
import shutil
from pathlib import Path
import numpy as np

LABEL_TO_USE = 'polarity'
LABEL_MAP = {
    0: "Negativo",
    1: "Positivo"
}
VALID_LABELS = list(LABEL_MAP.keys())

INPUT_TEXT_COLUMN = 'review_text'
FINAL_TEXT_COLUMN = 'sentence'
OUTPUT_BASE_DIR = "data"
NUM_FOLDS = 5
RANDOM_SEED = 42

DATASETS_TO_PROCESS = [
    {
        "input_file": "olist.csv",
        "corpus_name": "OlistCorpus"
    },
    {
        "input_file": "buscape.csv",
        "corpus_name": "BuscapeCorpus"
    },
]


def load_data_from_csv(file_path, label_column):
    print(f"Carregando dados do arquivo CSV: '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
        return None
    except UnicodeDecodeError:
        print("Erro de Unicode. Tentando carregar com 'latin-1'...")
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            print(f"Falha ao carregar com 'latin-1': {e}")
            return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao carregar o arquivo CSV: {e}")
        return None

    print(f"Total de {len(df)} amostras carregadas. Limpando e preparando dados...")

    if INPUT_TEXT_COLUMN not in df.columns or label_column not in df.columns:
        print(f"ERRO: Colunas esperadas '{INPUT_TEXT_COLUMN}' ou '{label_column}' não encontradas em '{file_path}'.")
        print(f"Colunas encontradas: {list(df.columns)}")
        return None

    df[FINAL_TEXT_COLUMN] = df[INPUT_TEXT_COLUMN].astype(str).fillna('')
    df[label_column] = pd.to_numeric(df[label_column], errors='coerce')

    initial_rows = len(df)

    df.dropna(subset=[FINAL_TEXT_COLUMN, label_column], inplace=True)
    df[label_column] = df[label_column].astype(int)

    df = df[df[label_column].isin(VALID_LABELS)].copy()

    clean_df = df[[FINAL_TEXT_COLUMN, label_column]].copy()
    clean_df[label_column] = clean_df[label_column].map(LABEL_MAP)

    print(f"{initial_rows - len(clean_df)} linhas com dados inválidos ou labels indesejadas foram removidas. Total final: {len(clean_df)} amostras.")
    return clean_df


def save_json_pool(dataframe, file_path, label_column):
    data_list = dataframe.rename(columns={label_column: 'label'}).to_dict('records')

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"ERRO ao salvar o arquivo {file_path}: {e}")


def process_single_dataset(input_file, dataset_name, label_column):
    print(f"PROCESSANDO: {dataset_name} (Arquivo: {input_file}, Rótulo: {label_column})")

    full_df = load_data_from_csv(input_file, label_column)
    if full_df is None:
        print(f"Falha ao carregar '{input_file}'. Pulando para o próximo.")
        return

    print(f"\nTotal de amostras antes da deduplicação: {len(full_df)}")
    df_deduplicated = full_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    removidas = len(full_df) - len(df_deduplicated)
    print(f"Total de amostras únicas (após deduplicação): {len(df_deduplicated)} (removidas {removidas} duplicatas)")

    print(f"\nEmbaralhando o dataset {dataset_name} (único) com a semente {RANDOM_SEED}...")
    df_shuffled = df_deduplicated.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Dividindo os dados únicos em {NUM_FOLDS} folds...")
    folds = np.array_split(df_shuffled, NUM_FOLDS)

    output_root = Path(OUTPUT_BASE_DIR) / dataset_name / "few_shot"
    print(f"Preparando pasta de saída: {output_root}")

    for i in range(NUM_FOLDS):
        fold_name = f"{i+1:02d}"
        print(f"Processando Fold {fold_name}/{NUM_FOLDS}")

        output_path = output_root / fold_name
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        df_test = folds[i].reset_index(drop=True)
        df_valid = folds[(i + 1) % NUM_FOLDS].reset_index(drop=True)
        train_folds_indices = [j for j in range(NUM_FOLDS) if j != i and j != (i + 1) % NUM_FOLDS]
        df_train = pd.concat([folds[j] for j in train_folds_indices]).reset_index(drop=True)

        print(f"Tamanhos: Treino={len(df_train)}, Validação={len(df_valid)}, Teste={len(df_test)}")

        save_json_pool(df_train, output_path / "train.json", label_column)
        save_json_pool(df_valid, output_path / "valid.json", label_column)
        save_json_pool(df_test, output_path / "test.json", label_column)

    print(f"\nPROCESSO CONCLUÍDO PARA: {dataset_name}")


def main():
    print("Iniciando processamento em lote...")

    if not DATASETS_TO_PROCESS:
        print("A lista 'DATASETS_TO_PROCESS' está vazia. Nenhum dataset para processar.")
        return

    for dataset in DATASETS_TO_PROCESS:
        process_single_dataset(
            input_file=dataset["input_file"],
            dataset_name=dataset["corpus_name"],
            label_column=LABEL_TO_USE
        )

    print("PROCESSAMENTO EM LOTE CONCLUÍDO!")
    print(f"Verifique a pasta '{OUTPUT_BASE_DIR}' para os resultados.")


if __name__ == "__main__":
    main()
