import pandas as pd
import json
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

INPUT_FILE_PATH = "rulingbr-v1.2.jsonl"

INPUT_TEXT_COLUMN = 'ementa'
INPUT_LABEL_COLUMN = 'area'

FINAL_TEXT_COLUMN = 'sentence'
FINAL_LABEL_COLUMN = 'label'

OUTPUT_BASE_DIR = "data"
DATASET_NAME = "RulingBRCorpus"
NUM_FOLDS = 5
RANDOM_SEED = 42


VALID_LABELS = [
    "direito administrativo",
    "direito ambiental",
    "direito civil",
    "direito constitucional",
    "direito do consumidor",
    "direito econômico",
    "direito eleitoral",
    "direito financeiro",
    "direito internacional público",
    "direito processual civil",
    "direito processual penal",
    "direito penal",
    "direito previdenciário",
    "direito do trabalho",
    "direito notarial",
    "direito tributário",
    "direito urbanístico"
]

def load_data_from_jsonl(file_path):
    """
    Carrega, extrai e LIMPA os dados do arquivo JSONL.
    """
    all_data = []
    print(f"Carregando e filtrando arquivo JSONL: {file_path}...")
    
    valid_labels_set = set(VALID_LABELS)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processando linhas"):
                try:
                    data = json.loads(line)
                    text = data.get(INPUT_TEXT_COLUMN)
                    label = data.get(INPUT_LABEL_COLUMN)
                    
                    if text and label and label in valid_labels_set:
                        all_data.append({
                            FINAL_TEXT_COLUMN: text,
                            FINAL_LABEL_COLUMN: label
                        })
                except json.JSONDecodeError:
                    print(f"Aviso: Ignorando linha mal formatada")
                    
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo {file_path}: {e}")
        return None
            
    if not all_data:
        print("ERRO: Nenhum dado foi carregado. Verifique o arquivo e os nomes das colunas.")
        return None
        
    df = pd.DataFrame(all_data)
    print(f"Total de {len(df)} amostras válidas carregadas e filtradas.")
    
    return df

def save_json_pool(dataframe, file_path):
    """
    Salva o DataFrame como um pool de dados em formato JSONL (JSON Lines).
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
    
    full_df = load_data_from_jsonl(INPUT_FILE_PATH)
    if full_df is None or len(full_df) == 0:
        print("ERRO: Nenhum dado processado. Encerrando.")
        return
    

    print(f"\nTotal de amostras antes da deduplicação: {len(full_df)}")
    df_deduplicated = full_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    print(f"Total de amostras únicas (após deduplicação): {len(df_deduplicated)}")


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
            print(f"Removendo diretório antigo: {output_path}")
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