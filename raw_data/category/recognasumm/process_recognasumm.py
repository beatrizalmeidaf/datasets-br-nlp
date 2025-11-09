import pandas as pd
import json
import shutil
from pathlib import Path
import numpy as np

INPUT_FILES = ["train.jsonl", "validation.jsonl", "test.jsonl"]

INPUT_TEXT_COLUMN = 'Noticia'    
INPUT_LABEL_COLUMN = 'Categoria' 

FINAL_TEXT_COLUMN = 'sentence'
FINAL_LABEL_COLUMN = 'label'

OUTPUT_BASE_DIR = "data"
DATASET_NAME = "RecognasummCorpus"
NUM_FOLDS = 5
RANDOM_SEED = 42


LABEL_MAP = {
    # saúde e bem-estar
    "Bem-Estar": "saúde e bem-estar",
    "Ciência e Saúde": "saúde e bem-estar",
    "Saúde": "saúde e bem-estar",
    "saúde": "saúde e bem-estar",
    "VIVA BEM": "saúde e bem-estar",
    
    # política
    "Política": "política",
    "política": "política",
    "Governo Lula": "política", 
    
    # entretenimento
    "Entretenimento": "entretenimento e cultura",
    "entretenimento": "entretenimento e cultura",
    "Pop e Arte": "entretenimento e cultura",
    
    # esportes
    "Esporte": "esportes",
    "Esportes": "esportes",
    
    # mundo
    "Internacional": "notícia internacional",
    "Mundo": "notícia internacional",
    
    # turismo
    "Turismo e Gastronomia": "turismo e viagem",
    "Turismo e Viagem": "turismo e viagem",
    
    # labels únicas
    "Brasil": "notícia sobre o Brasil",
    "Ciência e Tecnologia": "ciência e tecnologia",
    "Economia": "economia",
    "Educação": "educação",
    "Jornais e Programas": "mídia e programas",
    "Meio-Ambiente": "meio ambiente",
    "Podcast": "podcast",
}



def load_and_merge_splits(files_list):
    """
    Carrega, junta, extrai e limpa os dados de múltiplos arquivos JSONL.
    """
    all_data = []
    print("Carregando e juntando arquivos JSONL...")
    
    for file_path in files_list:
        print(f"Lendo '{file_path}'...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get(INPUT_TEXT_COLUMN)
                        label = data.get(INPUT_LABEL_COLUMN)
                        
                        if text and label:
                            all_data.append({
                                FINAL_TEXT_COLUMN: text,
                                FINAL_LABEL_COLUMN: label
                            })
                    except json.JSONDecodeError:
                        print(f"Aviso: Ignorando linha mal formatada em {file_path}")
                        
        except FileNotFoundError:
            print(f"AVISO: O arquivo '{file_path}' não foi encontrado. Pulando.")
        except Exception as e:
            print(f"Ocorreu um erro ao ler o arquivo {file_path}: {e}")
            
    if not all_data:
        print("ERRO: Nenhum dado foi carregado. Verifique os nomes dos arquivos e colunas.")
        return None
        
    df = pd.DataFrame(all_data)
    print(f"Total de {len(df)} amostras carregadas de {len(files_list)} arquivos.")
    
    initial_rows = len(df)
    
    df.dropna(subset=[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN], inplace=True)
    
    print(f"{initial_rows - len(df)} linhas com dados inválidos foram removidas.")

    if LABEL_MAP:
        print("Traduzindo labels (Categoria) para o formato descritivo...")
        mapped_labels = df[FINAL_LABEL_COLUMN].map(LABEL_MAP)
        
        unmapped_mask = mapped_labels.isna()
        if unmapped_mask.any():
            unmapped_labels_list = df.loc[unmapped_mask, FINAL_LABEL_COLUMN].unique()
            print(f"ATENÇÃO: As seguintes {len(unmapped_labels_list)} labels não foram encontradas no mapa e serão DESCARTADAS:")
            print(unmapped_labels_list)
            
            df = df[~unmapped_mask].copy()
            df[FINAL_LABEL_COLUMN] = df[FINAL_LABEL_COLUMN].map(LABEL_MAP)
        else:
            print("Todas as labels foram mapeadas com sucesso.")
            df[FINAL_LABEL_COLUMN] = mapped_labels
            
    else:
        print("Nenhum LABEL_MAP fornecido. Usando 'Categoria' originais como labels.")
    
    print(f"Total final: {len(df)} amostras limpas e traduzidas.")
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
    
    full_df = load_and_merge_splits(INPUT_FILES)
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