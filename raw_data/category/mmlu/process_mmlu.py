"""
curl -L -o mmlu_PT-BR.csv https://huggingface.co/datasets/openai/MMMLU/resolve/main/test/mmlu_PT-BR.csv
"""

import pandas as pd
import json
import shutil
from pathlib import Path
import numpy as np

INPUT_FILE_PATH = "mmlu_PT-BR.csv"

INPUT_TEXT_COLUMN = 'Question'
INPUT_LABEL_COLUMN = 'Subject'

FINAL_TEXT_COLUMN = 'sentence'
FINAL_LABEL_COLUMN = 'label'

OUTPUT_BASE_DIR = "data"
DATASET_NAME = "MMLU_PTBR_Corpus"
NUM_FOLDS = 5
RANDOM_SEED = 42

LABEL_MAP = {
    "abstract_algebra": "álgebra abstrata",
    "anatomy": "anatomia",
    "astronomy": "astronomia",
    "business_ethics": "ética empresarial",
    "clinical_knowledge": "conhecimento clínico",
    "college_biology": "biologia (ensino superior)",
    "college_chemistry": "química (ensino superior)",
    "college_computer_science": "ciência da computação (ensino superior)",
    "college_mathematics": "matemática (ensino superior)",
    "college_medicine": "medicina (ensino superior)",
    "college_physics": "física (ensino superior)",
    "computer_security": "segurança da computação",
    "conceptual_physics": "física conceitual",
    "econometrics": "econometria",
    "electrical_engineering": "engenharia elétrica",
    "elementary_mathematics": "matemática elementar",
    "formal_logic": "lógica formal",
    "global_facts": "fatos globais",
    "high_school_biology": "biologia (ensino médio)",
    "high_school_chemistry": "química (ensino médio)",
    "high_school_computer_science": "ciência da computação (ensino médio)",
    "high_school_european_history": "história da europa (ensino médio)",
    "high_school_geography": "geografia (ensino médio)",
    "high_school_government_and_politics": "governo e política (ensino médio)",
    "high_school_macroeconomics": "macroeconomia (ensino médio)",
    "high_school_mathematics": "matemática (ensino médio)",
    "high_school_microeconomics": "microeconomia (ensino médio)",
    "high_school_physics": "física (ensino médio)",
    "high_school_psychology": "psicologia (ensino médio)",
    "high_school_statistics": "estatística (ensino médio)",
    "high_school_us_history": "história dos eua (ensino médio)",
    "high_school_world_history": "história mundial (ensino médio)",
    "human_aging": "envelhecimento humano",
    "human_sexuality": "sexualidade humana",
    "international_law": "direito internacional",
    "jurisprudence": "jurisprudência",
    "logical_fallacies": "falácias lógicas",
    "machine_learning": "aprendizado de máquina",
    "management": "administração",
    "marketing": "marketing",
    "medical_genetics": "genética médica",
    "miscellaneous": "conhecimentos gerais",
    "moral_disputes": "disputas morais",
    "moral_scenarios": "cenários morais",
    "nutrition": "nutrição",
    "philosophy": "filosofia",
    "prehistory": "pré-história",
    "professional_accounting": "contabilidade profissional",
    "professional_law": "direito profissional",
    "professional_medicine": "medicina profissional",
    "professional_psychology": "psicologia profissional",
    "public_relations": "relações públicas",
    "security_studies": "estudos de segurança",
    "sociology": "sociologia",
    "us_foreign_policy": "política externa dos eua",
    "virology": "virologia",
    "world_religions": "religiões mundiais"
}


def load_data_from_csv(file_path):
    """
    Carrega, extrai, limpa e traduz os dados do arquivo CSV MMLU.
    """
    print(f"Carregando dados do arquivo CSV: '{file_path}'...")
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception:
        print("Aviso: Falha ao ler com TAB")
        try:
            df = pd.read_csv(file_path, sep=',')
        except FileNotFoundError:
            print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
            return None
        except Exception as e2:
            print(f"ERRO: Falha ao carregar o arquivo CSV com TAB ou vírgula. Erro: {e2}")
            return None

    print(f"Total de {len(df)} amostras carregadas.")

    if INPUT_TEXT_COLUMN not in df.columns or INPUT_LABEL_COLUMN not in df.columns:
        print(f"ERRO: O arquivo CSV não contém as colunas necessárias: '{INPUT_TEXT_COLUMN}' e '{INPUT_LABEL_COLUMN}'")
        print(f"Colunas encontradas: {df.columns.tolist()}")
        return None

    initial_rows = len(df)
    
    df.dropna(subset=[INPUT_TEXT_COLUMN, INPUT_LABEL_COLUMN], inplace=True)
    
    clean_df = df[[INPUT_TEXT_COLUMN, INPUT_LABEL_COLUMN]].copy()

    clean_df.rename(columns={
        INPUT_TEXT_COLUMN: FINAL_TEXT_COLUMN,
        INPUT_LABEL_COLUMN: FINAL_LABEL_COLUMN
    }, inplace=True)
    
    print(f"{initial_rows - len(clean_df)} linhas com dados inválidos foram removidas.")

    print("Traduzindo labels (Subject) para o formato descritivo...")
    mapped_labels = clean_df[FINAL_LABEL_COLUMN].map(LABEL_MAP)

    unmapped_mask = mapped_labels.isna()
    if unmapped_mask.any():
        unmapped_labels = clean_df.loc[unmapped_mask, FINAL_LABEL_COLUMN].unique()
        print(f"ATENÇÃO: As seguintes labels não foram encontradas no mapa de tradução e serão mantidas com o nome original:")
        print(unmapped_labels)
        
    clean_df[FINAL_LABEL_COLUMN] = mapped_labels.fillna(clean_df[FINAL_LABEL_COLUMN])
    
    print(f"Total final: {len(clean_df)} amostras limpas e traduzidas.")
    return clean_df

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
    
    full_df = load_data_from_csv(INPUT_FILE_PATH)
    if full_df is None:
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