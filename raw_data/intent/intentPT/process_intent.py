"""
Fonte: AmazonScience/massive (HuggingFace/Amazon S3) — locale pt-PT
https://huggingface.co/datasets/AmazonScience/massive

O script baixa automaticamente o dataset compactado do Amazon S3,
extrai o arquivo 'pt-PT.jsonl', junta os dados, deduplica por texto,
embaralha e gera 5 folds de validação cruzada.
"""

import json
import shutil
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

INPUT_TEXT_COLUMN  = "utt"
INPUT_LABEL_COLUMN = "intent"

FINAL_TEXT_COLUMN  = "text"
FINAL_LABEL_COLUMN = "label"

OUTPUT_BASE_DIR = str(Path(__file__).resolve().parents[3] / "intent")
DATASET_NAME    = "IntentPTCorpus"
NUM_FOLDS       = 5
RANDOM_SEED     = 42

# Mapeamento dos 60 intents para PT-BR
LABEL_MAP = {
    "alarm_query":              "consultar alarme",
    "alarm_remove":             "remover alarme",
    "alarm_set":                "definir alarme",
    "audio_volume_down":        "diminuir volume",
    "audio_volume_mute":        "silenciar áudio",
    "audio_volume_other":       "controlar áudio",
    "audio_volume_up":          "aumentar volume",
    "calendar_query":           "consultar agenda",
    "calendar_remove":          "remover evento da agenda",
    "calendar_set":             "adicionar evento na agenda",
    "cooking_query":            "buscar sobre culinária",
    "cooking_recipe":           "buscar receita",
    "datetime_convert":         "converter data ou hora",
    "datetime_query":           "consultar data ou hora",
    "email_addcontact":         "adicionar contato",
    "email_query":              "consultar emails",
    "email_querycontact":       "consultar contato",
    "email_sendemail":          "enviar email",
    "general_greet":            "saudação",
    "general_joke":             "pedir piada",
    "general_quirky":           "conversa aleatória",
    "iot_cleaning":             "ativar limpeza",
    "iot_coffee":               "preparar café",
    "iot_hue_lightchange":      "mudar cor da luz",
    "iot_hue_lightdim":         "diminuir brilho da luz",
    "iot_hue_lightoff":         "apagar luz",
    "iot_hue_lighton":          "acender luz",
    "iot_hue_lightup":          "aumentar brilho da luz",
    "iot_wemo_off":             "desligar dispositivo",
    "iot_wemo_on":              "ligar dispositivo",
    "lists_createoradd":        "criar ou adicionar à lista",
    "lists_query":              "consultar lista",
    "lists_remove":             "remover da lista",
    "music_dislikeness":        "indicar 'não gostei' da música",
    "music_likeness":           "indicar 'gostei' da música",
    "music_query":              "consultar música",
    "music_settings":           "configurar música",
    "news_query":               "buscar notícias",
    "play_audiobook":           "tocar audiolivro",
    "play_game":                "jogar",
    "play_music":               "tocar música",
    "play_podcasts":            "tocar podcast",
    "play_radio":               "tocar rádio",
    "qa_currency":              "perguntar sobre câmbio",
    "qa_definition":            "pedir definição",
    "qa_factoid":               "perguntar fato ou curiosidade",
    "qa_maths":                 "perguntar matemática",
    "qa_stock":                 "perguntar sobre ações da bolsa",
    "recommendation_events":    "recomendar eventos",
    "recommendation_locations": "recomendar lugares",
    "recommendation_movies":    "recomendar filmes",
    "social_post":              "postar em rede social",
    "social_query":             "consultar rede social",
    "takeaway_order":           "pedir comida",
    "takeaway_query":           "consultar pedido de comida",
    "transport_query":          "consultar transporte",
    "transport_taxi":           "chamar táxi",
    "transport_ticket":         "comprar passagem",
    "transport_traffic":        "consultar trânsito",
    "weather_query":            "consultar previsão do tempo",
}

# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def download_and_extract_data() -> Path:
    """
    Verifica se o arquivo pt-PT.jsonl já existe localmente.
    Se não, faz o download do tarball v1.1 do MASSIVE do S3 e extrai o arquivo.
    Retorna o Path do arquivo jsonl.
    """
    script_dir = Path(__file__).resolve().parent
    local_jsonl = script_dir / "1.1" / "data" / "pt-PT.jsonl"
    
    if local_jsonl.exists():
        print(f"Arquivo local encontrado em: {local_jsonl}")
        return local_jsonl
        
    url = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
    archive_path = script_dir / "massive.tar.gz"
    
    print(f"Baixando base de dados MASSIVE do Amazon S3: {url}...")
    try:
        urllib.request.urlretrieve(url, archive_path)
        print("Download concluído com sucesso!")
        
        print("Extraindo pt-PT.jsonl...")
        with tarfile.open(archive_path, "r:gz") as tar:
            found = False
            for member in tar.getmembers():
                if "pt-PT.jsonl" in member.name:
                    print(f"Extraindo: {member.name}")
                    tar.extract(member, path=script_dir)
                    found = True
                    break
            if not found:
                raise FileNotFoundError("pt-PT.jsonl não foi encontrado no arquivo tar.gz")
                
    finally:
        if archive_path.exists():
            print("Limpando arquivo compactado temporário...")
            archive_path.unlink()
            
    return local_jsonl


def load_data_from_jsonl(file_path: Path) -> pd.DataFrame | None:
    """
    Carrega, extrai, limpa e traduz os dados do arquivo JSONL.
    """
    print(f"Carregando dados do arquivo JSONL: '{file_path}'...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Aviso: Ignorando linha mal formatada: {line}")
                    
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo JSONL: {e}")
        return None

    df = pd.DataFrame(data)
    print(f"Total de {len(df)} amostras carregadas.")

    if INPUT_TEXT_COLUMN not in df.columns or INPUT_LABEL_COLUMN not in df.columns:
        print(f"ERRO: O arquivo JSONL não contém as colunas necessárias: '{INPUT_TEXT_COLUMN}' e '{INPUT_LABEL_COLUMN}'")
        return None

    initial_rows = len(df)
    df.dropna(subset=[INPUT_TEXT_COLUMN, INPUT_LABEL_COLUMN], inplace=True)
    
    clean_df = df[[INPUT_TEXT_COLUMN, INPUT_LABEL_COLUMN]].copy()

    clean_df.rename(columns={
        INPUT_TEXT_COLUMN: FINAL_TEXT_COLUMN,
        INPUT_LABEL_COLUMN: FINAL_LABEL_COLUMN
    }, inplace=True)
    
    print(f"{initial_rows - len(clean_df)} linhas com dados inválidos foram removidas.")

    print("Traduzindo labels para o formato descritivo (PT-BR)...")
    mapped_labels = clean_df[FINAL_LABEL_COLUMN].map(LABEL_MAP)

    unmapped_mask = mapped_labels.isna()
    if unmapped_mask.any():
        unmapped_labels = clean_df.loc[unmapped_mask, FINAL_LABEL_COLUMN].unique()
        print(f"ATENÇÃO: As seguintes labels não foram encontradas no mapa de tradução e serão mantidas com o nome original:")
        print(unmapped_labels)
        
    clean_df[FINAL_LABEL_COLUMN] = mapped_labels.fillna(clean_df[FINAL_LABEL_COLUMN])

    print(f"Total final: {len(clean_df)} amostras limpas e traduzidas.")
    return clean_df


def save_json_pool(dataframe: pd.DataFrame, file_path: Path) -> None:
    """
    Salva o DataFrame como um pool de dados em formato JSONL (JSON Lines).
    """
    data_list = dataframe.to_dict('records')
    print(f"Salvando {len(data_list)} registros em JSONL em: {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')


def main() -> None:
    """
    Função principal que orquestra a criação dos folds de validação cruzada.
    """
    # Garante que os dados do MASSIVE estão disponíveis localmente
    try:
        jsonl_path = download_and_extract_data()
    except Exception as e:
        print(f"ERRO: Não foi possível obter os dados do MASSIVE: {e}")
        return
        
    full_df = load_data_from_jsonl(jsonl_path)
    if full_df is None:
        return

    print(f"\nTotal de amostras antes da deduplicação: {len(full_df)}")
    df_deduplicated = full_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    print(f"Total de amostras únicas (após deduplicação): {len(df_deduplicated)}")

    print(f"\nEmbaralhando o dataset único com a semente {RANDOM_SEED}...")
    df_shuffled = df_deduplicated.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Dividindo os dados em {NUM_FOLDS} folds ESTRATIFICADOS por classe...")
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(skf.split(df_shuffled, df_shuffled[FINAL_LABEL_COLUMN]))
    folds = [df_shuffled.iloc[test_idx].reset_index(drop=True) for _, test_idx in fold_indices]

    output_root = Path(OUTPUT_BASE_DIR) / DATASET_NAME / "few_shot"

    for i in range(NUM_FOLDS):
        fold_name = f"{i+1:02d}"
        print(f"\nProcessando Fold {fold_name}/{NUM_FOLDS}")
        
        output_path = output_root / fold_name
        if output_path.exists():
            print(f"Removendo diretório antigo: {output_path}")
            try:
                shutil.rmtree(output_path)
            except Exception as e:
                print(f"Aviso: Falha ao remover {output_path}: {e}")
        output_path.mkdir(parents=True, exist_ok=True)

        df_test = folds[i].reset_index(drop=True)
        df_valid = folds[(i + 1) % NUM_FOLDS].reset_index(drop=True)
        train_folds_indices = [j for j in range(NUM_FOLDS) if j != i and j != (i + 1) % NUM_FOLDS]
        df_train = pd.concat([folds[j] for j in train_folds_indices]).reset_index(drop=True)

        print(f"Tamanhos para o Fold {fold_name}: Treino={len(df_train)}, Validação={len(df_valid)}, Teste={len(df_test)}")

        print("Salvando pools de dados (.json)...")
        save_json_pool(df_train, output_path / "train.jsonl")
        save_json_pool(df_valid, output_path / "valid.jsonl")
        save_json_pool(df_test, output_path / "test.jsonl")

    print(f"\nPROCESSO CONCLUÍDO! {NUM_FOLDS} folds de validação cruzada foram criados em '{output_root}'")


if __name__ == "__main__":
    main()