"""
validate_pipeline.py
====================
Valida rigorosamente a integridade estrutural de todos os datasets gerados
pelo pipeline de processamento (cross-validation folds few-shot).

Checks:
  1. Schema — colunas obrigatórias ('text', 'label'), tipos, nulos
  2. Consistência entre splits — labels iguais em treino/val/teste
  3. Data leakage — amostras repetidas entre splits dentro de um fold
  4. Duplicatas — dentro de cada split
  5. Compatibilidade com treinamento — JSON válido, campos não vazios, encoding
  6. Tamanhos — proporção treino ≈ 60%, val ≈ 20%, teste ≈ 20%
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent

DATASETS = [
    # category
    {"task": "category", "name": "RulingBRCorpus"},
    {"task": "category", "name": "EniacCorpus"},
    {"task": "category", "name": "MMLU_PTBR_Corpus"},
    {"task": "category", "name": "RecognasummCorpus"},
    # hate
    {"task": "hate",     "name": "HateBRCorpus"},
    {"task": "hate",     "name": "TuPyCorpus"},
    # intent
    {"task": "intent",   "name": "CourtDecisionCorpus"},
    {"task": "intent",   "name": "IntentPTCorpus"},
    # reviews
    {"task": "reviews",  "name": "B2WReviewsCorpus"},
    {"task": "reviews",  "name": "BrandsCorpus"},
    {"task": "reviews",  "name": "BuscapeCorpus"},
    {"task": "reviews",  "name": "OlistCorpus"},
    {"task": "reviews",  "name": "KaggleTweetsCorpus"},
    {"task": "reviews",  "name": "ReProCorpus"},
    {"task": "reviews",  "name": "UTLCorpus"},
]

NUM_FOLDS   = 5
SPLITS      = ["train", "valid", "test"]
REQUIRED_COLS = {"text", "label"}

class Color:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def ok(msg):    print(f"  {Color.GREEN}[OK]{Color.RESET} {msg}")
def warn(msg):  print(f"  {Color.YELLOW}[AVISO]{Color.RESET} {msg}")
def err(msg):   print(f"  {Color.RED}[ERRO]{Color.RESET} {msg}")
def info(msg):  print(f"  {Color.CYAN}[INFO]{Color.RESET} {msg}")

def load_jsonl(path: Path):
    """Lê um JSONL e retorna lista de dicts + lista de erros de parsing."""
    records, errors = [], []
    if not path.exists():
        return None, [f"Arquivo não encontrado: {path}"]
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"  linha {i}: {e}")
    return records, errors

# ---------------------------------------------------------------------------
# Checks individuais
# ---------------------------------------------------------------------------

def check_schema(records, split_name, issues):
    """Verifica colunas obrigatórias, nulos, tipos."""
    if not records:
        issues.append(f"ERRO: split '{split_name}' está vazio.")
        return

    # colunas presentes
    sample_keys = set(records[0].keys())
    missing = REQUIRED_COLS - sample_keys
    if missing:
        issues.append(f"ERRO [{split_name}]: Colunas ausentes: {missing}")
    extra = sample_keys - REQUIRED_COLS
    if extra:
        issues.append(f"AVISO [{split_name}]: Colunas extras (não esperadas): {extra}")

    # nulos / vazios
    null_text  = sum(1 for r in records if not r.get("text") or str(r.get("text","")).strip() == "")
    null_label = sum(1 for r in records if not r.get("label") or str(r.get("label","")).strip() == "")
    if null_text:
        issues.append(f"ERRO [{split_name}]: {null_text} registros com 'text' nulo/vazio.")
    if null_label:
        issues.append(f"ERRO [{split_name}]: {null_label} registros com 'label' nulo/vazio.")

    # tipos — ambos devem ser string
    wrong_text_type  = sum(1 for r in records if not isinstance(r.get("text"), str))
    wrong_label_type = sum(1 for r in records if not isinstance(r.get("label"), str))
    if wrong_text_type:
        issues.append(f"ERRO [{split_name}]: {wrong_text_type} registros com 'text' não-string.")
    if wrong_label_type:
        issues.append(f"ERRO [{split_name}]: {wrong_label_type} registros com 'label' não-string.")


def check_label_consistency(splits_data, issues):
    """Verifica se todos os splits do fold compartilham o mesmo conjunto de labels."""
    label_sets = {}
    for split, records in splits_data.items():
        if records:
            label_sets[split] = set(r.get("label","") for r in records)

    all_labels = set().union(*label_sets.values()) if label_sets else set()
    reference  = label_sets.get("train", set())

    for split, labels in label_sets.items():
        if split == "train":
            continue
        diff = labels - reference
        missing_in_split = reference - labels
        if diff:
            issues.append(
                f"AVISO [label_consistency]: Labels em '{split}' ausentes no treino: {diff}. "
                f"Pode indicar mistura de dados ou split inconsistente."
            )
        if missing_in_split:
            issues.append(
                f"AVISO [label_consistency]: Labels do treino ausentes em '{split}': {missing_in_split}. "
                f"Splits podem ter distribuições diferentes de classes."
            )


def check_leakage(splits_data, issues):
    """Detecta textos idênticos entre splits diferentes (data leakage direto)."""
    sets_by_split = {
        split: set(r.get("text","") for r in records)
        for split, records in splits_data.items()
        if records
    }

    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]
    for a, b in pairs:
        if a in sets_by_split and b in sets_by_split:
            overlap = sets_by_split[a] & sets_by_split[b]
            if overlap:
                examples = list(overlap)[:3]
                issues.append(
                    f"ERRO [leakage]: {len(overlap)} texto(s) idêntico(s) em '{a}' e '{b}'. "
                    f"Exemplos: {[e[:60] for e in examples]}"
                )


def check_duplicates_within_split(records, split_name, issues):
    """Detecta amostras duplicadas dentro de um único split."""
    texts = [r.get("text","") for r in records]
    seen  = set()
    dups  = 0
    for t in texts:
        if t in seen:
            dups += 1
        seen.add(t)
    if dups:
        issues.append(
            f"ERRO [duplicatas]: {dups} texto(s) duplicado(s) dentro de '{split_name}'."
        )


def check_split_sizes(splits_data, issues):
    """Verifica proporção aproximada treino ≈ 60%, val ≈ 20%, teste ≈ 20%."""
    sizes = {s: len(r) for s, r in splits_data.items() if r}
    total = sum(sizes.values())
    if total == 0:
        return
    for split, n in sizes.items():
        pct = 100 * n / total
        # expected: train ~60%, valid ~20%, test ~20%
        expected = {"train": 60, "valid": 20, "test": 20}.get(split, 20)
        if abs(pct - expected) > 5:
            issues.append(
                f"AVISO [tamanho]: Split '{split}' representa {pct:.1f}% do total "
                f"(esperado ~{expected}%). Pode indicar desequilíbrio."
            )


def check_json_encoding(path: Path, issues):
    """Verifica se cada linha é JSON válido e decodificável em UTF-8."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    issues.append(f"ERRO [encoding/JSON]: Linha {i} inválida em {path.name}: {e}")
                    if i > 5:  # limita saída
                        break
    except UnicodeDecodeError as e:
        issues.append(f"ERRO [encoding]: Arquivo {path.name} com encoding inválido: {e}")

# ---------------------------------------------------------------------------
# Validação de um fold
# ---------------------------------------------------------------------------

def validate_fold(fold_path: Path, fold_name: str):
    fold_issues = []
    splits_data = {}

    for split in SPLITS:
        fpath = fold_path / f"{split}.jsonl"
        records, parse_errors = load_jsonl(fpath)

        if records is None:
            fold_issues.append(f"ERRO: Arquivo ausente — {fpath}")
            splits_data[split] = []
            continue

        if parse_errors:
            for pe in parse_errors:
                fold_issues.append(f"ERRO [parse]: {pe}")

        splits_data[split] = records

        # 1. Schema
        check_schema(records, split, fold_issues)

        # 2. Duplicatas internas
        check_duplicates_within_split(records, split, fold_issues)

        # 3. Encoding / JSON
        check_json_encoding(fpath, fold_issues)

    # 4. Consistência de labels
    check_label_consistency(splits_data, fold_issues)

    # 5. Data leakage
    check_leakage(splits_data, fold_issues)

    # 6. Proporção de tamanhos
    check_split_sizes(splits_data, fold_issues)

    return splits_data, fold_issues

# ---------------------------------------------------------------------------
# Validação de um corpus completo (todos os folds)
# ---------------------------------------------------------------------------

def validate_corpus(task: str, name: str):
    corpus_path = BASE_DIR / task / name / "few_shot"
    print(f"\n{Color.BOLD}{'='*60}{Color.RESET}")
    print(f"{Color.BOLD}[{task.upper()}] {name}{Color.RESET}")
    print(f"  Path: {corpus_path}")

    if not corpus_path.exists():
        err(f"Diretório do corpus não encontrado: {corpus_path}")
        return {"dataset": name, "task": task, "status": "AUSENTE", "folds": {}}

    fold_dirs = sorted([d for d in corpus_path.iterdir() if d.is_dir()])
    if not fold_dirs:
        err(f"Nenhum fold encontrado em: {corpus_path}")
        return {"dataset": name, "task": task, "status": "VAZIO", "folds": {}}

    if len(fold_dirs) != NUM_FOLDS:
        warn(f"Esperado {NUM_FOLDS} folds, encontrado {len(fold_dirs)}: {[d.name for d in fold_dirs]}")

    corpus_result = {"dataset": name, "task": task, "folds": {}, "global_issues": []}
    all_issues = []

    fold_test_sets = {}

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        print(f"\n  Fold {fold_name}:")
        splits_data, fold_issues = validate_fold(fold_dir, fold_name)
        corpus_result["folds"][fold_name] = {
            "issues": fold_issues,
            "sizes": {s: len(r) for s, r in splits_data.items()},
        }
        all_issues.extend(fold_issues)

        # Reporta
        if not fold_issues:
            ok(f"Fold {fold_name} — sem problemas.")
        else:
            for iss in fold_issues:
                if iss.startswith("ERRO"):
                    err(iss)
                elif iss.startswith("AVISO"):
                    warn(iss)
                else:
                    info(iss)

        # coleta teste para análise global
        test_path = fold_dir / "test.jsonl"
        recs, _ = load_jsonl(test_path)
        if recs:
            fold_test_sets[fold_name] = set(r.get("text","") for r in recs)

    fold_names = list(fold_test_sets.keys())
    test_overlap_found = False
    for i in range(len(fold_names)):
        for j in range(i + 1, len(fold_names)):
            a, b = fold_names[i], fold_names[j]
            overlap = fold_test_sets[a] & fold_test_sets[b]
            if overlap:
                msg = (f"ERRO [CV-integrity]: Conjuntos de TESTE dos folds '{a}' e '{b}' "
                       f"têm {len(overlap)} amostra(s) em comum — isso viola a validação cruzada!")
                corpus_result["global_issues"].append(msg)
                err(msg)
                test_overlap_found = True

    if not test_overlap_found and len(fold_test_sets) > 1:
        ok("Conjuntos de teste de todos os folds são disjuntos (CV correto).")

    # Status global
    all_errors  = [i for i in all_issues if "ERRO" in i]
    all_warnings = [i for i in all_issues if "AVISO" in i]
    if all_errors or corpus_result["global_issues"]:
        corpus_result["status"] = "FALHA"
    elif all_warnings:
        corpus_result["status"] = "AVISO"
    else:
        corpus_result["status"] = "OK"

    return corpus_result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{Color.BOLD}{'#'*60}")
    print("  VALIDAÇÃO DE INTEGRIDADE DO PIPELINE DE DATASETS")
    print(f"{'#'*60}{Color.RESET}")

    results = []
    for ds in DATASETS:
        result = validate_corpus(ds["task"], ds["name"])
        results.append(result)

    # -----------------------------------------------------------------------
    # Resumo final
    # -----------------------------------------------------------------------
    print(f"\n\n{Color.BOLD}{'='*60}")
    print("  RESUMO FINAL")
    print(f"{'='*60}{Color.RESET}")
    print(f"{'Dataset':<30} {'Task':<12} {'Status'}")
    print("-" * 60)

    total_ok   = 0
    total_warn = 0
    total_fail = 0
    total_miss = 0

    for r in results:
        status = r.get("status", "?")
        if status == "OK":
            color = Color.GREEN
            total_ok += 1
        elif status == "AVISO":
            color = Color.YELLOW
            total_warn += 1
        elif status == "FALHA":
            color = Color.RED
            total_fail += 1
        else:
            color = Color.CYAN
            total_miss += 1

        print(f"{r['dataset']:<30} {r['task']:<12} {color}{status}{Color.RESET}")

    print(f"\n  Total OK:      {total_ok}")
    print(f"  Total AVISO:   {total_warn}")
    print(f"  Total FALHA:   {total_fail}")
    print(f"  Total AUSENTE: {total_miss}")

    if total_fail > 0 or total_miss > 0:
        print(f"\n{Color.RED}{Color.BOLD}>>> PIPELINE COM ERROS CRÍTICOS <<<{Color.RESET}")
        sys.exit(1)
    elif total_warn > 0:
        print(f"\n{Color.YELLOW}{Color.BOLD}>>> PIPELINE COM AVISOS (revisar){Color.RESET}")
    else:
        print(f"\n{Color.GREEN}{Color.BOLD}>>> TODOS OS DATASETS ÍNTEGROS <<<{Color.RESET}")


if __name__ == "__main__":
    main()
