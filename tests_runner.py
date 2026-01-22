"""
Carregamento e avaliação de casos de teste definidos em `tests.txt`.
Formata resultados em JSON com percentagem e estado.
"""
import re
from typing import Any, Dict, List, Tuple

from pipeline import TextPipeline, AnalysisResult


def _parse_bool(v: str) -> bool:
    v = v.strip().lower()
    return v in {"1", "true", "yes", "sim", "y", "t"}


def load_tests_txt(path: str) -> List[Dict[str, Any]]:
    """Lê `tests.txt` e devolve a lista de casos estruturados.

    Cada bloco separado por `---` deve ter linhas `INPUT:` e `EXPECTED:`.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("---") if b.strip()]
    tests = []

    for b in blocks:
        m_in = re.search(r"INPUT\s*:\s*(.+)", b, flags=re.IGNORECASE)
        m_ex = re.search(r"EXPECTED\s*:\s*(.+)", b, flags=re.IGNORECASE)
        if not m_in or not m_ex:
            raise ValueError(f"Bloco inválido (faltou INPUT ou EXPECTED):\n{b}\n")

        inp = m_in.group(1).strip()
        exp_raw = m_ex.group(1).strip()

        expected = {}
        parts = [p.strip() for p in exp_raw.split(";") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            expected[k.strip().lower()] = v.strip()

        tests.append({"input": inp, "expected": expected})

    return tests


def evaluate_case(result: AnalysisResult, expected: Dict[str, str]) -> Tuple[int, int, List[str]]:
    """Compara o resultado obtido com o esperado e devolve (ok, total, falhas)."""
    total = 0
    ok = 0
    fails = []

    def norm(s: str) -> str:
        return s.strip().lower()

    if "negation" in expected:
        total += 1
        want = _parse_bool(expected["negation"])
        got = bool(result.negation)
        if got == want:
            ok += 1
        else:
            fails.append(f"negation: esperado={want} obtido={got}")

    if "utterance" in expected or "utterance_type" in expected:
        total += 1
        want = norm(expected.get("utterance", expected.get("utterance_type", "")))
        got = norm(result.utterance_type or "")
        if got == want:
            ok += 1
        else:
            fails.append(f"utterance_type: esperado={want} obtido={got}")

    if "pvf" in expected or "personal_vs_factual" in expected:
        total += 1
        want = norm(expected.get("pvf", expected.get("personal_vs_factual", "")))
        got = norm(result.personal_vs_factual or "")
        if got == want:
            ok += 1
        else:
            fails.append(f"pvf: esperado={want} obtido={got}")

    if "sentiment" in expected:
        total += 1
        want = expected["sentiment"].strip().upper()
        got = (result.sentiment or {}).get("label", "")
        if got == want:
            ok += 1
        else:
            fails.append(f"sentiment: esperado={want} obtido={got}")

    if "emotion" in expected:
        total += 1
        want = norm(expected["emotion"])
        got = norm(result.primary_emotion_pt or "")
        if got == want:
            ok += 1
        else:
            fails.append(f"emotion: esperado={want} obtido={got}")

    return ok, total, fails


def run_tests(tests_path: str) -> Dict[str, Any]:
    """Corre todos os casos e devolve um relatório agregado em dicionário."""
    tp = TextPipeline()
    tests = load_tests_txt(tests_path)

    total_checks = 0
    total_ok = 0
    cases = []

    for i, t in enumerate(tests, start=1):
        res = tp.analyze(t["input"])
        ok, total, fails = evaluate_case(res, t["expected"])
        total_checks += total
        total_ok += ok

        cases.append({
            "id": i,
            "input": t["input"],
            "expected": t["expected"],
            "score": (ok / total * 100.0) if total else 0.0,
            "fails": fails,
            "got": {
                "corrected": res.corrected,
                "utterance_type": res.utterance_type,
                "negation": res.negation,
                "personal_vs_factual": res.personal_vs_factual,
                "sentiment": (res.sentiment or {}).get("label"),
                "primary_emotion_pt": res.primary_emotion_pt,
                "primary_emotion_fine_pt": res.primary_emotion_fine_pt,
                "summary_tags": res.summary_tags,
            }
        })

    percent = (total_ok / total_checks * 100.0) if total_checks else 0.0

    if percent >= 100.0 - 1e-9:
        status, color = "SUCCESS", "GREEN"
    elif percent >= 50.0:
        status, color = "PARTIAL", "YELLOW"
    else:
        status, color = "FAIL", "RED"

    return {
        "status": status,
        "color": color,
        "percent": round(percent, 2),
        "checks_ok": total_ok,
        "checks_total": total_checks,
        "cases": cases,
    }
