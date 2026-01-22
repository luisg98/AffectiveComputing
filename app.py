import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

from pipeline import TextPipeline
from tests_runner import run_tests
from speech_emotion_detector import SpeechEmotionDetector

from helpers import (
    _ask_yes_no,
    _ask,
    _looks_like_path,
    _is_audio_file,
    _json_default,
    _transcribe_audio_file,
    transcription_info_to_dict,
    _record_mic_wav,
)

# -----------------------------
# Config
# -----------------------------
EMOTION_MODELS = [
    "superb/wav2vec2-base-superb-er",
    "superb/hubert-base-superb-er",
]

# Para estabilidade: CPU (-1). Se tiver GPU e drivers ok, podemos pôr 0.
AUDIO_EMOTION_DEVICE = -1

RESULTS_DIR = "results"


# -----------------------------
# Results helpers (tests only)
# -----------------------------
def _ensure_results_dir():
    """Garante que a pasta de resultados existe (results/)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _timestamp() -> str:
    """Devolve timestamp compacto para nomes de ficheiro."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_stem(path: str) -> str:
    """Normaliza o nome-base do caminho para uso seguro em ficheiros de saída."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    return stem[:80] if len(stem) > 80 else stem


def _save_json_result(payload: dict, prefix: str, source: Optional[str] = None) -> str:
    """Guarda um payload JSON em results/ com nome prefixado e timestamp.

    Args:
        payload: dicionário a serializar.
        prefix: prefixo do ficheiro (ex.: "tests", "audio_batch").
        source: identificador opcional da origem (ex.: caminho do TSV).
    Returns:
        Caminho absoluto do ficheiro JSON criado.
    """
    _ensure_results_dir()
    name_parts = [prefix, _timestamp()]
    if source:
        name_parts.append(_safe_stem(source))
    filename = "_".join(name_parts) + ".json"
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return out_path


# -----------------------------
# UI helpers
# -----------------------------
def _print_header():
    """Imprime o cabeçalho do UI interativo no terminal."""
    print("\n" + "=" * 72)
    print(" Análise Afetiva (PT-PT) — Pipeline Local")
    print("=" * 72)


def _print_result_human(r):
    """Apresenta o resultado de análise de texto de forma legível por humanos."""
    print("\n" + "-" * 72)
    print(f"Original:   {r.input}")
    print(f"Corrigida:  {r.corrected}")
    if r.summary_tags:
        print(f"Tags:       {', '.join(r.summary_tags)}")
    print(f"Tipo:       {r.utterance_type}")
    print(f"Negações:   {r.negation}")
    print(f"PVF:        {r.personal_vs_factual}")
    print(f"Sentimento: {r.sentiment}")
    print(f"Emoção:     {r.primary_emotion_pt} (fine: {r.primary_emotion_fine_pt})")
    print(f"Top emoções:{r.emotion_text}")
    print(f"Subj.:      {r.subjectivity}")

    if r.notes:
        print("\nNotas:")
        for n in r.notes:
            print(f"  - {n}")

    if r.warnings:
        print("\nAvisos:")
        for w in r.warnings:
            print(f"  - {w}")

    if r.timings_ms:
        print("\nTempos (ms):")
        for k, v in r.timings_ms.items():
            print(f"  - {k}: {v}")


# -----------------------------
# Audio aggregation + overview
# -----------------------------
def _aggregate_emotions(emotions_by_model: Dict[str, Any]):
    """Agrega emoções por etiqueta através de múltiplos modelos.

    Calcula a média das pontuações por etiqueta e devolve também a etiqueta
    com maior média.
    """
    totals = {}
    counts = {}
    for _, emotions in emotions_by_model.items():
        if not isinstance(emotions, list):
            continue
        for e in emotions:
            if not isinstance(e, dict):
                continue
            label = e.get("label")
            if not label:
                continue
            try:
                score = float(e.get("score", 0.0))
            except Exception:
                score = 0.0
            totals[label] = totals.get(label, 0.0) + score
            counts[label] = counts.get(label, 0) + 1

    if not totals:
        return {"averages": {}, "top_label": None, "top_score": None}

    averages = {lbl: (totals[lbl] / counts[lbl]) for lbl in totals}
    sorted_avgs = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = sorted_avgs[0] if sorted_avgs else (None, None)
    return {"averages": dict(sorted_avgs), "top_label": top_label, "top_score": top_score}


def _build_overview(audio_agg, analysis):
    """Combina achados de áudio e texto para sugerir emoção provável.

    Heurística:
    - Se áudio (top) e texto (primary) concordam, usar essa emoção.
    - Caso contrário, preferir áudio se a média for forte (>= 0.6).
    - Caso contrário, preferir a emoção principal do texto (fallback para áudio).
    """
    text_label = getattr(analysis, "primary_emotion_pt", None)
    fine_label = getattr(analysis, "primary_emotion_fine_pt", None)
    audio_label = audio_agg.get("top_label")
    audio_score = audio_agg.get("top_score") or 0.0

    agree = bool(audio_label and text_label and str(audio_label).lower() == str(text_label).lower())

    if agree and audio_label:
        probable = audio_label
        basis = "audio+text agree"
    else:
        if audio_label and audio_score >= 0.6:
            probable = audio_label
            basis = "audio strongest (avg score >= 0.6)"
        else:
            probable = text_label or audio_label
            basis = "text primary" if text_label else "audio fallback"

    return {
        "probable_emotion": probable,
        "agree": agree,
        "basis": basis,
        "audio_top": {"label": audio_label, "avg_score": audio_score},
        "text_primary": {"label": text_label, "fine": fine_label},
    }


# -----------------------------
# Audio pipeline (single file)
# -----------------------------
def audio_model_processor(
    tp: TextPipeline,
    audio_path: str,
    want_json: bool = False,
    top_k: int = 3,
    models=None,
):
    """Corre o pipeline de um único ficheiro de áudio e imprime/retorna resultados.

    Passos:
    1) Emoções por modelo (com robustez a exceções por modelo)
    2) Transcrição do áudio (Faster-Whisper via helpers)
    3) Análise de texto da transcrição
    4) Overview com emoção provável (combinação áudio+texto)

    Args:
        tp: instância de TextPipeline.
        audio_path: caminho do ficheiro de áudio a processar.
        want_json: se True, imprime JSON com payload completo.
        top_k: top-k de emoções por modelo.
        models: lista de modelos de emoção (default = EMOTION_MODELS).
    Returns:
        Payload (dict) com resultados. None se a transcrição estiver vazia.
    """
    if models is None:
        models = EMOTION_MODELS

    emotions_by_model = {}
    for model_name in models:
        try:
            detector = SpeechEmotionDetector(model_name=model_name, top_k=top_k, device=AUDIO_EMOTION_DEVICE)
            emotions_by_model[model_name] = detector.detect(audio_path)
        except Exception as e:
            print("\n" + "!" * 72)
            print(f"[ERRO] Emoções no áudio falharam no modelo: {model_name}")
            print(f"Exception: {repr(e)}")
            traceback.print_exc()
            print("!" * 72 + "\n")

            emotions_by_model[model_name] = [{
                "label": f"erro: {type(e).__name__}",
                "score": 0.0,
                "error_message": str(e),
            }]

    audio_agg = _aggregate_emotions(emotions_by_model)

    transcript, info = _transcribe_audio_file(audio_path)
    if not transcript:
        print("Transcrição vazia. Verifica o áudio e o ffmpeg.")
        return None

    analysis = tp.analyze(transcript)
    overview = _build_overview(audio_agg, analysis)

    payload = {
        "audio_path": audio_path,
        "transcript": transcript,
        "transcribe_info": transcription_info_to_dict(info),
        "analysis": asdict(analysis),
        "audio_emotions": emotions_by_model,
        "audio_emotions_aggregated": {
            "averages": audio_agg.get("averages"),
            "top_emotion": {
                "label": audio_agg.get("top_label"),
                "avg_score": audio_agg.get("top_score"),
            },
        },
        "overview": overview,
    }

    if want_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        return payload

    print("\nTranscrição:")
    print(transcript)
    _print_result_human(analysis)

    if emotions_by_model:
        print("\n==== Emoções no áudio (top-{} por modelo) ====".format(top_k))
        for model_name, emotions in emotions_by_model.items():
            print(f"\nModelo: {model_name}")
            for i, e in enumerate(emotions, 1):
                try:
                    print(f"  {i}. {e['label']} (score={float(e['score']):.3f})")
                except Exception:
                    print(f"  {i}. {e}")

    if audio_agg.get("averages"):
        print("\n==== Média de emoções do áudio (across models) ====")
        for label, avg in audio_agg["averages"].items():
            print(f"  - {label}: {float(avg):.3f}")

    print("\n==== Overview: emoção provável ====")
    agree_str = "sim" if overview.get("agree") else "não"
    print(f"  Emoção provável: {overview.get('probable_emotion')}")
    print(f"  Base: {overview.get('basis')} | Áudio e texto concordam: {agree_str}")
    at = overview.get("audio_top") or {}
    tt = overview.get("text_primary") or {}
    print(f"  Áudio (top): {at.get('label')} (média={at.get('avg_score')})")
    print(f"  Texto (emoção principal): {tt.get('label')} (fine={tt.get('fine')})")

    return payload


# -----------------------------
# Batch helpers (TSV)
# -----------------------------
def _read_tsv_audio_paths(tsv_path: str) -> List[str]:
    """Lê TSV e devolve lista de valores da coluna 'path'."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "path" not in reader.fieldnames:
            raise SystemExit(f"TSV sem coluna 'path': {tsv_path}")
        out = []
        for row in reader:
            p = (row.get("path") or "").strip()
            if p:
                out.append(p)
        return out


def _resolve_clip_path(dataset_dir: str, rel_path: str) -> Optional[str]:
    """Resolve <dataset_dir>/clips/<rel_path> (com fallback para basename)."""
    rel_path = rel_path.strip().strip('"').strip("'")
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path

    clips_dir = os.path.join(dataset_dir, "clips")
    p1 = os.path.join(clips_dir, rel_path)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(clips_dir, os.path.basename(rel_path))
    if os.path.exists(p2):
        return p2

    return None


def run_audio_batch_from_tsv(
    tp: TextPipeline,
    tsv_path: str,
    dataset_dir: str,
    top_k: int = 3,
    limit: Optional[int] = None,
) -> str:
    """Corre batch de áudio e guarda JSON em results/ (com barra de progresso)."""
    # import local para não obrigar tqdm no modo normal (real-time)
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if not os.path.exists(tsv_path):
        raise SystemExit(f"TSV não encontrado: {tsv_path}")

    clips_dir = os.path.join(dataset_dir, "clips")
    if not os.path.isdir(clips_dir):
        raise SystemExit(f"Pasta clips/ não encontrada em: {clips_dir}")

    rel_paths = _read_tsv_audio_paths(tsv_path)
    if limit is not None:
        rel_paths = rel_paths[: int(limit)]

    total = len(rel_paths)
    if total == 0:
        raise SystemExit("TSV sem paths para processar.")

    print(f"\n[Batch] TSV: {tsv_path}")
    print(f"[Batch] Clips: {clips_dir}")
    print(f"[Batch] Total: {total}")
    if tqdm is None:
        print("[Batch] Nota: instala 'tqdm' para barra de progresso (pip install tqdm)\n")
    else:
        print()

    # Cache dos detectors (1x por modelo)
    detectors: Dict[str, Any] = {}
    for model_name in EMOTION_MODELS:
        try:
            detectors[model_name] = SpeechEmotionDetector(
                model_name=model_name,
                top_k=top_k,
                device=AUDIO_EMOTION_DEVICE,
            )
        except Exception as e:
            detectors[model_name] = e

    started = time.time()
    ok = 0
    fail = 0
    results = []
    errors = []

    # wrapper para ter barra de progresso se existir tqdm
    iterator = tqdm(rel_paths, total=total, desc="A processar clips", unit="clip") if tqdm else rel_paths

    for i, rel in enumerate(iterator, 1):
        audio_path = _resolve_clip_path(dataset_dir, rel)
        t0 = time.time()

        if not audio_path:
            fail += 1
            err = {"index": i, "path": rel, "error": "FileNotFound", "message": "Ficheiro não encontrado em clips/"}
            errors.append(err)
            results.append({"ok": False, "path": rel, "error": err})
            if tqdm:
                iterator.set_postfix(ok=ok, fail=fail)
            continue

        try:
            emotions_by_model = {}
            for model_name in EMOTION_MODELS:
                det = detectors.get(model_name)
                if isinstance(det, Exception):
                    emotions_by_model[model_name] = [{
                        "label": f"error: {type(det).__name__}",
                        "score": 0.0,
                        "error_message": str(det),
                    }]
                    continue
                try:
                    emotions_by_model[model_name] = det.detect(audio_path)
                except Exception as e:
                    emotions_by_model[model_name] = [{
                        "label": f"erro: {type(e).__name__}",
                        "score": 0.0,
                        "error_message": str(e),
                    }]

            audio_agg = _aggregate_emotions(emotions_by_model)

            transcript, info = _transcribe_audio_file(audio_path)
            if not transcript:
                raise RuntimeError("Transcrição vazia.")

            analysis = tp.analyze(transcript)
            overview = _build_overview(audio_agg, analysis)

            payload = {
                "audio_path": audio_path,
                "transcript": transcript,
                "transcribe_info": transcription_info_to_dict(info),
                "analysis": asdict(analysis),
                "audio_emotions": emotions_by_model,
                "audio_emotions_aggregated": {
                    "averages": audio_agg.get("averages"),
                    "top_emotion": {
                        "label": audio_agg.get("top_label"),
                        "avg_score": audio_agg.get("top_score"),
                    },
                },
                "overview": overview,
            }

            ok += 1
            results.append({
                "ok": True,
                "path": rel,
                "audio_path": audio_path,
                "duration_sec": round(time.time() - t0, 4),
                "payload": payload,
            })

        except Exception as e:
            fail += 1
            err = {
                "index": i,
                "path": rel,
                "audio_path": audio_path,
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "duration_sec": round(time.time() - t0, 4),
            }
            errors.append(err)
            results.append({"ok": False, "path": rel, "audio_path": audio_path, "error": err})

        if tqdm:
            iterator.set_postfix(ok=ok, fail=fail)

    elapsed = time.time() - started

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "tsv_path": tsv_path,
            "dataset_dir": dataset_dir,
            "clips_dir": clips_dir,
            "total": total,
            "ok": ok,
            "fail": fail,
            "elapsed_sec": round(elapsed, 2),
            "avg_sec_per_clip": round(elapsed / total, 4) if total else None,
            "emotion_models": EMOTION_MODELS,
            "top_k": top_k,
        },
        "results": results,
        "errors": errors,
    }

    out_path = _save_json_result(payload=report, prefix="audio_batch", source=tsv_path)
    return out_path


def _auto_handle_input(tp: TextPipeline, raw: str):
    """Tenta inferir o tipo de input: caminho (tests.txt/áudio) ou texto.

    Returns:
        Tuplo (kind, out), onde kind ∈ {"tests", "audio", "text"}.
    Pode lançar SystemExit em erros de validação de ficheiro.
    """
    s = raw.strip().strip('"').strip("'")

    if _looks_like_path(s) and os.path.exists(s):
        if s.lower().endswith(".txt"):
            report = run_tests(s)
            return ("tests", report)

        if _is_audio_file(s):
            text, info = _transcribe_audio_file(s)
            if not text:
                raise SystemExit("Transcrição vazia. Verifica o ficheiro de áudio e a instalação do ffmpeg.")
            r = tp.analyze(text)

            emotions_by_model = {}
            for model_name in EMOTION_MODELS:
                try:
                    detector = SpeechEmotionDetector(model_name=model_name, top_k=3, device=AUDIO_EMOTION_DEVICE)
                    emotions_by_model[model_name] = detector.detect(s)
                except Exception as e:
                    print("\n" + "!" * 72)
                    print(f"[ERRO] Emoções no áudio falharam no modelo: {model_name}")
                    print(f"Exception: {repr(e)}")
                    traceback.print_exc()
                    print("!" * 72 + "\n")
                    emotions_by_model[model_name] = [{
                        "label": f"error: {type(e).__name__}",
                        "score": 0.0,
                        "error_message": str(e),
                    }]

            return (
                "audio",
                {
                    "audio_path": s,
                    "transcript": text,
                    "info": info,
                    "analysis": r,
                    "audio_emotions": emotions_by_model,
                },
            )

        raise SystemExit("O ficheiro existe, mas não é um áudio suportado nem um tests.txt.")

    r = tp.analyze(s)
    return ("text", r)


# -----------------------------
# Interactive UI
# -----------------------------
def interactive():
    """UI interativa com menu para texto e áudio (inclui sub-menus)."""
    tp = TextPipeline()

    while True:
        _print_header()
        print("Escolhe uma opção:")
        print("  1) Analisar texto")
        print("  2) Analisar áudio")
        print("  0) Sair")

        choice = _ask("\nOpção: ").strip()

        if choice == "0":
            print("Até já!")
            return

        # ---------------- TEXT ----------------
        if choice == "1":
            print("\nComo queres analisar o texto?")
            print("  1) Escrever texto manualmente (em tempo real)")
            print("  2) Inserir ficheiro (áudio ou tests.txt)")
            print("  3) Correr testes (tests.txt) — guarda JSON em results/")
            sub = _ask("\nOpção: ").strip()

            if sub == "1":
                text = _ask("\nEscreve o texto: ")
                if not text:
                    continue
                want_json = _ask_yes_no("Mostrar em JSON?", default=False)
                r = tp.analyze(text)
                if want_json:
                    print(json.dumps(asdict(r), ensure_ascii=False, indent=2, default=_json_default))
                else:
                    _print_result_human(r)
                input("\nEnter para continuar...")

            elif sub == "2":
                path = _ask("\nCaminho para ficheiro (áudio ou tests.txt): ").strip('"').strip("'")
                if not path:
                    continue
                if not os.path.exists(path):
                    print("Ficheiro não encontrado.")
                    input("\nEnter para continuar...")
                    continue

                if path.lower().endswith(".txt"):
                    report = run_tests(path)
                    out_path = _save_json_result(payload=report, prefix="tests", source=path)
                    print(f"\n[OK] Report de testes guardado em: {out_path}")
                    input("\nEnter para continuar...")
                    continue

                want_json = _ask_yes_no("Mostrar em JSON?", default=False)
                try:
                    kind, out = _auto_handle_input(tp, path)
                except SystemExit as e:
                    print(str(e))
                    input("\nEnter para continuar...")
                    continue

                if kind == "audio":
                    audio_model_processor(tp, path, want_json=want_json, top_k=3)
                elif kind == "text":
                    r = out
                    if want_json:
                        print(json.dumps(asdict(r), ensure_ascii=False, indent=2, default=_json_default))
                    else:
                        _print_result_human(r)
                else:
                    print("Tipo de input inesperado.")
                input("\nEnter para continuar...")

            elif sub == "3":
                path = _ask("\nCaminho para o tests.txt (Enter = text/tests.txt): ").strip()
                if not path:
                    path = "text/tests.txt"
                path = path.strip('"').strip("'")

                if not os.path.exists(path):
                    print("Ficheiro de testes não encontrado.")
                    input("\nEnter para continuar...")
                    continue

                report = run_tests(path)
                out_path = _save_json_result(payload=report, prefix="tests", source=path)
                print(f"\n[OK] Testes concluídos. JSON guardado em: {out_path}")
                input("\nEnter para continuar...")

            else:
                print("Opção inválida.")
                input("\nEnter para continuar...")

        # ---------------- AUDIO ----------------
        elif choice == "2":
            print("\nEscolhe como queres analisar o áudio:")
            print("  1) Real-time (ficheiro ou microfone) -> imprime")
            print("  2) Testes completos (TSV) -> guarda JSON em results/")
            suba = _ask("\nOpção: ").strip()

            if suba == "1":
                is_mic = _ask_yes_no("Usar microfone?", default=False)

                if not is_mic:
                    path = _ask("\nCaminho do ficheiro de áudio (wav/mp3/m4a...): ").strip('"').strip("'")
                    if not path:
                        continue
                    if not os.path.exists(path):
                        print("Ficheiro não encontrado.")
                        input("\nEnter para continuar...")
                        continue
                    if not _is_audio_file(path):
                        print("Formato de áudio não suportado.")
                        input("\nEnter para continuar...")
                        continue
                    want_json = _ask_yes_no("Mostrar em JSON?", default=False)
                else:
                    try:
                        dur_str = _ask("\nQuanto tempo queres gravar (segundos, Enter = 10): ").strip()
                        duration = int(dur_str) if dur_str else 10
                        if duration <= 0:
                            duration = 10
                    except Exception:
                        duration = 10

                    want_json = _ask_yes_no("Mostrar em JSON?", default=False)
                    path = _record_mic_wav(duration_sec=duration, sample_rate=16000, channels=1)
                    if not path:
                        input("\nEnter para continuar...")
                        continue

                audio_model_processor(tp, path, want_json=want_json, top_k=3)
                input("\nEnter para continuar...")

            elif suba == "2":
                default_tsv = os.path.join("audios", "pt", "global_ptpt.tsv")
                tsv_path = _ask(f"\nCaminho para o TSV (Enter = {default_tsv}): ").strip()
                if not tsv_path:
                    tsv_path = default_tsv
                tsv_path = tsv_path.strip('"').strip("'")

                if not os.path.exists(tsv_path):
                    print("TSV não encontrado.")
                    input("\nEnter para continuar...")
                    continue

                dataset_dir = os.path.dirname(tsv_path)

                lim_str = _ask("\nLimite de clips (Enter = todos): ").strip()
                limit = int(lim_str) if lim_str else None

                print("\nA correr batch… (isto pode demorar)")
                try:
                    out = run_audio_batch_from_tsv(tp, tsv_path=tsv_path, dataset_dir=dataset_dir, top_k=3, limit=limit)
                    print(f"\n[OK] Batch concluído. JSON guardado em: {out}")
                except SystemExit as e:
                    print(str(e))
                except Exception as e:
                    print("[ERRO] Batch falhou:", repr(e))
                    traceback.print_exc()

                input("\nEnter para continuar...")

            else:
                print("Opção inválida.")
                input("\nEnter para continuar...")

        else:
            print("Opção inválida.")
            input("\nEnter para continuar...")



def main():
    """Entrada principal: modo argumentos (CLI) ou modo menu (interativo)."""
    if len(sys.argv) > 1:
        # Modo ARGS (CLI)
        tp = TextPipeline()
        raw = " ".join(sys.argv[1:])
        kind, out = _auto_handle_input(tp, raw)

        if kind == "tests":
            out_path = _save_json_result(payload=out, prefix="tests", source=raw)
            print(json.dumps({"ok": True, "saved_to": out_path}, ensure_ascii=False, indent=2))
            return

        if kind == "audio":
            payload = {
                "audio_path": out.get("audio_path"),
                "transcript": out["transcript"],
                "transcribe_info": transcription_info_to_dict(out["info"]),
                "analysis": asdict(out["analysis"]),
                "audio_emotions": out.get("audio_emotions"),
                "emotion_models": EMOTION_MODELS if out.get("audio_emotions") else None,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return

        print(json.dumps(asdict(out), ensure_ascii=False, indent=2, default=_json_default))
        return
    # Modo menu (interativo)
    interactive()


if __name__ == "__main__":
    main()
