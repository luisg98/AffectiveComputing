"""
Funções auxiliares para prompts, deteção de caminhos/áudio,
serialização de JSON e transcrição/gravação de áudio.
"""
import os
import tempfile
from typing import Any, Dict, Optional
from dataclasses import asdict

from audio_transcriber import Transcriber

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Pergunta Sim/Não ao utilizador.

    Args:
        prompt: texto da pergunta.
        default: valor por defeito quando o utilizador só carrega Enter.
    Returns:
        True para sim, False para não.
    """
    suffix = " [S/n] " if default else " [s/N] "
    ans = input(prompt + suffix).strip().lower()
    if not ans:
        return default
    return ans in {"s", "sim", "y", "yes"}

def _ask(prompt: str) -> str:
    """Prompt simples que devolve a resposta do utilizador (stripped)."""
    return input(prompt).strip()

def _looks_like_path(s: str) -> bool:
    """Verifica se a string parece ser um caminho de ficheiro (áudio ou .txt)."""
    s = s.strip().strip('"').strip("'")
    ext = os.path.splitext(s)[1].lower()
    return (os.sep in s) or ("/" in s) or (ext in AUDIO_EXTS) or s.lower().endswith(".txt")

def _is_audio_file(path: str) -> bool:
    """Verifica se um caminho aponta para um ficheiro de áudio suportado."""
    ext = os.path.splitext(path)[1].lower()
    return ext in AUDIO_EXTS

def _json_default(o: Any):
    """Serializer JSON para dataclasses/objetos não nativamente serializáveis."""
    try:
        return asdict(o)
    except Exception:
        pass
    try:
        return vars(o)
    except Exception:
        return str(o)

def _transcribe_audio_file(path: str, language: str = "pt") -> str:
    """Transcreve um ficheiro de áudio e devolve (texto, info)."""
    tr = Transcriber()
    transcript, info = tr.transcribe_file(path, language)
    return transcript, info

def _record_mic_wav(duration_sec: int = 10, sample_rate: int = 16000, channels: int = 1) -> Optional[str]:
    """Grava do microfone para um WAV temporário.

    Args:
        duration_sec: duração em segundos (default 10).
        sample_rate: taxa de amostragem (default 16 kHz).
        channels: número de canais (default 1, mono).
    Returns:
        Caminho do ficheiro WAV gravado, ou None em caso de falha.

    Requer pacotes 'sounddevice' e 'soundfile'.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception:
        print("[Erro] 'sounddevice' ou 'soundfile' não estão instalados. Instala-os para gravar do microfone.")
        return None

    try:
        duration_sec = int(duration_sec) if duration_sec else 10
        if duration_sec <= 0:
            duration_sec = 10
    except Exception:
        duration_sec = 10

    try:
        # Create temp file path
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="mic_")
        os.close(fd)

        print(f"\n[Gravação] A gravar {duration_sec}s do microfone...")
        audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
        sf.write(tmp_path, audio, sample_rate)
        print(f"[Gravação] Ficheiro guardado: {tmp_path}")
        return tmp_path
    except Exception as e:
        print(f"[Erro] Falha ao gravar do microfone: {e}")
        return None

def transcription_info_to_dict(info: Any) -> Optional[Dict[str, Any]]:
    """Converte um objeto de info de transcrição para dicionário simples.

    Se não reconhecer campos, devolve uma representação mínima.
    """
    if info is None:
        return None
    if isinstance(info, dict):
        return info

    d: Dict[str, Any] = {}
    for key in ("language", "duration", "confidence", "vad", "beam_size"):
        if hasattr(info, key):
            try:
                d[key] = getattr(info, key)
            except Exception:
                pass

    try:
        segs = getattr(info, "segments", None)
        if segs is not None:
            d["segments"] = len(segs)
    except Exception:
        pass

    if not d:
        try:
            raw = vars(info)
            if isinstance(raw, dict):
                d = {"raw_keys": sorted(list(raw.keys()))[:30]}
        except Exception:
            d = {"type": type(info).__name__}

    return d