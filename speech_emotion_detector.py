"""
Deteção de emoções na fala usando modelos SUPERB (Hugging Face).
Carregamento de áudio via ffmpeg para maior compatibilidade.
"""
import os
import glob
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except Exception:
    class RepositoryNotFoundError(Exception):
        pass


def _get_hf_token() -> Optional[str]:
    """Obtém token Hugging Face de variáveis de ambiente (se existir)."""
    for key in ("HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN"):
        tok = os.getenv(key)
        if tok:
            return tok
    return None


def _prefetch_model(repo_id: str) -> None:
    """Pré-download do modelo do Hub para evitar falhas de rede em runtime."""
    if snapshot_download is None:
        return
    token = _get_hf_token()
    try:
        snapshot_download(repo_id=repo_id, local_files_only=False, token=token)
    except RepositoryNotFoundError as e:
        raise RepositoryNotFoundError(
            f"Repositório não encontrado ou privado: '{repo_id}'. Verifica o ID do modelo ou autenticação Hugging Face."
        ) from e
    except Exception:
        pass


def _find_ffmpeg_in_winget() -> Optional[str]:
    """
    Procura ffmpeg.exe instalado via WinGet (Gyan.FFmpeg costuma ficar aqui).
    """
    localapp = os.getenv("LOCALAPPDATA")
    if not localapp:
        return None

    root = Path(localapp) / "Microsoft" / "WinGet" / "Packages"
    if not root.exists():
        return None

    pattern = str(root / "**" / "bin" / "ffmpeg.exe")
    matches = glob.glob(pattern, recursive=True)
    matches = [m for m in matches if os.path.isfile(m)]
    if not matches:
        return None

    # escolhe o mais recente
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _resolve_ffmpeg_exe() -> str:
    """
    Resolve o executável ffmpeg por esta ordem:
      1) env FFMPEG_PATH (caminho completo para ffmpeg.exe)
      2) PATH (shutil.which)
      3) WinGet cache (LOCALAPPDATA\Microsoft\WinGet\Packages\...\bin\ffmpeg.exe)
      4) imageio-ffmpeg (se instalado)
    """
    # 1) FFMPEG_PATH
    envp = os.getenv("FFMPEG_PATH")
    if envp:
        envp = envp.strip('"').strip("'")
        if os.path.isfile(envp):
            return envp

    # 2) PATH
    which = shutil.which("ffmpeg")
    if which and os.path.isfile(which):
        return which

    # 3) winget
    w = _find_ffmpeg_in_winget()
    if w:
        return w

    # 4) imageio-ffmpeg
    try:
        import imageio_ffmpeg  # pip install imageio-ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    raise RuntimeError(
        "ffmpeg não encontrado. Soluções:\n"
        "  - Adiciona ffmpeg ao PATH, ou\n"
        "  - Define FFMPEG_PATH com o caminho para ffmpeg.exe, ou\n"
        "  - Instala 'imageio-ffmpeg' (pip install imageio-ffmpeg).\n"
        "Dica: como tens WinGet, o ffmpeg costuma estar em:\n"
        "  %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\...\\bin\\ffmpeg.exe\n"
    )


def _load_audio_ffmpeg(audio_path: str, target_sr: int = 16000) -> Dict:
    """
    Decodifica o áudio para mono float32 a 16kHz usando ffmpeg, e devolve:
      {"array": np.ndarray, "sampling_rate": int}
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Ficheiro de áudio não encontrado: {audio_path}")

    ffmpeg_exe = _resolve_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            "ffmpeg falhou a descodificar o áudio.\n"
            f"ffmpeg: {ffmpeg_exe}\n"
            f"ficheiro: {audio_path}\n"
            f"erro: {err}"
        )

    audio = np.frombuffer(p.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Áudio vazio após decode (ffmpeg). Verifica o ficheiro de áudio.")

    return {"array": audio, "sampling_rate": target_sr}


class SpeechEmotionDetector:
    """
    Detecta emoções em áudio usando modelos de classificação de áudio do Hugging Face.

    IMPORTANTE:
      - para evitar problemas com MP3/M4A no Windows, o áudio é sempre carregado com ffmpeg
        e passado ao pipeline como array+sampling_rate.
    """

    def __init__(self, model_name: str = "superb/wav2vec2-base-superb-er", top_k: int = 3, device: int = -1):
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self._pipe = None

    def _ensure_pipe(self):
        """Garante que o pipeline transformers foi inicializado."""
        if pipeline is None:
            raise RuntimeError("Pacote 'transformers' não está instalado. Instala com: pip install transformers")

        if self._pipe is not None:
            return

        try:
            self._pipe = pipeline("audio-classification", model=self.model_name, device=self.device)
        except OSError:
            # tenta pré-download e repete
            _prefetch_model(self.model_name)
            self._pipe = pipeline("audio-classification", model=self.model_name, device=self.device)

    def detect(self, audio_path: str) -> List[Dict[str, float]]:
        """Deteta emoções no áudio indicado e devolve lista de {label, score}."""
        self._ensure_pipe()
        audio = _load_audio_ffmpeg(audio_path, target_sr=16000)

        try:
            results = self._pipe(audio, top_k=self.top_k)
        except Exception as e:
            raise RuntimeError(f"[{self.model_name}] falha ao correr pipeline: {e}") from e

        if isinstance(results, dict):
            results = [results]

        return [{"label": r.get("label", "unknown"), "score": float(r.get("score", 0.0))} for r in results]
