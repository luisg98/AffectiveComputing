"""
Transcrição de áudio (PT) para texto usando Faster-Whisper.
"""
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


@dataclass
class TranscriptionInfo:
    """Metadados e tempos da transcrição."""
    model: str
    device: str
    compute_type: str
    timings_ms: Dict[str, float]
    meta: Dict[str, float]


class Transcriber:
    """
    Transcreve áudio (PT) para texto usando Faster-Whisper.

    Requisitos:
      - Pacote: faster-whisper (instalado via requirements)
      - Externo: ffmpeg instalado e disponível no PATH
    """

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device or ("cuda" if os.getenv("USE_CUDA") == "1" else "cpu")
        self.compute_type = compute_type
        self._model = None
        self._model_load_ms = None

    def _ensure_model(self) -> None:
        """Carrega o modelo Faster-Whisper se ainda não estiver carregado."""
        if WhisperModel is None:
            raise RuntimeError(
                "Pacote 'faster-whisper' não está instalado. Adiciona-o ao requirements.txt e instala com pip."
            )
        if self._model is None:
            t0 = time.perf_counter()
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self._model_load_ms = (time.perf_counter() - t0) * 1000

    def transcribe_file(self, audio_path: str, language: str = "pt") -> Tuple[str, TranscriptionInfo]:
        """Transcreve um ficheiro de áudio para texto.

        Args:
            audio_path: caminho do ficheiro de áudio.
            language: código de idioma (ex.: "pt").
        Returns:
            Tuplo (texto_transcrito, TranscriptionInfo).
        Raises:
            FileNotFoundError: se o ficheiro não existir.
            RuntimeError: se ffmpeg/formato falharem ou o pipeline lançar exceção.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Ficheiro de áudio não encontrado: {audio_path}")
        self._ensure_model()

        t1 = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                beam_size=5,
            )
        except Exception as e:
            msg = (
                "Falha na transcrição. Verifica se o ffmpeg está instalado e o formato do ficheiro é suportado. "
                f"Erro: {e}"
            )
            raise RuntimeError(msg) from e
        trans_ms = (time.perf_counter() - t1) * 1000

        text = " ".join([s.text.strip() for s in segments]) if segments else ""
        timings = {"transcribe": trans_ms}
        if self._model_load_ms is not None:
            timings["model_load"] = self._model_load_ms

        meta = {
            "duration": getattr(info, "duration", 0.0) or 0.0,
            "avg_logprob": getattr(info, "avg_logprob", 0.0) or 0.0,
            "compression_ratio": getattr(info, "compression_ratio", 0.0) or 0.0,
            "no_speech_prob": getattr(info, "no_speech_prob", 0.0) or 0.0,
        }

        return text.strip(), TranscriptionInfo(
            model=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            meta=meta,
        )
