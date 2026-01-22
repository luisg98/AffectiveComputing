"""
Pipeline de análise de texto em PT-PT: normalização, correção, tipo de enunciado,
negação, pessoal vs factual, sentimento, emoção e subjetividade.
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import spacy
import language_tool_python
from pysentimiento import create_analyzer


EMOTION_PT = {
    "joy": "alegria",
    "sadness": "tristeza",
    "anger": "raiva",
    "fear": "medo",
    "disgust": "nojo",
    "surprise": "surpresa",
    "gratitude": "gratidão",
    "relief": "alívio",
    "love": "amor",
    "optimism": "otimismo",
    "neutral": "neutro",
    "admiration": "admiração",
    "curiosity": "curiosidade",
    "disappointment": "deceção",
}

EMOTION_COARSE = {
    "joy": "alegria",
    "gratitude": "alegria",
    "relief": "alegria",
    "love": "alegria",
    "optimism": "alegria",
    "admiration": "alegria",

    "sadness": "tristeza",
    "disappointment": "tristeza",
    "fear": "medo",
    "anger": "raiva",
    "disgust": "nojo",

    "neutral": "neutro",
    "curiosity": "neutro",
    "surprise": "surpresa",
}

SENTIMENT_PT = {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}


# Léxico mínimo (fallback) — genérico, curto, não ajustado a exemplos
NEG_LEX = {
    "difícil", "dificil", "impossível", "impossivel", "pior", "mau", "má", "problema", "perdi", "perder",
    "falhar", "falhei", "atraso", "atrasado", "chumbar", "chumbei", "não", "nunca", "ninguém", "nada",
}
POS_LEX = {"ótimo", "otimo", "boa", "bom", "excelente", "contente", "feliz", "felizmente", "parabéns", "sucesso"}
SURPRISE_LEX = {"surpresa", "surpreendido", "surpreendida", "surpreender"}


@dataclass
class AnalysisResult:
    input: str
    corrected: str
    language: str = "pt-PT"

    utterance_type: str = "afirmacao"
    negation: bool = False
    personal_vs_factual: str = "unknown"

    sentiment: Optional[Dict[str, Any]] = None
    emotion_text: Optional[List[Dict[str, Any]]] = None
    subjectivity: Optional[Dict[str, Any]] = None

    primary_emotion_pt: Optional[str] = None
    primary_emotion_fine_pt: Optional[str] = None
    summary_tags: Optional[List[str]] = None

    notes: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    timings_ms: Optional[Dict[str, float]] = None


class TextPipeline:
    """Pipeline de NLP para texto em PT-PT usando spaCy, LanguageTool e pysentimiento."""
    def __init__(self, spacy_model: str = "pt_core_news_md"):
        self.warnings: List[str] = []
        self.nlp = self._load_spacy_model(spacy_model)

        self.lt = None
        self.lt_error = None
        try:
            self.lt = language_tool_python.LanguageTool("pt-PT")
        except Exception as e:
            self.lt_error = str(e)
            self.warnings.append("LanguageTool indisponível; correção desativada.")
            self.warnings.append(f"LT error: {self.lt_error}")

        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="pt")
        self.emotion_analyzer = create_analyzer(task="emotion", lang="pt")

    def _load_spacy_model(self, name: str):
        """Carrega o modelo spaCy solicitado, com fallback para `pt_core_news_sm`."""
        try:
            return spacy.load(name)
        except Exception:
            fallback = "pt_core_news_sm"
            try:
                self.warnings.append(f"Modelo spaCy '{name}' não encontrado. A usar '{fallback}'.")
                return spacy.load(fallback)
            except Exception as e2:
                raise RuntimeError(
                    "Não foi possível carregar modelos spaCy PT. "
                    "Instala: python -m spacy download pt_core_news_md"
                ) from e2

    def normalize_text(self, text: str) -> str:
        """Normaliza espaços e pontuação, e unifica aspas para reduzir ruído."""
        t = text.strip()
        t = re.sub(r"\s+", " ", t)
        t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("´", "'")
        t = re.sub(r"\s+([?.!,;:])", r"\1", t)
        return t

    def correct(self, text: str) -> Tuple[str, List[str], List[str]]:
        """Correção leve com LanguageTool; devolve (texto_corrigido, notas, avisos)."""
        notes: List[str] = []
        warns: List[str] = []

        if not self.lt:
            warns.append("LT desativado (Java>=17/PATH). A devolver apenas normalização.")
            return self.normalize_text(text), notes, warns

        matches = self.lt.check(text)
        corrected = language_tool_python.utils.correct(text, matches)

        for m in matches[:6]:
            rule = getattr(m, "rule_id", getattr(m, "ruleId", "UNKNOWN"))
            err_len = getattr(m, "error_length", getattr(m, "errorLength", 0))
            notes.append(f"LT: {rule} @ {m.offset}-{m.offset + err_len}")

        return corrected, notes, warns

    def detect_utterance_type(self, text: str, doc) -> str:
        """Classifica o enunciado: afirmação, pergunta ou exclamação."""
        t = text.strip()
        if t.endswith("?"):
            return "pergunta"
        if t.endswith("!"):
            return "exclamacao"

        q_words = {"quando", "onde", "porquê", "porque", "como", "quem", "qual", "quais"}
        if any(tok.lower_ in q_words for tok in doc[:2]):
            return "pergunta"
        if "será" in [tok.lower_ for tok in doc[:3]] and "que" in [tok.lower_ for tok in doc[:4]]:
            return "pergunta"

        return "afirmacao"

    def detect_negation(self, doc) -> bool:
        """Deteta negação usando dependências e vocabulário comum."""
        neg_tokens = {"não", "nunca", "jamais", "nem", "ninguém", "nada", "nenhum", "nenhuma", "nenhuns", "nenhumas"}
        for tok in doc:
            if tok.dep_ == "neg":
                return True
            if tok.lower_ in neg_tokens:
                return True
        for i, tok in enumerate(doc[:-1]):
            if tok.lower_ == "sem" and doc[i + 1].pos_ in {"NOUN", "VERB", "ADJ"}:
                return True
        return False

    def detect_personal_vs_factual(self, doc, utterance_type: str) -> str:
        """Heurística simples para classificar como pessoal ou factual."""
        first_person_tokens = {"eu", "nós", "me", "mim", "connosco", "comigo", "nos", "minha", "meu", "minhas", "meus"}
        if any(t.lower_ in first_person_tokens for t in doc):
            return "pessoal"

        for t in doc:
            if t.pos_ in {"VERB", "AUX"} and "Person=1" in t.morph:
                return "pessoal"

        opinion_markers = {"achar", "pensar", "crer", "sentir", "parecer"}
        if any(t.lemma_.lower() in opinion_markers for t in doc):
            return "pessoal"

        # Perguntas sem 1ª pessoa: por defeito factual (pedido de informação)
        if utterance_type == "pergunta":
            return "factual"

        if any(t.pos_ == "PROPN" for t in doc):
            return "factual"

        has_finite_verb = any(t.pos_ == "VERB" and "VerbForm=Fin" in t.morph for t in doc)
        has_subject = any(t.dep_ in {"nsubj", "nsubj:pass"} and t.pos_ in {"NOUN", "PRON", "DET"} for t in doc)
        if has_finite_verb and has_subject:
            return "factual"

        if any(t.pos_ == "NOUN" for t in doc):
            return "factual"

        return "unknown"

    @lru_cache(maxsize=512)
    def _sentiment_cached(self, text: str) -> Tuple[str, float]:
        r = self.sentiment_analyzer.predict(text)
        return r.output, float(max(r.probas.values()))

    def sentiment(self, text: str) -> Dict[str, Any]:
        """Calcula sentimento (POS/NEG/NEU) e mapeia para PT."""
        label, score = self._sentiment_cached(text)
        return {"label": label, "label_pt": SENTIMENT_PT.get(label, label), "score": score}

    @lru_cache(maxsize=512)
    def _emotion_cached(self, text: str) -> Tuple[Tuple[Tuple[str, float], ...], str]:
        r = self.emotion_analyzer.predict(text)
        items = tuple(sorted(r.probas.items(), key=lambda x: x[1], reverse=True))
        top_label = items[0][0] if items else "neutral"
        return items, top_label

    def emotions(self, text: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], str]:
        """Devolve top-k emoções com rótulos em PT e a emoção principal."""
        items, top_label = self._emotion_cached(text)
        top = items[:top_k]
        emo_list = [{"label": k, "label_pt": EMOTION_PT.get(k, k), "score": float(v)} for k, v in top]
        return emo_list, top_label

    def subjectivity(self, doc) -> Dict[str, Any]:
        """Estimativa heurística de subjetividade (0..1) e etiqueta (objective/subjective)."""
        markers = {"achar", "pensar", "crer", "sentir", "parecer"}
        has_marker = any(t.lemma_.lower() in markers for t in doc)
        first_person = {"eu", "nós", "me", "mim", "nos", "connosco"}
        has_first_person = any(t.lower_ in first_person for t in doc)

        score = 0.15
        if has_first_person:
            score += 0.45
        if has_marker:
            score += 0.30
        if any(t.lower_ in {"felizmente", "infelizmente"} for t in doc):
            score += 0.10

        score = float(min(score, 0.95))
        label = "subjective" if score >= 0.5 else "objective"
        return {"label": label, "score": score}

    def _lexical_polarity_hint(self, doc) -> Optional[str]:
        """Sugestão lexical mínima de polaridade (POS/NEG) baseada em léxico simples."""
        toks = {t.lemma_.lower() for t in doc if t.is_alpha} | {t.lower_ for t in doc if t.is_alpha}
        if toks & POS_LEX and not (toks & NEG_LEX):
            return "POS"
        if toks & NEG_LEX and not (toks & POS_LEX):
            return "NEG"
        return None

    def analyze(self, text: str) -> AnalysisResult:
        """Corre o pipeline completo e devolve `AnalysisResult` com tempos e tags."""
        timings: Dict[str, float] = {}
        notes: List[str] = []

        t0 = time.perf_counter()
        raw = text
        normalized = self.normalize_text(raw)
        timings["normalize"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        corrected, lt_notes, lt_warns = self.correct(normalized)
        notes.extend(lt_notes)
        timings["correct"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        doc = self.nlp(corrected)
        timings["spacy_parse"] = (time.perf_counter() - t2) * 1000

        utt_type = self.detect_utterance_type(corrected, doc)
        neg = self.detect_negation(doc)
        pvf = self.detect_personal_vs_factual(doc, utt_type)

        t3 = time.perf_counter()
        sent = self.sentiment(corrected)
        timings["sentiment"] = (time.perf_counter() - t3) * 1000

        t4 = time.perf_counter()
        emo, top_em_label = self.emotions(corrected, top_k=3)
        timings["emotion"] = (time.perf_counter() - t4) * 1000

        subj = self.subjectivity(doc)

        if any(t.lower_ == "felizmente" for t in doc):
            notes.append("valencia_lexical: felizmente")
        if any(t.lower_ == "infelizmente" for t in doc):
            notes.append("valencia_lexical: infelizmente")

        primary_emotion_fine_pt = EMOTION_PT.get(top_em_label, top_em_label)
        primary_emotion_pt = EMOTION_COARSE.get(top_em_label, primary_emotion_fine_pt)

        # 1) Ajuste por emoção quando sentimento tem baixa confiança
        if sent and sent.get("score", 1.0) < 0.60:
            if primary_emotion_pt == "alegria" and sent.get("label") != "POS":
                sent["label"] = "POS"
                sent["label_pt"] = "positivo"
                notes.append("sentiment_adjusted_by_emotion: POS")
            elif primary_emotion_pt in {"tristeza", "raiva", "medo", "nojo"} and sent.get("label") != "NEG":
                sent["label"] = "NEG"
                sent["label_pt"] = "negativo"
                notes.append("sentiment_adjusted_by_emotion: NEG")

        # 2) Ajuste lexical mínimo (só se ainda estiver NEU e/ou confiança baixa)
        hint = self._lexical_polarity_hint(doc)
        if hint and sent:
            if sent.get("label") == "NEU" or sent.get("score", 1.0) < 0.55:
                sent["label"] = hint
                sent["label_pt"] = SENTIMENT_PT.get(hint, hint)
                notes.append(f"sentiment_adjusted_by_lexicon: {hint}")

        # 3) Fallback de emoção: se sentimento NEG e emoção neutra -> tristeza
        if sent and sent.get("label") == "NEG" and primary_emotion_pt == "neutro":
            primary_emotion_pt = "tristeza"
            notes.append("emotion_fallback_by_sentiment: tristeza")

        # 4) Surpresa: tende a neutro se sentimento não for forte
        if primary_emotion_pt == "surpresa" and sent and sent.get("score", 1.0) < 0.70:
            sent["label"] = "NEU"
            sent["label_pt"] = "neutro"
            notes.append("sentiment_adjusted_for_surprise: NEU")

        tags = [
            ("negação" if neg else "afirmação"),
            pvf,
            primary_emotion_pt,
        ]

        all_warns = (self.warnings or []) + (lt_warns or [])

        return AnalysisResult(
            input=raw,
            corrected=corrected,
            utterance_type=utt_type,
            negation=neg,
            personal_vs_factual=pvf,
            sentiment=sent,
            emotion_text=emo,
            subjectivity=subj,
            primary_emotion_pt=primary_emotion_pt,
            primary_emotion_fine_pt=primary_emotion_fine_pt,
            summary_tags=tags,
            notes=notes[:16] if notes else None,
            warnings=all_warns[:12] if all_warns else None,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
        )
