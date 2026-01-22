# Análise Afetiva Multimodal (PT-PT)

Este projeto implementa um **pipeline local e multimodal de análise afetiva em Português Europeu (PT-PT)**, combinando **texto e áudio** para extrair emoção, sentimento, subjetividade e traços linguísticos explicáveis.

O sistema foi desenhado para ser:
- 🔍 **Explicável**
- 🧠 **Modular**
- 🔐 **Local / privacy-friendly**
- ⚙️ **Reprodutível**
- 🎧 **Multimodal (texto + fala)**

---

## ✨ Funcionalidades

### Texto (NLP)
- Normalização e correção gramatical
- Deteção de tipo de enunciado (afirmação, pergunta, exclamação)
- Deteção de negação
- Classificação pessoal vs factual
- Análise de sentimento (POS/NEG/NEU)
- Deteção de emoção (fine e coarse)
- Estimativa de subjetividade
- Ajustes semânticos explicáveis
- Profiling de tempos

### Áudio
- Transcrição automática (Faster-Whisper)
- Emoção na fala com modelos SUPERB
- Suporte robusto a múltiplos formatos (ffmpeg)
- Análise via ficheiro, microfone ou batch (TSV)

### Multimodal
- Fusão de emoção de texto e áudio
- Decisão explicável da emoção provável
- Métricas de concordância

---

## 🧠 Tecnologias

- spaCy (PT)
- LanguageTool
- Transformers (Hugging Face)
- pysentimiento
- Faster-Whisper
- ffmpeg
- wav2vec2 / HuBERT (SUPERB)

---


