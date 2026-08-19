"""Cross-encoder re-ranking: the final retrieval stage, after hybrid fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import Settings, get_settings
from app.models.payload import ChunkPayload
from app.rag.hybrid_search import FusedResult


class Scorer(Protocol):
    """Anything that can score `(query, document)` pairs for relevance, higher = more relevant."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


@dataclass(frozen=True)
class RerankedResult:
    """One re-ranked hit: the cross-encoder score plus the upstream fused result for provenance."""

    id: str
    score: float
    payload: ChunkPayload
    fused: FusedResult


class CrossEncoderReranker:
    """Local sentence-transformers cross-encoder behind `Scorer`, the non-default fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
        model: object | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._model_name = model_name or settings.reranker_model
        self._model = model  # allow injecting a preloaded model (or a fake)

    def _get_model(self) -> object:
        if self._model is None:
            # Imported here so the heavy dependency is only required when actually loading.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        predictions = self._get_model().predict(list(pairs))  # type: ignore[attr-defined]
        return [float(p) for p in predictions]


class OnnxReranker:
    """Torch-free reranker (`onnxruntime` + `tokenizers` only), the production default."""

    def __init__(
        self,
        model_name: str | None = None,
        onnx_filename: str | None = None,
        *,
        settings: Settings | None = None,
        session: object | None = None,
        tokenizer: object | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._repo = model_name or settings.reranker_model
        self._onnx_filename = onnx_filename or settings.onnx_reranker_filename
        self._session = session  # allow injecting a fake session (or tokenizer) for tests
        self._tokenizer = tokenizer

    def _get_session_and_tokenizer(self) -> tuple[object, object]:
        if self._session is None or self._tokenizer is None:
            # Imported here so onnxruntime/tokenizers/huggingface_hub are only required
            # when this path is actually used.
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            onnx_path = hf_hub_download(self._repo, self._onnx_filename)
            tok_path = hf_hub_download(self._repo, "tokenizer.json")

            tokenizer = Tokenizer.from_file(tok_path)
            tokenizer.enable_truncation(max_length=512, strategy="longest_first")
            tokenizer.enable_padding(
                pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>"
            )  # length=None -> pad to the longest sequence in each batch

            self._session = ort.InferenceSession(onnx_path)
            self._tokenizer = tokenizer
        return self._session, self._tokenizer

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        session, tokenizer = self._get_session_and_tokenizer()

        encodings = tokenizer.encode_batch([list(p) for p in pairs])  # type: ignore[attr-defined]
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        input_names = {i.name for i in session.get_inputs()}  # type: ignore[attr-defined]
        feed: dict[str, np.ndarray] = {}
        for name in input_names:
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = attention_mask
            elif name == "token_type_ids":
                # Some exports still declare this input though it's unused: feed zeros, not omit.
                feed[name] = np.zeros_like(input_ids)
            else:
                raise ValueError(
                    f"OnnxReranker: unexpected ONNX graph input {name!r} — inspect the "
                    "graph (session.get_inputs()) before feeding it blindly."
                )

        output_name = session.get_outputs()[0].name  # type: ignore[attr-defined]
        (logits,) = session.run([output_name], feed)  # type: ignore[attr-defined]
        return [float(x) for x in np.asarray(logits).reshape(-1)]


def build_scorer(settings: Settings | None = None) -> Scorer:
    """Construct the `Scorer` chosen by `reranker_provider` ("onnx" default, "local" fallback)."""
    settings = settings or get_settings()
    if settings.reranker_provider == "local":
        return CrossEncoderReranker(settings=settings)
    return OnnxReranker(settings=settings)


def rerank(
    query: str,
    candidates: Sequence[FusedResult],
    *,
    scorer: Scorer,
    top_n: int | None = None,
) -> list[RerankedResult]:
    """Re-rank hybrid-search candidates by cross-encoder relevance to `query`, top `top_n` first."""
    if not candidates:
        return []
    pairs = [(query, c.payload.chunk_text) for c in candidates]
    scores = scorer.score(pairs)

    reranked = [
        RerankedResult(id=c.id, score=score, payload=c.payload, fused=c)
        for c, score in zip(candidates, scores, strict=True)
    ]
    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked[:top_n] if top_n is not None else reranked
