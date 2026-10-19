"""
AssetProcessor — what happens to ONE reading of ONE fleet asset, whatever
transport delivered it (in-process simulator or MQTT ingestion):

    buffer append → (warm?) feature engineering + model → AssetTracking
    (prediction row, RN-03 episode, root-cause snapshot on open).
"""

from __future__ import annotations

import asyncio
from functools import partial

import pandas as pd
import structlog

from src.core.exceptions import ModelNotAvailableError
from src.schemas.stream import SensorReading
from src.services.asset_tracking import AssetTracking
from src.services.feature_buffer import SensorBuffer
from src.services.inference_pipeline import (
    _infer_with_history,
    has_full_window,
    _reading_to_dict,
    _reading_to_request,
)
from src.services.model_registry import ModelRegistry
from src.services.model_service import ModelService
from src.services.preprocessing import MetroPTPreprocessor

log = structlog.get_logger(__name__)

EQUIPMENT_NAME = "Compressor de ar da APU"


def _window_score(
    model_service: ModelService, preprocessor: MetroPTPreprocessor, window: pd.DataFrame
) -> float:
    return _infer_with_history(model_service, window, preprocessor).failure_probability


class AssetProcessor:
    def __init__(
        self,
        asset_id: str,
        registry: ModelRegistry,
        tracking: AssetTracking,
        preprocessor: MetroPTPreprocessor | None = None,
    ) -> None:
        self.asset_id = asset_id
        self._registry = registry
        self._tracking = tracking
        self._preprocessor = preprocessor or MetroPTPreprocessor()
        self.buffer = SensorBuffer()

    async def process(self, reading: SensorReading) -> float | None:
        """Returns P(failure) once the buffer is warm, else None."""
        try:
            model_service = await self._registry.get()
        except ModelNotAvailableError:
            return None
        self.buffer.append(_reading_to_dict(reading))
        if not self.buffer.is_warm() or not has_full_window(
            model_service, len(self.buffer)
        ):
            return None
        window = self.buffer.to_dataframe()
        try:
            result = await asyncio.to_thread(
                _infer_with_history, model_service, window, self._preprocessor
            )
        except Exception as exc:
            log.warning("asset_inference_error", asset_id=self.asset_id, error=str(exc))
            return None
        await self._tracking.record(
            self.asset_id,
            EQUIPMENT_NAME,
            _reading_to_request(reading),
            result,
            self._registry.active_name,
            window,
            partial(_window_score, model_service, self._preprocessor),
            getattr(model_service, "decision_threshold", None),
        )
        return result.failure_probability
