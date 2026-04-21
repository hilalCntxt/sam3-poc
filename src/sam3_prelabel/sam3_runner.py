from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import numpy as np


@dataclass
class Sam3Prediction:
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.array(value)
    return value


class Sam3ImagePredictor:
    def __init__(self) -> None:
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as exc:
            if exc.name == "triton":
                raise ImportError(
                    "SAM-3 requires Triton/CUDA runtime, but Triton is unavailable in this "
                    "environment. On macOS, run this pipeline on a Linux machine with an "
                    "NVIDIA GPU (CUDA 12.6+) instead."
                ) from exc
            if exc.name == "sam3":
                raise ImportError(
                    "SAM-3 import failed. Install SAM-3 first: "
                    "`pip install -e <path-to-sam3-repo>`"
                ) from exc
            raise ImportError(
                f"SAM-3 import failed due to missing module '{exc.name}'. "
                "Install SAM-3 dependencies from its repository README."
            ) from exc
        except Exception as exc:
            platform_hint = "Current platform: " + sys.platform
            raise ImportError(
                "SAM-3 initialization failed. Ensure SAM-3 is installed and compatible with "
                f"your runtime. {platform_hint}"
            ) from exc

        model = build_sam3_image_model()
        self._processor = Sam3Processor(model)

    def set_image(self, image: Any) -> Any:
        return self._processor.set_image(image)

    def predict_from_state(self, state: Any, prompt: str) -> Sam3Prediction:
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        return Sam3Prediction(
            masks=_to_numpy(output.get("masks")),
            boxes=_to_numpy(output.get("boxes")),
            scores=_to_numpy(output.get("scores")),
        )

    def predict(self, image: Any, prompt: str) -> Sam3Prediction:
        state = self.set_image(image)
        return self.predict_from_state(state, prompt=prompt)
