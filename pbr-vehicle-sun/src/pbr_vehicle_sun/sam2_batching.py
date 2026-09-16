"""Small dependency-light helpers for exhaustive SAM2.1 batching."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, ContextManager

import cv2
import numpy as np


def best_multimasks(masks: np.ndarray, scores: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Select SAM2's highest-score mask for every box, including singleton output."""
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        masks = masks[None, ...]
        scores = scores[None, ...]
    if masks.ndim != 4 or scores.ndim != 2 or masks.shape[:2] != scores.shape:
        raise ValueError(f"unexpected SAM2 mask/score shapes: {masks.shape}, {scores.shape}")
    indices = np.argmax(scores, axis=1)
    return [(masks[row, index].astype(bool), float(scores[row, index]))
            for row, index in enumerate(indices)]


def process_sam2_image_batch(
    predictor: Any,
    image_records: list[tuple[str, np.ndarray, list[dict]]],
    decoder_context: Callable[[], ContextManager] = nullcontext,
) -> list[tuple[dict, np.ndarray, float]]:
    """Batch image encoders and send all boxes of each image in one decoder call."""
    rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                  for _, image, _ in image_records]
    predictor.set_image_batch(rgb_images)
    box_batch = [np.asarray([record["bbox_xyxy"] for record in records], np.float32)
                 for _, _, records in image_records]
    # Match the frozen pipeline numerics: image encoding is full precision and
    # only the prompt/mask decoder may use CUDA bfloat16 autocast.
    with decoder_context():
        masks_batch, scores_batch, _ = predictor.predict_batch(
            box_batch=box_batch, multimask_output=True
        )
    output: list[tuple[dict, np.ndarray, float]] = []
    for (_, _, records), masks, scores in zip(image_records, masks_batch, scores_batch):
        choices = best_multimasks(masks, scores)
        if len(choices) != len(records):
            raise RuntimeError("SAM2 returned a different number of masks than box prompts")
        output.extend((record, mask, score)
                      for record, (mask, score) in zip(records, choices))
    return output
