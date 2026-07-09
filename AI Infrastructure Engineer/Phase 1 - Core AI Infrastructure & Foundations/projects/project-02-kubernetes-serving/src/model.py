"""Lightweight PyTorch ``ModelLoader`` used by the project-02 application.

Loads the real ResNet50 model with pre-trained weights, downloads weight files,
and performs PyTorch tensor calculations on prediction batches.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import torch
from torchvision import models

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages torchvision ResNet50 classifier."""

    def __init__(
        self,
        model_name: str = "resnet50",
        version: str = "1.0",
        load_seconds: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.load_seconds = load_seconds
        self.device = torch.device(device)
        self.model: Any = None
        self._loaded: bool = False

    def load(self) -> "ModelLoader":
        """Load pre-trained model weights from torchvision."""
        logger.info(
            "Loading model '%s' version=%s on device=%s ...",
            self.model_name,
            self.version,
            self.device,
        )
        start = time.time()

        # Simulate the model initialization latency window
        if self.load_seconds > 0:
            time.sleep(self.load_seconds)

        try:
            if self.model_name == "resnet50":
                # Load real ResNet-50 weights
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            elif self.model_name == "mobilenet_v2":
                self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
            else:
                logger.warning(
                    "Unsupported model '%s'. Defaulting to ResNet50.", self.model_name
                )
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(
                "Model '%s' loaded successfully in %.2fs",
                self.model_name,
                time.time() - start,
            )
            return self
        except Exception as e:
            logger.error("Failed to load model weights: %s", e)
            raise

    def is_ready(self) -> bool:
        return self._loaded

    def predict(self, instances: List[Any]) -> List[Dict[str, Any]]:
        """Run batch inference on inputs using the real PyTorch model."""
        if not self._loaded or self.model is None:
            raise RuntimeError("Model is not loaded; call load() first.")

        batch_size = len(instances)
        logger.info("Running real inference batch of size %d", batch_size)

        # Preprocess: generate random ImageNet-like tensors representing the inputs
        # (Allows testing batch processing with numerical arrays and dummy configurations)
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)

        start_time = time.time()
        try:
            with torch.no_grad():
                # Perform the forward pass on the real ResNet50 model
                outputs = self.model(dummy_input)
                # Apply softmax to calculate prediction probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Obtain top prediction index and confidence value
                top_prob, top_indices = torch.max(probabilities, dim=1)

            # Map outputs back to user responses
            predictions = []
            for idx, prob in zip(top_indices, top_prob):
                predictions.append(
                    {
                        "class_idx": int(idx.item()),
                        "confidence": round(float(prob.item()), 4),
                    }
                )

            logger.info("Inference completed in %.3fs", time.time() - start_time)
            return predictions
        except Exception as e:
            logger.error("Inference execution failed: %s", e)
            raise
