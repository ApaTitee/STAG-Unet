from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

import openslide


# BEETLE WSIs/masks are gigapixel TIFFs; allow opening trusted large images.
Image.MAX_IMAGE_PIXELS = None


DEFAULT_NUM_CLASSES = 5
DEFAULT_PATCH_SIZE = 512
DEFAULT_OVERLAP = 0.5
DEFAULT_STRIDE = int(DEFAULT_PATCH_SIZE * (1.0 - DEFAULT_OVERLAP))


@dataclass(frozen=True)
class PatchConfig:
	patch_size: int = DEFAULT_PATCH_SIZE
	overlap: float = DEFAULT_OVERLAP
	white_threshold: int = 245
	blank_ratio: float = 0.95
	num_folds: int = 5
	seed: int = 42

	@property
	def stride(self) -> int:
		return max(1, int(self.patch_size * (1.0 - self.overlap)))


def ensure_dir(path: str | Path) -> Path:
	output_path = Path(path)
	output_path.mkdir(parents=True, exist_ok=True)
	return output_path


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def resolve_path(root: str | Path, relative_path: str | Path) -> Path:
	relative = Path(relative_path)
	if relative.is_absolute():
		return relative
	return Path(root) / relative


def _safe_resolve_path(root: str | Path, value: object) -> str | None:
	if pd.isna(value):
		return None
	text = str(value).strip()
	if not text:
		return None
	return str(resolve_path(root, text))


def load_dataframe(csv_path: str | Path, data_root: str | Path) -> pd.DataFrame:
	frame = pd.read_csv(csv_path)
	root = Path(data_root)
	frame["wsi_abs_path"] = frame["wsi_path"].map(lambda value: _safe_resolve_path(root, value))
	frame["mask_abs_path"] = frame["annotation_mask_path"].map(lambda value: _safe_resolve_path(root, value))
	frame["json_abs_path"] = frame["annotation_json_path"].map(lambda value: _safe_resolve_path(root, value))
	frame["xml_abs_path"] = frame["annotation_xml_path"].map(lambda value: _safe_resolve_path(root, value))

	# Training/validation requires both WSI and mask paths; invalid rows are skipped.
	valid_required = frame["wsi_abs_path"].notna() & frame["mask_abs_path"].notna()
	frame = frame[valid_required].copy()
	return frame


def _open_slide(path: str | Path):
	if openslide is None:
		return None
	try:
		return openslide.OpenSlide(str(path))
	except Exception:
		return None


def get_image_size(path: str | Path) -> tuple[int, int]:
	slide = _open_slide(path)
	if slide is not None:
		return slide.dimensions
	with Image.open(path) as image:
		return image.size


def read_rgb_region(path: str | Path, x: int, y: int, size: int) -> np.ndarray:
	slide = _open_slide(path)
	if slide is not None:
		region = slide.read_region((int(x), int(y)), 0, (int(size), int(size))).convert("RGB")
		return np.asarray(region)
	with Image.open(path) as image:
		image = image.convert("RGB")
		crop = image.crop((int(x), int(y), int(x + size), int(y + size)))
		return np.asarray(crop)


def read_mask_region(path: str | Path, x: int, y: int, size: int) -> np.ndarray:
	with Image.open(path) as image:
		crop = image.crop((int(x), int(y), int(x + size), int(y + size)))
		mask = np.asarray(crop)
	if mask.ndim == 3:
		mask = mask[..., 0]
	return mask


def iter_patch_coordinates(width: int, height: int, patch_size: int, stride: int) -> Iterator[tuple[int, int]]:
	if width <= patch_size and height <= patch_size:
		yield 0, 0
		return

	x_positions = list(range(0, max(1, width - patch_size + 1), stride))
	y_positions = list(range(0, max(1, height - patch_size + 1), stride))

	if not x_positions or x_positions[-1] != max(0, width - patch_size):
		x_positions.append(max(0, width - patch_size))
	if not y_positions or y_positions[-1] != max(0, height - patch_size):
		y_positions.append(max(0, height - patch_size))

	for y in y_positions:
		for x in x_positions:
			yield x, y


def is_blank_patch(image: np.ndarray, mask: np.ndarray | None = None, white_threshold: int = 245, blank_ratio: float = 0.95) -> bool:
	if mask is not None and np.any(mask > 0):
		return False
	if image.ndim != 3 or image.shape[2] != 3:
		return False
	white_pixels = np.all(image >= white_threshold, axis=-1)
	return float(white_pixels.mean()) >= blank_ratio


def patch_weight_from_mask(mask: np.ndarray) -> float:
	foreground = float(np.count_nonzero(mask > 0))
	total = float(mask.size)
	if total == 0:
		return 1.0
	foreground_ratio = foreground / total
	return 0.2 if foreground_ratio == 0.0 else 1.0 + 4.0 * foreground_ratio


def stable_patch_fold_key(*parts: object, num_folds: int = 5) -> int:
	digest = hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()
	return int(digest, 16) % num_folds


def dice_score(prediction: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int | None = 0, eps: float = 1e-6) -> float:
	if prediction.ndim == 4:
		prediction = prediction.argmax(dim=1)

	scores: list[float] = []
	for class_index in range(num_classes):
		if ignore_index is not None and class_index == ignore_index:
			continue
		pred_mask = prediction == class_index
		target_mask = target == class_index
		intersection = float((pred_mask & target_mask).sum())
		denominator = float(pred_mask.sum() + target_mask.sum())
		score = (2.0 * intersection + eps) / (denominator + eps)
		scores.append(score)

	return float(sum(scores) / max(1, len(scores)))


def build_weighted_sampler_weights(mask_paths: Sequence[str], data_root: str | Path | None = None) -> list[float]:
	weights: list[float] = []
	for mask_path in mask_paths:
		absolute_path = resolve_path(data_root, mask_path) if data_root is not None else Path(mask_path)
		with Image.open(absolute_path) as image:
			mask = np.asarray(image)
		if mask.ndim == 3:
			mask = mask[..., 0]
		weights.append(patch_weight_from_mask(mask))
	return weights


class MacenkoNormalizer:
	def __init__(self, alpha: float = 1.0, beta: float = 0.15) -> None:
		self.alpha = alpha
		self.beta = beta
		self.reference_stain_matrix_: np.ndarray | None = None
		self.reference_max_concentration_: np.ndarray | None = None

	def fit(self, reference_image: np.ndarray) -> "MacenkoNormalizer":
		stain_matrix, concentration = self._estimate_stains(reference_image)
		self.reference_stain_matrix_ = stain_matrix
		self.reference_max_concentration_ = np.percentile(concentration, 99, axis=0)
		return self

	def transform(self, image: np.ndarray) -> np.ndarray:
		if self.reference_stain_matrix_ is None or self.reference_max_concentration_ is None:
			raise RuntimeError("MacenkoNormalizer must be fitted before calling transform().")

		source_stain_matrix, source_concentration = self._estimate_stains(image)
		source_max_concentration = np.percentile(source_concentration, 99, axis=0)
		source_max_concentration[source_max_concentration == 0] = 1.0

		normalized_concentration = source_concentration / source_max_concentration
		normalized_concentration = normalized_concentration * self.reference_max_concentration_
		reconstructed = np.exp(-(normalized_concentration @ self.reference_stain_matrix_)) * 255.0
		reconstructed = np.clip(reconstructed, 0.0, 255.0)
		return reconstructed.reshape(image.shape).astype(np.uint8)

	def _estimate_stains(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		od = self._rgb_to_od(image)
		od = od.reshape(-1, 3)
		od = od[np.sum(od > self.beta, axis=1) >= 2]
		if od.size == 0:
			od = self._rgb_to_od(image).reshape(-1, 3)

		covariance = np.cov(od, rowvar=False)
		_, _, v_t = np.linalg.svd(covariance)
		top2 = v_t[:2].T
		projected = od @ top2
		angles = np.arctan2(projected[:, 1], projected[:, 0])
		min_angle = np.percentile(angles, self.alpha)
		max_angle = np.percentile(angles, 100.0 - self.alpha)

		vector1 = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
		vector2 = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])
		stain_matrix = np.stack([vector1, vector2], axis=0)
		stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

		concentrations = np.linalg.lstsq(stain_matrix.T, self._rgb_to_od(image).reshape(-1, 3).T, rcond=None)[0].T
		concentrations = np.maximum(concentrations, 0.0)
		return stain_matrix, concentrations

	@staticmethod
	def _rgb_to_od(image: np.ndarray) -> np.ndarray:
		image = image.astype(np.float32)
		image = np.clip(image, 1.0, 255.0)
		return -np.log(image / 255.0)

