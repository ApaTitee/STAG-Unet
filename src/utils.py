from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import openslide

try:
	import tifffile
except Exception:  # pragma: no cover - optional dependency
	tifffile = None


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


@dataclass(frozen=True)
class AnnotationPolygon:
	label: int
	points: tuple[tuple[float, float], ...]
	bounding_box: tuple[float, float, float, float]


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


def read_mask_array(path: str | Path) -> np.ndarray:
	try:
		with Image.open(path) as image:
			mask = np.asarray(image)
	except Exception:
		if tifffile is None:
			raise
		mask = tifffile.imread(str(path))
	if mask.ndim == 3:
		mask = mask[..., 0]
	return mask



@lru_cache(maxsize=8)
def load_label_map(label_map_path: str | Path) -> dict[str, int]:
	with open(label_map_path, "r", encoding="utf-8") as file_handle:
		label_map = json.load(file_handle)
	return {str(key): int(value) for key, value in label_map.items()}


def _default_label_map_path(annotation_path: str | Path) -> Path:
	path = Path(annotation_path)
	return path.parents[1] / "label_map.json"


@lru_cache(maxsize=512)
def load_annotation_polygons(annotation_path: str | Path) -> tuple[AnnotationPolygon, ...]:
	path = Path(annotation_path)
	label_map = load_label_map(_default_label_map_path(path))
	polygons: list[AnnotationPolygon] = []

	if path.suffix.lower() == ".json":
		with path.open("r", encoding="utf-8") as file_handle:
			entries = json.load(file_handle)
		for entry in entries:
			coordinates = entry.get("coordinates") or []
			if len(coordinates) < 3:
				continue
			label_info = entry.get("label") or {}
			label_value = label_info.get("value")
			if label_value is None:
				label_name = str(label_info.get("name", "unannotated"))
				label_value = label_map.get(label_name, 0)
			points = tuple((float(x_coord), float(y_coord)) for x_coord, y_coord in coordinates)
			xs = [point[0] for point in points]
			ys = [point[1] for point in points]
			polygons.append(
				AnnotationPolygon(
					label=int(label_value),
					points=points,
					bounding_box=(min(xs), min(ys), max(xs), max(ys)),
				)
			)
		return tuple(polygons)

	if path.suffix.lower() == ".xml":
		root = ET.parse(path).getroot()
		for annotation in root.findall(".//Annotation"):
			group_name = str(annotation.attrib.get("PartOfGroup", "unannotated"))
			label_value = int(label_map.get(group_name, 0))
			coordinates: list[tuple[float, float]] = []
			for coordinate in annotation.findall(".//Coordinate"):
				x_coord = float(coordinate.attrib.get("X", 0.0))
				y_coord = float(coordinate.attrib.get("Y", 0.0))
				coordinates.append((x_coord, y_coord))
			if len(coordinates) < 3:
				continue
			points = tuple(coordinates)
			xs = [point[0] for point in points]
			ys = [point[1] for point in points]
			polygons.append(
				AnnotationPolygon(
					label=label_value,
					points=points,
					bounding_box=(min(xs), min(ys), max(xs), max(ys)),
				)
			)
		return tuple(polygons)

	raise ValueError(f"Unsupported annotation format: {path.suffix}")


def read_annotation_region(annotation_path: str | Path, x: int, y: int, size: int) -> np.ndarray:
	canvas = Image.new("L", (int(size), int(size)), 0)
	draw = ImageDraw.Draw(canvas)
	region_box = (int(x), int(y), int(x + size), int(y + size))
	for polygon in load_annotation_polygons(annotation_path):
		left, top, right, bottom = polygon.bounding_box
		if right < region_box[0] or bottom < region_box[1] or left > region_box[2] or top > region_box[3]:
			continue
		translated_points = [(point_x - x, point_y - y) for point_x, point_y in polygon.points]
		draw.polygon(translated_points, fill=int(polygon.label))
	return np.asarray(canvas)


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

