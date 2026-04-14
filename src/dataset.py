from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import random

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils import (
	DEFAULT_NUM_CLASSES,
	PatchConfig,
	build_weighted_sampler_weights,
	get_image_size,
	is_blank_patch,
	iter_patch_coordinates,
	load_dataframe,
	patch_weight_from_mask,
	read_annotation_region,
	read_mask_region,
	read_rgb_region,
	stable_patch_fold_key,
)


def build_train_transform() -> A.Compose:
	return A.Compose(
		[
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5),
			A.RandomRotate90(p=0.5),
			A.ShiftScaleRotate(
				shift_limit=0.05,
				scale_limit=0.15,
				rotate_limit=30,
				border_mode=0,
				p=0.7,
			),
			A.GaussNoise(p=0.3),
			A.CoarseDropout(
				num_holes_range=(1, 4),
				hole_height_range=(32, 96),
				hole_width_range=(32, 96),
				fill=0,
				fill_mask=0,
				p=0.3,
			),
		]
	)


def build_eval_transform() -> A.Compose:
	return A.Compose([])


@dataclass(frozen=True)
class PatchRecord:
	patient_id: str
	wsi_id: str
	wsi_path: str
	mask_path: str
	json_path: str | None
	xml_path: str | None
	x: int
	y: int
	fold: int
	weight: float


def build_patch_records(
	csv_path: str | Path,
	data_root: str | Path,
	patch_config: PatchConfig | None = None,
	split: str = "development",
	num_folds: int = 5,
	skip_blank: bool = False,
	max_wsis: int | None = None,
	patch_sampling_rate: float = 1.0,
	use_stratified_sampling: bool = False,
	max_patch_candidates_per_wsi: int = 256,
	random_seed: int | None = None,
) -> list[PatchRecord]:
	patch_config = patch_config or PatchConfig()
	if patch_sampling_rate <= 0 or patch_sampling_rate > 1:
		raise ValueError("patch_sampling_rate must be within (0, 1].")
	rng = random.Random(patch_config.seed if random_seed is None else random_seed)
	frame = load_dataframe(csv_path, data_root)
	if split is not None and "split" in frame.columns:
		frame = frame[frame["split"] == split].copy()
	if max_wsis is not None and max_wsis > 0:
		frame = frame.sample(n=min(max_wsis, len(frame)), random_state=patch_config.seed if random_seed is None else random_seed).copy()

	records: list[PatchRecord] = []
	skipped_missing_files = 0
	skipped_invalid_slides = 0
	skipped_invalid_masks = 0
	for row in frame.itertuples(index=False):
		if not Path(row.wsi_abs_path).exists() or not Path(row.mask_abs_path).exists():
			skipped_missing_files += 1
			continue
		try:
			width, height = get_image_size(row.wsi_abs_path)
		except Exception:
			skipped_invalid_slides += 1
			continue

		x_positions = list(range(0, max(1, width - patch_config.patch_size + 1), patch_config.stride))
		y_positions = list(range(0, max(1, height - patch_config.patch_size + 1), patch_config.stride))
		if not x_positions or not y_positions:
			continue
		if x_positions[-1] != max(0, width - patch_config.patch_size):
			x_positions.append(max(0, width - patch_config.patch_size))
		if y_positions[-1] != max(0, height - patch_config.patch_size):
			y_positions.append(max(0, height - patch_config.patch_size))

		total_coordinates = len(x_positions) * len(y_positions)
		target_count = max(1, int(round(total_coordinates * patch_sampling_rate)))
		candidate_count = min(total_coordinates, max(32, min(max_patch_candidates_per_wsi, target_count * 4)))
		coordinates = _sample_coordinate_candidates(x_positions, y_positions, candidate_count, rng)

		selected_coordinates = coordinates
		if patch_sampling_rate < 1.0 or skip_blank or use_stratified_sampling:
			try:
				mask_buckets: dict[int, list[tuple[int, int]]] = {}
				for x, y in coordinates:
					mask_patch = _read_mask_patch(
						mask_path=row.mask_abs_path,
						json_path=getattr(row, "json_abs_path", None),
						xml_path=getattr(row, "xml_abs_path", None),
						x=x,
						y=y,
						size=patch_config.patch_size,
					)
					label = _dominant_mask_label(mask_patch)
					if skip_blank and label == 0:
						continue
					mask_buckets.setdefault(label, []).append((int(x), int(y)))
			except Exception:
				skipped_invalid_masks += 1
				continue

			filtered_coordinates = [coordinate for bucket in mask_buckets.values() for coordinate in bucket]
			if not filtered_coordinates:
				continue

			if use_stratified_sampling:
				selected_coordinates = _sample_coordinates_by_label(mask_buckets, target_count, rng)
			else:
				selected_coordinates = rng.sample(filtered_coordinates, k=min(target_count, len(filtered_coordinates)))

		for x, y in selected_coordinates:
			fold = stable_patch_fold_key(row.patient_id, row.wsi_id, x, y, patch_config.patch_size, num_folds=num_folds)
			records.append(
				PatchRecord(
					patient_id=str(row.patient_id),
					wsi_id=str(row.wsi_id),
					wsi_path=str(row.wsi_abs_path),
					mask_path=str(row.mask_abs_path),
					json_path=str(row.json_abs_path) if getattr(row, "json_abs_path", None) is not None else None,
					xml_path=str(row.xml_abs_path) if getattr(row, "xml_abs_path", None) is not None else None,
					x=int(x),
					y=int(y),
					fold=int(fold),
					# Weight is initialized uniformly; per-patch balancing can be added in offline indexing.
					weight=1.0,
				)
			)

	if skipped_missing_files > 0:
		print(f"[build_patch_records] skipped {skipped_missing_files} rows due to missing WSI/mask files.")
	if skipped_invalid_slides > 0:
		print(f"[build_patch_records] skipped {skipped_invalid_slides} rows due to unreadable WSI files.")
	if skipped_invalid_masks > 0:
		print(f"[build_patch_records] skipped {skipped_invalid_masks} rows due to unreadable mask files.")
	return records


def build_patch_records_from_manifest(
	cache_root: str | Path,
	csv_path: str | Path,
	data_root: str | Path,
	split: str = "development",
	num_folds: int = 5,
) -> list[PatchRecord]:
	cache_root = Path(cache_root).expanduser().resolve()
	manifest_path = cache_root / "manifest.csv"
	if not manifest_path.exists():
		raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")

	frame = load_dataframe(csv_path, data_root)
	if split is not None and "split" in frame.columns:
		frame = frame[frame["split"] == split].copy()

	metadata_by_wsi: dict[str, object] = {}
	for row in frame.itertuples(index=False):
		wsi_id = str(row.wsi_id)
		if wsi_id not in metadata_by_wsi:
			metadata_by_wsi[wsi_id] = row

	manifest = pd.read_csv(manifest_path)
	required = {"fold", "wsi_id", "patch_x", "patch_y", "file_path"}
	missing_cols = required.difference(manifest.columns)
	if missing_cols:
		raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

	records: list[PatchRecord] = []
	skipped_missing_metadata = 0
	skipped_missing_cache = 0

	for row in manifest.itertuples(index=False):
		try:
			fold = int(row.fold)
			x = int(row.patch_x)
			y = int(row.patch_y)
		except Exception:
			continue

		if fold < 0 or fold >= num_folds:
			continue

		wsi_id = str(row.wsi_id)
		meta = metadata_by_wsi.get(wsi_id)
		if meta is None:
			skipped_missing_metadata += 1
			continue

		cache_file = cache_root / str(row.file_path)
		if not cache_file.exists():
			skipped_missing_cache += 1
			continue

		records.append(
			PatchRecord(
				patient_id=str(meta.patient_id),
				wsi_id=wsi_id,
				wsi_path=str(meta.wsi_abs_path),
				mask_path=str(meta.mask_abs_path),
				json_path=str(meta.json_abs_path) if getattr(meta, "json_abs_path", None) is not None else None,
				xml_path=str(meta.xml_abs_path) if getattr(meta, "xml_abs_path", None) is not None else None,
				x=x,
				y=y,
				fold=fold,
				weight=1.0,
			)
		)

	if skipped_missing_metadata > 0:
		print(f"[build_patch_records_from_manifest] skipped {skipped_missing_metadata} rows due to missing WSI metadata.")
	if skipped_missing_cache > 0:
		print(f"[build_patch_records_from_manifest] skipped {skipped_missing_cache} rows due to missing cache files.")

	print(f"[build_patch_records_from_manifest] loaded {len(records)} records from {manifest_path}")
	return records


def _sample_coordinate_candidates(
	x_positions: list[int],
	y_positions: list[int],
	count: int,
	rng: random.Random,
) -> list[tuple[int, int]]:
	if count <= 0:
		return []
	total = len(x_positions) * len(y_positions)
	if count >= total:
		return [(x, y) for y in y_positions for x in x_positions]

	selected: set[tuple[int, int]] = set()
	max_attempts = count * 10
	attempts = 0
	while len(selected) < count and attempts < max_attempts:
		selected.add((rng.choice(x_positions), rng.choice(y_positions)))
		attempts += 1

	if len(selected) < count:
		for y in y_positions:
			for x in x_positions:
				selected.add((x, y))
				if len(selected) >= count:
					break
			if len(selected) >= count:
				break

	coordinates = list(selected)
	rng.shuffle(coordinates)
	return coordinates[:count]


def _dominant_mask_label(mask: np.ndarray) -> int:
	if mask.size == 0:
		return 0
	foreground = mask[mask > 0]
	if foreground.size == 0:
		return 0
	labels, counts = np.unique(foreground, return_counts=True)
	return int(labels[int(np.argmax(counts))])


def _sample_coordinates_by_label(
	buckets: dict[int, list[tuple[int, int]]],
	target_count: int,
	rng: random.Random,
) -> list[tuple[int, int]]:
	if target_count <= 0:
		return []
	total = sum(len(coordinates) for coordinates in buckets.values())
	if total <= target_count:
		return [coordinate for bucket in buckets.values() for coordinate in bucket]

	labels = sorted(buckets)
	base_allocations: dict[int, int] = {}
	remainders: list[tuple[float, int]] = []
	allocated = 0
	for label in labels:
		fraction = len(buckets[label]) / total
		exact_count = fraction * target_count
		count = min(len(buckets[label]), int(exact_count))
		base_allocations[label] = count
		allocated += count
		remainders.append((exact_count - count, label))

	remaining = target_count - allocated
	for _, label in sorted(remainders, key=lambda item: (item[0], item[1]), reverse=True):
		if remaining <= 0:
			break
		if base_allocations[label] >= len(buckets[label]):
			continue
		base_allocations[label] += 1
		remaining -= 1

	selected: list[tuple[int, int]] = []
	for label in labels:
		count = base_allocations[label]
		if count <= 0:
			continue
		selected.extend(rng.sample(buckets[label], k=count))

	rng.shuffle(selected)
	return selected


class WSIPatchDataset(Dataset):
	def __init__(
		self,
		records: Sequence[PatchRecord],
		patch_size: int = 512,
		num_classes: int = DEFAULT_NUM_CLASSES,
		transform: Callable | None = None,
		normalize: Callable | None = None,
		cache_root: str | Path | None = None,
		cache_only: bool = False,
	) -> None:
		self.records = list(records)
		self.patch_size = patch_size
		self.num_classes = num_classes
		self.transform = transform
		self.normalize = normalize
		self.cache_root = Path(cache_root) if cache_root is not None else None
		self.cache_only = cache_only
		if self.cache_only and self.cache_root is None:
			raise ValueError("cache_only=True requires cache_root to be set.")

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, index: int):
		if not self.records:
			raise RuntimeError("WSIPatchDataset has no records.")

		last_error: Exception | None = None
		for attempt in range(8):
			candidate_index = index if attempt == 0 else random.randint(0, len(self.records) - 1)
			record = self.records[candidate_index]
			
			# Try to load from cache first
			if self.cache_root is not None:
				cache_file = self.cache_root / f"fold_{record.fold}" / record.wsi_id / f"patch_{record.x}_{record.y}.pt"
				if cache_file.exists():
					try:
						cached_data = torch.load(cache_file, weights_only=False)
						image = cached_data["image"].numpy()  # (512, 512, 3) float32
						mask = cached_data["mask"].numpy() if torch.is_tensor(cached_data["mask"]) else cached_data["mask"]
						image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
						
						if self.transform is not None:
							# Denormalize to apply augmentation
							image_uint8 = (image * 255.0).astype(np.uint8)
							image_uint8 = np.transpose(image_uint8, (1, 2, 0))
							
							# Ensure mask is uint8 numpy array
							if not isinstance(mask, np.ndarray):
								mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
							elif mask.dtype != np.uint8:
								mask = mask.astype(np.uint8)
							if mask.shape != (self.patch_size, self.patch_size):
								mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
							
							transformed = self.transform(image=image_uint8, mask=mask)
							image_uint8 = transformed["image"]
							mask = transformed["mask"]
							
							image = (image_uint8.astype(np.float32) / 255.0)
							image = np.transpose(image, (2, 0, 1))  # back to (3, 512, 512)
						else:
							# Ensure mask is uint8 numpy array
							if not isinstance(mask, np.ndarray):
								mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
							elif mask.dtype != np.uint8:
								mask = mask.astype(np.uint8)
							if mask.shape != (self.patch_size, self.patch_size):
								mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
						
						mask = mask.astype(np.int64)
						return {
							"image": torch.from_numpy(image),
							"mask": torch.from_numpy(mask),
						}
					except Exception as error:
						last_error = error
						continue
				elif self.cache_only:
					last_error = FileNotFoundError(f"Missing cache file: {cache_file}")
					continue

			if self.cache_only:
				last_error = RuntimeError("cache_only=True and cache loading did not return data.")
				continue
			
			# Fallback to raw data loading
			try:
				image = read_rgb_region(record.wsi_path, record.x, record.y, self.patch_size)
				mask = _read_mask_patch(
					mask_path=record.mask_path,
					json_path=record.json_path,
					xml_path=record.xml_path,
					x=record.x,
					y=record.y,
					size=self.patch_size,
				)

				if is_blank_patch(image, mask):
					mask = np.zeros_like(mask)

				if self.normalize is not None:
					image = self.normalize(image)

				if self.transform is not None:
					transformed = self.transform(image=image, mask=mask)
					image = transformed["image"]
					mask = transformed["mask"]

				image = image.astype(np.float32) / 255.0
				image = np.transpose(image, (2, 0, 1))
				mask = mask.astype(np.int64)

				return {
					"image": torch.from_numpy(image),
					"mask": torch.from_numpy(mask),
				}
			except Exception as error:
				last_error = error
				continue

		raise RuntimeError(f"Failed to load a valid patch after retries: {last_error}")


def _read_mask_patch(
	mask_path: str,
	json_path: str | None,
	xml_path: str | None,
	x: int,
	y: int,
	size: int,
) -> np.ndarray:
	try:
		return read_mask_region(mask_path, x, y, size)
	except Exception:
		annotation_path = json_path or xml_path
		if annotation_path is None:
			raise
		return read_annotation_region(annotation_path, x, y, size)


def split_records_by_fold(records: Sequence[PatchRecord], fold_index: int) -> tuple[list[PatchRecord], list[PatchRecord]]:
	train_records = [record for record in records if record.fold != fold_index]
	val_records = [record for record in records if record.fold == fold_index]
	return train_records, val_records


def build_sampler(records: Sequence[PatchRecord]) -> WeightedRandomSampler:
	weights = torch.tensor([record.weight for record in records], dtype=torch.double)
	return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_dataloader(
	dataset: Dataset,
	batch_size: int,
	shuffle: bool = False,
	sampler: WeightedRandomSampler | None = None,
	num_workers: int = 4,
) -> DataLoader:
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle if sampler is None else False,
		sampler=sampler,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=num_workers > 0,
	)

