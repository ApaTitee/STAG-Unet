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
	skip_blank: bool = True,
	max_wsis: int | None = None,
) -> list[PatchRecord]:
	patch_config = patch_config or PatchConfig()
	frame = load_dataframe(csv_path, data_root)
	if split is not None and "split" in frame.columns:
		frame = frame[frame["split"] == split].copy()
	if max_wsis is not None and max_wsis > 0:
		frame = frame.head(max_wsis).copy()

	records: list[PatchRecord] = []
	skipped_missing_files = 0
	for row in frame.itertuples(index=False):
		if not Path(row.wsi_abs_path).exists() or not Path(row.mask_abs_path).exists():
			skipped_missing_files += 1
			continue
		width, height = get_image_size(row.wsi_abs_path)
		for x, y in iter_patch_coordinates(width, height, patch_config.patch_size, patch_config.stride):
			fold = stable_patch_fold_key(row.patient_id, row.wsi_id, x, y, patch_config.patch_size, num_folds=num_folds)
			records.append(
				PatchRecord(
					patient_id=str(row.patient_id),
					wsi_id=str(row.wsi_id),
					wsi_path=str(row.wsi_abs_path),
					mask_path=str(row.mask_abs_path),
					x=int(x),
					y=int(y),
					fold=int(fold),
					# Weight is initialized uniformly; per-patch balancing can be added in offline indexing.
					weight=1.0,
				)
			)

	if skipped_missing_files > 0:
		print(f"[build_patch_records] skipped {skipped_missing_files} rows due to missing WSI/mask files.")
	return records


class WSIPatchDataset(Dataset):
	def __init__(
		self,
		records: Sequence[PatchRecord],
		patch_size: int = 512,
		num_classes: int = DEFAULT_NUM_CLASSES,
		transform: Callable | None = None,
		normalize: Callable | None = None,
	) -> None:
		self.records = list(records)
		self.patch_size = patch_size
		self.num_classes = num_classes
		self.transform = transform
		self.normalize = normalize

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, index: int):
		if not self.records:
			raise RuntimeError("WSIPatchDataset has no records.")

		last_error: Exception | None = None
		for attempt in range(8):
			candidate_index = index if attempt == 0 else random.randint(0, len(self.records) - 1)
			record = self.records[candidate_index]
			try:
				image = read_rgb_region(record.wsi_path, record.x, record.y, self.patch_size)
				mask = read_mask_region(record.mask_path, record.x, record.y, self.patch_size)

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

