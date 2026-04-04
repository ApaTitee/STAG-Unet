#!/usr/bin/env python
"""Generate a minimal test dataset with 3 synthetic WSI and matching masks."""

from pathlib import Path

import numpy as np
from PIL import Image

# Create test data directory structure
test_data_dir = Path("data_test")
test_wsi_dir = test_data_dir / "images" / "development" / "wsis"
test_mask_dir = test_data_dir / "annotations" / "masks"

test_wsi_dir.mkdir(parents=True, exist_ok=True)
test_mask_dir.mkdir(parents=True, exist_ok=True)

# Generate 3 synthetic test WSIs and masks
for idx in range(1, 4):
	# Create a realistic-looking histology patch: H&E like colors
	height, width = 2048, 2048
	wsi_image = np.ones((height, width, 3), dtype=np.uint8)
	
	# Add some purple/pink tones (typical H&E)
	wsi_image[:, :, 0] = np.random.randint(180, 220, (height, width))  # R
	wsi_image[:, :, 1] = np.random.randint(120, 160, (height, width))  # G
	wsi_image[:, :, 2] = np.random.randint(180, 220, (height, width))  # B
	
	# Add some tissue structures (darker areas = cells)
	for _ in range(5):
		y = np.random.randint(0, height - 256)
		x = np.random.randint(0, width - 256)
		wsi_image[y:y+256, x:x+256, :] = np.random.randint(50, 100, (256, 256, 3))
	
	# Save WSI as single-layer TIFF
	wsi_path = test_wsi_dir / f"test_wsi_{idx}.tif"
	Image.fromarray(wsi_image, mode="RGB").save(wsi_path, compression="lzw")
	print(f"Created: {wsi_path}")
	
	# Create corresponding mask: same size, 5 classes
	mask = np.zeros((height, width), dtype=np.uint8)
	mask[:, :] = 0  # Background
	
	# Add some tissue regions
	for class_label in [1, 2, 3, 4]:
		num_regions = np.random.randint(1, 3)
		for _ in range(num_regions):
			y = np.random.randint(0, max(1, height - 256))
			x = np.random.randint(0, max(1, width - 256))
			size = np.random.randint(64, 256)
			mask[y:y+size, x:x+size] = class_label
	
	# Save mask as grayscale TIFF
	mask_path = test_mask_dir / f"test_wsi_{idx}.tif"
	Image.fromarray(mask, mode="L").save(mask_path, compression="lzw")
	print(f"Created: {mask_path}")

# Create a minimal CSV for the test dataset
csv_content = """patient_id,wsi_id,name,source,specimen_type,scanner,wsi_path,annotation_mask_path,annotation_xml_path,annotation_json_path,split,validation_fold
test_patient_1,wsi1,test_wsi_1,test,biopsy,synthetic,images/development/wsis/test_wsi_1.tif,annotations/masks/test_wsi_1.tif,annotations/xmls/test_wsi_1.xml,annotations/jsons/test_wsi_1.json,development,0
test_patient_2,wsi1,test_wsi_2,test,biopsy,synthetic,images/development/wsis/test_wsi_2.tif,annotations/masks/test_wsi_2.tif,annotations/xmls/test_wsi_2.xml,annotations/jsons/test_wsi_2.json,development,1
test_patient_3,wsi1,test_wsi_3,test,biopsy,synthetic,images/development/wsis/test_wsi_3.tif,annotations/masks/test_wsi_3.tif,annotations/xmls/test_wsi_3.xml,annotations/jsons/test_wsi_3.json,development,2
"""

csv_path = test_data_dir / "data_overview.csv"
csv_path.write_text(csv_content)
print(f"Created: {csv_path}")

print("\n✓ Test dataset ready at:", test_data_dir.absolute())
print("  - 3 synthetic WSI images (2048x2048)")
print("  - 3 corresponding mask images")
print("  - data_overview.csv with 3 rows for 3-fold validation")
