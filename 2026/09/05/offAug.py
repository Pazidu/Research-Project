# ============================================================
# OFFLINE MELANOMA AUGMENTATION
# V27 DATASET PREPARATION
#
# Original dataset:
#   /content/newdata
#
# New dataset:
#   /content/newdata_offline_aug
#
# IMPORTANT:
#   - ONLY training melanoma images are augmented
#   - Validation is copied unchanged
#   - Test is copied unchanged
#   - Original dataset is never modified
# ============================================================

import os
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("/content/newdata")
OUTPUT_DIR = Path("/content/newdata_offline_aug")

# Create 3x total melanoma training images
#
# Example:
# 500 original melanoma images
# -> 1500 total melanoma images
#
# Therefore we generate:
# 1500 - 500 = 1000 augmented images
AUGMENT_MULTIPLIER = 3

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CLASS_NAMES = [
    "non_melanoma",
    "melanoma"
]

SPLITS = [
    "train",
    "valid",
    "test"
]


# ============================================================
# CHECK SOURCE DATASET
# ============================================================

print("=" * 70)
print("CHECKING SOURCE DATASET")
print("=" * 70)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE_DIR}"
    )

print(f"Source dataset: {SOURCE_DIR}")
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder):
    """
    Return all supported image files inside a folder.
    """
    if not folder.exists():
        return []

    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def count_images(split, class_name):
    """
    Count images in:
    /content/newdata/<split>/<class_name>
    """
    folder = SOURCE_DIR / split / class_name
    return len(get_images(folder))


def print_dataset_counts(base_dir, title):
    """
    Print dataset class counts.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    total_all = 0

    for split in SPLITS:

        print(f"\n{split.upper()}")

        split_total = 0

        for class_name in CLASS_NAMES:

            folder = base_dir / split / class_name
            count = len(get_images(folder))

            split_total += count

            print(
                f"  {class_name:<15}: {count:>6}"
            )

        print(
            f"  {'TOTAL':<15}: {split_total:>6}"
        )

        total_all += split_total

    print()
    print(f"TOTAL DATASET: {total_all}")


# ============================================================
# REALISTIC DERMOSCOPY AUGMENTATION
# ============================================================

def augment_melanoma_image(image):
    """
    Apply realistic, label-preserving transformations.

    These transformations are deliberately moderate.

    We DO NOT:
      - add artificial lesions
      - change lesion shape dramatically
      - use extreme color changes
      - crop away important lesion regions
      - distort the image heavily
    """

    img = image.copy()

    # --------------------------------------------------------
    # 1. Horizontal flip
    # --------------------------------------------------------

    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # --------------------------------------------------------
    # 2. Vertical flip
    # --------------------------------------------------------

    if random.random() < 0.25:
        img = ImageOps.flip(img)

    # --------------------------------------------------------
    # 3. Small rotation
    # --------------------------------------------------------

    if random.random() < 0.75:

        angle = random.uniform(
            -25,
            25
        )

        img = img.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=tuple(
                np.array(img).mean(
                    axis=(0, 1)
                ).astype(np.uint8)
            )
        )

    # --------------------------------------------------------
    # 4. Slight brightness adjustment
    # --------------------------------------------------------

    if random.random() < 0.50:

        factor = random.uniform(
            0.90,
            1.10
        )

        img = ImageEnhance.Brightness(
            img
        ).enhance(factor)

    # --------------------------------------------------------
    # 5. Slight contrast adjustment
    # --------------------------------------------------------

    if random.random() < 0.50:

        factor = random.uniform(
            0.90,
            1.10
        )

        img = ImageEnhance.Contrast(
            img
        ).enhance(factor)

    # --------------------------------------------------------
    # 6. Slight color/saturation adjustment
    # --------------------------------------------------------

    if random.random() < 0.35:

        factor = random.uniform(
            0.92,
            1.08
        )

        img = ImageEnhance.Color(
            img
        ).enhance(factor)

    # --------------------------------------------------------
    # 7. Very small blur
    # --------------------------------------------------------

    if random.random() < 0.10:

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(
                    0.2,
                    0.5
                )
            )
        )

    # --------------------------------------------------------
    # 8. Slight sharpness variation
    # --------------------------------------------------------

    if random.random() < 0.15:

        factor = random.uniform(
            0.9,
            1.1
        )

        img = ImageEnhance.Sharpness(
            img
        ).enhance(factor)

    return img


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print()
print("=" * 70)
print("CREATING OUTPUT DATASET")
print("=" * 70)

if OUTPUT_DIR.exists():

    print(
        f"Removing existing output dataset:\n"
        f"{OUTPUT_DIR}"
    )

    shutil.rmtree(
        OUTPUT_DIR
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPLAY ORIGINAL COUNTS
# ============================================================

print_dataset_counts(
    SOURCE_DIR,
    "ORIGINAL DATASET"
)


# ============================================================
# STEP 1
# COPY ENTIRE VALIDATION AND TEST SETS
# ============================================================

print()
print("=" * 70)
print("COPYING VALIDATION AND TEST SETS")
print("=" * 70)

for split in ["valid", "test"]:

    source_split = SOURCE_DIR / split
    output_split = OUTPUT_DIR / split

    if not source_split.exists():

        raise FileNotFoundError(
            f"Missing split: {source_split}"
        )

    shutil.copytree(
        source_split,
        output_split
    )

    print(
        f"✓ Copied {split} unchanged"
    )


# ============================================================
# STEP 2
# COPY TRAINING NON-MELANOMA
# ============================================================

print()
print("=" * 70)
print("COPYING TRAINING NON-MELANOMA")
print("=" * 70)

source_non_melanoma = (
    SOURCE_DIR
    / "train"
    / "non_melanoma"
)

output_non_melanoma = (
    OUTPUT_DIR
    / "train"
    / "non_melanoma"
)

output_non_melanoma.mkdir(
    parents=True,
    exist_ok=True
)

non_melanoma_images = get_images(
    source_non_melanoma
)

for image_path in non_melanoma_images:

    destination = (
        output_non_melanoma
        / image_path.name
    )

    shutil.copy2(
        image_path,
        destination
    )

print(
    f"✓ Copied {len(non_melanoma_images)} "
    f"non-melanoma training images unchanged"
)


# ============================================================
# STEP 3
# COPY ORIGINAL MELANOMA TRAINING IMAGES
# ============================================================

print()
print("=" * 70)
print("COPYING ORIGINAL MELANOMA TRAINING IMAGES")
print("=" * 70)

source_melanoma = (
    SOURCE_DIR
    / "train"
    / "melanoma"
)

output_melanoma = (
    OUTPUT_DIR
    / "train"
    / "melanoma"
)

output_melanoma.mkdir(
    parents=True,
    exist_ok=True
)

melanoma_images = get_images(
    source_melanoma
)

original_melanoma_count = len(
    melanoma_images
)

for image_path in melanoma_images:

    destination = (
        output_melanoma
        / image_path.name
    )

    shutil.copy2(
        image_path,
        destination
    )

print(
    f"✓ Copied {original_melanoma_count} "
    f"original melanoma images"
)


# ============================================================
# STEP 4
# CALCULATE NUMBER OF AUGMENTED IMAGES
# ============================================================

target_melanoma_count = (
    original_melanoma_count
    * AUGMENT_MULTIPLIER
)

images_to_generate = (
    target_melanoma_count
    - original_melanoma_count
)

print()
print("=" * 70)
print("AUGMENTATION PLAN")
print("=" * 70)

print(
    f"Original melanoma images : "
    f"{original_melanoma_count}"
)

print(
    f"Target melanoma images   : "
    f"{target_melanoma_count}"
)

print(
    f"New augmented images     : "
    f"{images_to_generate}"
)

print(
    f"Augmentation multiplier  : "
    f"{AUGMENT_MULTIPLIER}x"
)


# ============================================================
# STEP 5
# GENERATE OFFLINE MELANOMA IMAGES
# ============================================================

print()
print("=" * 70)
print("GENERATING AUGMENTED MELANOMA IMAGES")
print("=" * 70)

generated_count = 0

while generated_count < images_to_generate:

    # Randomly select an ORIGINAL melanoma image.
    #
    # IMPORTANT:
    # We always use the original image as the source.
    # We do NOT repeatedly augment an already augmented image.
    source_path = random.choice(
        melanoma_images
    )

    try:

        with Image.open(
            source_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            augmented = augment_melanoma_image(
                image
            )

            generated_count += 1

            output_filename = (
                f"{source_path.stem}"
                f"_aug_{generated_count:05d}.jpg"
            )

            output_path = (
                output_melanoma
                / output_filename
            )

            augmented.save(
                output_path,
                format="JPEG",
                quality=95
            )

    except Exception as e:

        print(
            f"⚠ Error processing "
            f"{source_path.name}: {e}"
        )

        generated_count -= 1

    # Progress display
    if (
        generated_count % 100 == 0
        or generated_count == images_to_generate
    ):

        print(
            f"Generated "
            f"{generated_count}/"
            f"{images_to_generate}"
        )


# ============================================================
# STEP 6
# FINAL COUNTS
# ============================================================

print_dataset_counts(
    OUTPUT_DIR,
    "NEW DATASET AFTER OFFLINE AUGMENTATION"
)


# ============================================================
# STEP 7
# VERIFY VALIDATION / TEST WERE NOT CHANGED
# ============================================================

print()
print("=" * 70)
print("VERIFYING VALIDATION / TEST")
print("=" * 70)

verification_passed = True

for split in ["valid", "test"]:

    print(f"\n{split.upper()}")

    for class_name in CLASS_NAMES:

        original_folder = (
            SOURCE_DIR
            / split
            / class_name
        )

        new_folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        original_images = get_images(
            original_folder
        )

        new_images = get_images(
            new_folder
        )

        original_names = sorted(
            p.name
            for p in original_images
        )

        new_names = sorted(
            p.name
            for p in new_images
        )

        if original_names != new_names:

            verification_passed = False

            print(
                f"❌ {split}/{class_name} "
                f"CHANGED"
            )

        else:

            print(
                f"✓ {split}/{class_name} "
                f"unchanged "
                f"({len(new_names)} images)"
            )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

if verification_passed:

    print(
        "✅ VERIFICATION PASSED"
    )

    print(
        "Validation and test sets were "
        "not modified."
    )

else:

    print(
        "❌ VERIFICATION FAILED"
    )

print("=" * 70)

print()
print(
    "NEW DATASET LOCATION:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "You can now use:"
)

print(
    str(OUTPUT_DIR)
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)