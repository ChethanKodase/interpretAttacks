#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import scipy.io



'''

cd robustClassify
export CUDA_VISIBLE_DEVICES=6
conda activate robustness
python prepare_imagenet_val.py --val_dir imagenet/val --devkit_dir imagenet/ILSVRC2012_devkit_t12 


'''

def build_id_to_wnid(meta_mat_path: Path):
    """
    meta.mat contains a struct array "synsets".
    We map ILSVRC2012_ID (1..1000) -> WNID (e.g., n01440764).
    """
    meta = scipy.io.loadmat(str(meta_mat_path), squeeze_me=True, struct_as_record=False)
    synsets = meta["synsets"]

    id_to_wnid = {}
    # synsets includes more than 1000 entries; we only want ILSVRC2012_ID in [1..1000]
    for s in synsets:
        ilsvrc_id = getattr(s, "ILSVRC2012_ID", None)
        wnid = getattr(s, "WNID", None)
        if ilsvrc_id is None or wnid is None:
            continue
        # some entries have NaN / empty ILSVRC2012_ID
        try:
            ilsvrc_id_int = int(ilsvrc_id)
        except Exception:
            continue
        if 1 <= ilsvrc_id_int <= 1000:
            id_to_wnid[ilsvrc_id_int] = str(wnid)

    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"Expected 1000 classes, got {len(id_to_wnid)}. "
                           f"Check that meta.mat is from ILSVRC2012 devkit t12.")
    return id_to_wnid


def main():
    ap = argparse.ArgumentParser(description="Reorganize ILSVRC2012 val set into WNID folders.")
    ap.add_argument("--val_dir", required=True,
                    help="Path to folder containing ILSVRC2012_val_*.JPEG images")
    ap.add_argument("--devkit_dir", required=True,
                    help="Path to extracted ILSVRC2012_devkit_t12 directory (contains data/meta.mat)")
    ap.add_argument("--dry_run", action="store_true", help="Do not move files, just print actions")
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    devkit_dir = Path(args.devkit_dir)

    meta_mat = devkit_dir / "data" / "meta.mat"
    gt_file = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"

    if not meta_mat.exists():
        raise FileNotFoundError(f"Missing {meta_mat}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing {gt_file}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing {val_dir}")

    id_to_wnid = build_id_to_wnid(meta_mat)

    # Validation images are named like: ILSVRC2012_val_00000001.JPEG ... 50000
    # Ground truth file has 50000 lines of integer labels (1..1000), line i = image i.
    with gt_file.open("r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    if len(labels) != 50000:
        raise RuntimeError(f"Expected 50000 val labels, got {len(labels)}")

    # Move each image into its wnid folder
    for i, class_id in enumerate(labels, start=1):
        wnid = id_to_wnid[class_id]
        src = val_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
        dst_dir = val_dir / wnid
        dst = dst_dir / src.name

        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")

        if args.dry_run:
            print(f"Would move {src} -> {dst}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    print("Done. Validation set reorganized into 1000 class folders.")


if __name__ == "__main__":
    main()