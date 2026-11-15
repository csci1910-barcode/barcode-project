"""Download SKU-110K images from an Ultralytics Platform NDJSON export into YOLO layout.

The images are not in git (18 GB). The image URLs in the NDJSON are signed and expire
(the current export expires 2026-11-20); after that, download a fresh export from
https://platform.ultralytics.com/ultralytics/datasets/sku110k (Download dataset button).

Usage:
    python -m barcode.dataset                # everything (~18.5 GB)
    python -m barcode.dataset --limit 20     # quick test
Rerunning skips images that are already on disk.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import ssl
import time
import urllib.request

import certifi
from loguru import logger
from tqdm import tqdm
import typer

from barcode.config import RAW_DATA_DIR

app = typer.Typer()

SKU_DIR = RAW_DATA_DIR / "SKU-110K"
# python.org / uv Python builds on macOS don't use the system CA store, so default to certifi's
SSL_CONTEXT = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())


def fetch(record: dict, out_dir: Path, retries: int = 5) -> str:
    dst = out_dir / "images" / record["split"] / record["file"]
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(record["url"], timeout=60, context=SSL_CONTEXT) as resp:
                tmp.write_bytes(resp.read())
            tmp.rename(dst)
            return "ok"
        except OSError as e:
            error = e
            time.sleep(2**attempt)
    tmp.unlink(missing_ok=True)
    return f"fail: {error}"


@app.command()
def main(
    ndjson_path: Path = SKU_DIR / "sku110k.ndjson",
    out_dir: Path = SKU_DIR,
    limit: int = 0,
    workers: int = 24,
):
    records = [json.loads(line) for line in ndjson_path.open(encoding="utf-8") if line.strip()]
    meta = next(r for r in records if r["type"] == "dataset")
    images = [r for r in records if r["type"] == "image"][: limit or None]

    # YOLO labels + data.yaml (cheap, always rewritten)
    for r in images:
        label = out_dir / "labels" / r["split"] / Path(r["file"]).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        boxes = (r.get("annotations") or {}).get("boxes", [])
        label.write_text(
            "".join(f"{int(b[0])} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n" for b in boxes)
        )
    names = "\n".join(f"  {k}: {v}" for k, v in meta["class_names"].items())
    (out_dir / "data.yaml").write_text(
        "# SKU-110K (Ultralytics Platform export); no 'path' key: splits resolve relative to this file\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n"
    )

    logger.info(f"Downloading {len(images)} images to {out_dir / 'images'}...")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, r, out_dir): r["file"] for r in images}
        for future in tqdm(as_completed(futures), total=len(futures)):
            if (status := future.result()).startswith("fail"):
                failed.append(f"{futures[future]}\t{status}")

    if failed:
        (out_dir / "failed.txt").write_text("\n".join(failed) + "\n")
        logger.warning(
            f"{len(failed)} downloads failed, see {out_dir / 'failed.txt'}; rerun to retry"
        )
    else:
        logger.success("SKU-110K download complete.")


if __name__ == "__main__":
    app()
