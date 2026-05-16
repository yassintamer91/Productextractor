#!/usr/bin/env python3
"""
Product Name & Price Extractor
Recursively scans a folder and its subfolders for images, uses Claude AI
to identify the product name and price in each image, and saves results to a CSV.

Requirements:
    pip install anthropic

Usage:
    python extract_product_names.py --folder /path/to/your/images --output products.csv

Get your API key at: https://console.anthropic.com/
"""

import anthropic
import base64
import csv
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

MEDIA_TYPE_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/png",
}


def get_image_files(folder: Path) -> list[Path]:
    """Recursively return all supported image files in folder and subfolders."""
    files = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for the image."""
    suffix = image_path.suffix.lower()
    media_type = MEDIA_TYPE_MAP.get(suffix, "image/jpeg")

    # BMP isn't natively supported — convert via Pillow if available
    if suffix == ".bmp":
        try:
            from PIL import Image
            import io
            img = Image.open(image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            return data, "image/png"
        except ImportError:
            pass  # Fall through to raw read

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_product_info(client: anthropic.Anthropic, image_path: Path) -> tuple[str, str]:
    """
    Ask Claude to identify the product name and price in the image.
    Returns (product_name, price) — price is 'N/A' if not visible.
    """
    try:
        image_data, media_type = encode_image(image_path)
    except Exception as e:
        return f"ERROR: could not read image — {e}", "N/A"

    prompt = (
        "Look at this product image and extract two things:\n"
        "1. The product name\n"
        "2. The price (exactly as shown, e.g. '$12.99'). If no price is visible, use null.\n\n"
        "Reply ONLY with a JSON object in this exact format, nothing else:\n"
        '{"product_name": "...", "price": "..." }'
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        raw = message.content[0].text.strip()
        # Strip markdown code fences if Claude adds them
        raw = raw.strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        product_name = parsed.get("product_name", "N/A") or "N/A"
        price = parsed.get("price", "N/A") or "N/A"
        return product_name, price

    except (json.JSONDecodeError, KeyError):
        # Fallback: return raw text in product name, N/A for price
        return message.content[0].text.strip(), "N/A"
    except anthropic.APIError as e:
        return f"ERROR: API error — {e}", "N/A"


def process_folder(folder: Path, output_csv: Path, api_key: str) -> None:
    client = anthropic.Anthropic(api_key=api_key)

    image_files = get_image_files(folder)
    if not image_files:
        print(f"No supported image files found in: {folder}")
        sys.exit(1)

    total = len(image_files)
    print(f"Found {total} image(s) across all subfolders. Starting extraction...\n")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subfolder", "Filename", "Product Name", "Price"])

        for i, image_path in enumerate(image_files, start=1):
            # Show relative path so you know which subfolder it came from
            relative = image_path.relative_to(folder)
            subfolder = str(relative.parent) if relative.parent != Path(".") else "(root)"

            print(f"[{i}/{total}] {relative} ...", end=" ", flush=True)
            product_name, price = extract_product_info(client, image_path)
            writer.writerow([subfolder, image_path.name, product_name, price])
            csvfile.flush()  # Save progress after every row
            print(f"→ {product_name} | {price}")

            # Polite rate-limit: ~2 requests/sec
            if i < total:
                time.sleep(0.5)

    print(f"\nDone! Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract product names and prices from images using Claude AI."
    )
    parser.add_argument("--folder", required=True, help="Path to the root folder containing images/subfolders")
    parser.add_argument("--output", default="products.csv", help="Output CSV file path (default: products.csv)")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    output_csv = Path(args.output).expanduser().resolve()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: No API key provided. Use --api-key or set the ANTHROPIC_API_KEY environment variable.")
        print("Get your key at: https://console.anthropic.com/")
        sys.exit(1)

    process_folder(folder, output_csv, api_key)


if __name__ == "__main__":
    main()
