#!/usr/bin/env python3
"""
Product Name & Price Extractor — Ollama Edition (FREE, runs locally)
Recursively scans a folder and its subfolders for images, uses a local
Ollama vision model to identify the product name and price, and saves
results to a CSV file.

Requirements:
    pip install ollama

Usage:
    python extract_product_names.py --folder /path/to/your/images --output products.csv

Optional — change the model (default: gemma4:e4b):
    python extract_product_names.py --folder ./images --model llama3.2-vision
"""

import ollama
import csv
import argparse
import json
import sys
import time
from pathlib import Path

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# Default model — good balance of speed and accuracy, runs on most machines.
# Alternatives:
#   "llama3.2-vision"  — great quality, needs ~8GB VRAM or 16GB RAM
#   "gemma4:e12b"      — higher quality, needs ~12GB VRAM
#   "gemma4:e4b"       — fastest, works on CPU, needs ~4GB RAM (default)
DEFAULT_MODEL = "gemma4:e4b"


def get_image_files(folder: Path) -> list[Path]:
    """Recursively return all supported image files in folder and subfolders."""
    files = [
        f for f in sorted(folder.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def extract_product_info(model: str, image_path: Path) -> tuple[str, str]:
    """
    Ask the local Ollama vision model to identify the product name and price.
    Returns (product_name, price) — price is 'N/A' if not visible.
    """
    prompt = (
        "Look at this product image and extract two things:\n"
        "1. The product name\n"
        "2. The price exactly as shown (e.g. '$12.99'). If no price is visible, use null.\n\n"
        "Reply ONLY with a JSON object in this exact format, no extra text:\n"
        '{"product_name": "...", "price": "..."}'
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],  # Ollama accepts file paths directly
                }
            ],
        )

        raw = response["message"]["content"].strip()
        # Strip markdown fences if the model adds them
        raw = raw.strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        product_name = parsed.get("product_name") or "N/A"
        price = parsed.get("price") or "N/A"
        return product_name, price

    except json.JSONDecodeError:
        # Model didn't return clean JSON — return raw text as product name
        raw = response["message"]["content"].strip()
        return raw[:200], "N/A"
    except ollama.ResponseError as e:
        return f"ERROR: {e}", "N/A"
    except Exception as e:
        return f"ERROR: {e}", "N/A"


def check_model_available(model: str) -> None:
    """Check if the model is pulled locally; exit with helpful message if not."""
    try:
        available = [m["name"] for m in ollama.list()["models"]]
        # ollama list returns names like "gemma4:e4b" — check loose match
        if not any(model in m for m in available):
            print(f"\nModel '{model}' is not downloaded yet.")
            print(f"Run this first:  ollama pull {model}\n")
            sys.exit(1)
    except Exception:
        print("\nCould not connect to Ollama. Is it running?")
        print("Start it with:  ollama serve\n")
        sys.exit(1)


def process_folder(folder: Path, output_csv: Path, model: str) -> None:
    check_model_available(model)

    image_files = get_image_files(folder)
    if not image_files:
        print(f"No supported image files found in: {folder}")
        sys.exit(1)

    total = len(image_files)
    print(f"Model : {model}")
    print(f"Images: {total} found across all subfolders")
    print(f"Output: {output_csv}\n")
    print("Starting extraction...\n")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subfolder", "Filename", "Product Name", "Price"])

        for i, image_path in enumerate(image_files, start=1):
            relative = image_path.relative_to(folder)
            subfolder = str(relative.parent) if str(relative.parent) != "." else "(root)"

            print(f"[{i}/{total}] {relative} ...", end=" ", flush=True)
            product_name, price = extract_product_info(model, image_path)
            writer.writerow([subfolder, image_path.name, product_name, price])
            csvfile.flush()  # Save after every row so progress isn't lost
            print(f"→ {product_name} | {price}")

    print(f"\nDone! Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract product names and prices from images using a local Ollama vision model."
    )
    parser.add_argument("--folder",  required=True, help="Root folder containing images / subfolders")
    parser.add_argument("--output",  default="products.csv", help="Output CSV file (default: products.csv)")
    parser.add_argument("--model",   default=DEFAULT_MODEL,  help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    output_csv = Path(args.output).expanduser().resolve()
    process_folder(folder, output_csv, args.model)


if __name__ == "__main__":
    main()
