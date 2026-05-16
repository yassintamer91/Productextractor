import ollama
import csv
import json
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
DEFAULT_MODEL = "llama3.2-vision"

def get_image_files(folder):
    return [f for f in sorted(folder.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

def extract_product_info(model, image_path):
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
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
        )
        raw = response["message"]["content"].strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        return parsed.get("product_name") or "N/A", parsed.get("price") or "N/A"
    except json.JSONDecodeError:
        return response["message"]["content"].strip()[:200], "N/A"
    except Exception as e:
        return f"ERROR: {e}", "N/A"

def process_folder(folder, output_csv, model):
    image_files = get_image_files(folder)
    if not image_files:
        print(f"No images found in: {folder}")
        sys.exit(1)
    total = len(image_files)
    print(f"Model : {model}")
    print(f"Images: {total} found")
    print(f"Output: {output_csv}\n")
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subfolder", "Filename", "Product Name", "Price"])
        for i, image_path in enumerate(image_files, start=1):
            relative = image_path.relative_to(folder)
            subfolder = str(relative.parent) if str(relative.parent) != "." else "(root)"
            print(f"[{i}/{total}] {relative} ...", end=" ", flush=True)
            product_name, price = extract_product_info(model, image_path)
            writer.writerow([subfolder, image_path.name, product_name, price])
            csvfile.flush()
            print(f"-> {product_name} | {price}")
    print(f"\nDone! Results saved to: {output_csv}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True)
parser.add_argument("--output", default="products.csv")
parser.add_argument("--model", default=DEFAULT_MODEL)
args = parser.parse_args()
folder = Path(args.folder).expanduser().resolve()
if not folder.is_dir():
    print(f"Error: '{folder}' is not a valid directory.")
    sys.exit(1)
process_folder(folder, Path(args.output).resolve(), args.model)
