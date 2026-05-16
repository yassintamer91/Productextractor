import ollama
import csv
import json
import sys
import argparse
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
DEFAULT_MODEL = "llama3.2-vision"

def get_image_files(folder):
    return [f for f in sorted(folder.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

def extract_product_info(model, image_path):
    prompt = "Look at this product image and extract two things: 1. The product name 2. The price exactly as shown. If no price is visible, use null. Reply ONLY with JSON: {\"product_name\": \"...\", \"price\": \"...\"}"
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            options={"num_predict": 100}
        )
        raw = response["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("product_name") or "N/A", parsed.get("price") or "N/A"
    except Exception:
        return "N/A", "N/A"

def process_folder(folder, output_csv, model):
    image_files = get_image_files(folder)
    if not image_files:
        print("No images found in: " + str(folder))
        sys.exit(1)
    total = len(image_files)
    print("Model : " + model)
    print("Images: " + str(total) + " found")
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subfolder", "Filename", "Product Name", "Price"])
        for i, image_path in enumerate(image_files, start=1):
            relative = image_path.relative_to(folder)
            subfolder = str(relative.parent) if str(relative.parent) != "." else "(root)"
            print("[" + str(i) + "/" + str(total) + "] " + image_path.name + " ...", end=" ", flush=True)
            product_name, price = extract_product_info(model, image_path)
            writer.writerow([subfolder, image_path.name, product_name, price])
            csvfile.flush()
            print("-> " + product_name + " | " + price)
    print("Done! Results saved to: " + str(output_csv))

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True)
parser.add_argument("--output", default="products.csv")
parser.add_argument("--model", default=DEFAULT_MODEL)
args = parser.parse_args()
folder = Path(args.folder).expanduser().resolve()
if not folder.is_dir():
    print("Error: not a valid directory: " + str(folder))
    sys.exit(1)
process_folder(folder, Path(args.output).resolve(), args.model)
