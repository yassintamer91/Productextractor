python3 -c "
import ollama, csv, json, sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
MODEL = 'llama3.2-vision'

def get_image_files(folder):
    return [f for f in sorted(folder.rglob('*')) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

def extract_product_info(model, image_path):
    prompt = 'Look at this product image and extract two things:\n1. The product name\n2. The price exactly as shown. If no price is visible, use null.\n\nReply ONLY with JSON: {\"product_name\": \"...\", \"price\": \"...\"}'
    try:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}], options={'num_predict': 100})
        raw = response['message']['content'].strip().strip('\`\`\`json').strip('\`\`\`').strip()
        parsed = json.loads(raw)
        return parsed.get('product_name') or 'N/A', parsed.get('price') or 'N/A'
    except:
        return 'N/A', 'N/A'

folder = Path('/workspace/images')
image_files = get_image_files(folder)
total = len(image_files)
print(f'Found {total} images')
with open('/workspace/products.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Subfolder', 'Filename', 'Product Name', 'Price'])
    for i, p in enumerate(image_files, 1):
        rel = p.relative_to(folder)
        sub = str(rel.parent) if str(rel.parent) != '.' else '(root)'
        print(f'[{i}/{total}] {p.name} ...', end=' ', flush=True)
        name, price = extract_product_info(MODEL, p)
        w.writerow([sub, p.name, name, price])
        f.flush()
        print(f'-> {name} | {price}')
print('Done!')
"
