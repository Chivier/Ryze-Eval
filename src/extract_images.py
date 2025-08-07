import os
from datasets import load_dataset
from PIL import Image
import json

def extract_sample_images():
    """Extract sample images from FigQA and TableQA subsets"""
    
    output_dir = "./sample_images"
    os.makedirs(output_dir, exist_ok=True)
    
    image_info = {}
    
    # Load FigQA subset
    print("Loading FigQA subset...")
    figqa = load_dataset("futurehouse/lab-bench", "FigQA")['train']
    
    print(f"FigQA has {len(figqa)} samples")
    fig_count = 0
    
    for idx, item in enumerate(figqa):
        if 'figure' in item and item['figure'] is not None and fig_count < 3:
            try:
                img = item['figure']
                if isinstance(img, Image.Image):
                    img_path = os.path.join(output_dir, f"figqa_sample_{fig_count+1}.png")
                    img.save(img_path)
                    
                    image_info[f"figqa_sample_{fig_count+1}"] = {
                        "id": item.get('id', 'N/A'),
                        "path": img_path,
                        "size": f"{img.size[0]}x{img.size[1]}",
                        "mode": img.mode,
                        "format": img.format if hasattr(img, 'format') else 'PIL',
                        "question": item.get('question', '')[:200],
                        "ideal_answer": item.get('ideal', ''),
                        "subtask": item.get('subtask', '')
                    }
                    
                    print(f"✓ Saved FigQA sample {fig_count+1}: {img_path}")
                    print(f"  Size: {img.size[0]}x{img.size[1]}")
                    print(f"  Question: {item.get('question', '')[:100]}...")
                    fig_count += 1
            except Exception as e:
                print(f"Error processing FigQA image {idx}: {e}")
    
    # Load TableQA subset
    print("\nLoading TableQA subset...")
    tableqa = load_dataset("futurehouse/lab-bench", "TableQA")['train']
    
    print(f"TableQA has {len(tableqa)} samples")
    table_count = 0
    
    for idx, item in enumerate(tableqa):
        if 'tables' in item and item['tables'] is not None and table_count < 3:
            try:
                tables = item['tables']
                # TableQA has a list of images, take the first one
                if isinstance(tables, list) and len(tables) > 0:
                    img = tables[0]
                    if isinstance(img, Image.Image):
                        img_path = os.path.join(output_dir, f"tableqa_sample_{table_count+1}.png")
                        img.save(img_path)
                        
                        image_info[f"tableqa_sample_{table_count+1}"] = {
                            "id": item.get('id', 'N/A'),
                            "path": img_path,
                            "size": f"{img.size[0]}x{img.size[1]}",
                            "mode": img.mode,
                            "format": img.format if hasattr(img, 'format') else 'PIL',
                            "question": item.get('question', '')[:200],
                            "ideal_answer": item.get('ideal', ''),
                            "subtask": item.get('subtask', '')
                        }
                        
                        print(f"✓ Saved TableQA sample {table_count+1}: {img_path}")
                        print(f"  Size: {img.size[0]}x{img.size[1]}")
                        print(f"  Question: {item.get('question', '')[:100]}...")
                        table_count += 1
            except Exception as e:
                print(f"Error processing TableQA image {idx}: {e}")
    
    # Save image info to JSON
    info_path = os.path.join(output_dir, "image_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(image_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Image information saved to: {info_path}")
    print(f"✓ Total images extracted: {fig_count + table_count}")
    
    return image_info

if __name__ == "__main__":
    print("="*60)
    print("EXTRACTING SAMPLE IMAGES FROM LAB-BENCH DATASET")
    print("="*60)
    
    image_info = extract_sample_images()
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print("\nExtracted images are saved in: ./sample_images/")
    print("Image metadata is saved in: ./sample_images/image_info.json")