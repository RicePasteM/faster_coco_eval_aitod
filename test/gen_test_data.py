import torch
import numpy as np
import random

def generate_random_test_data(
    num_images=5000,
    num_categories=12,
    min_objects_per_image=10,
    max_objects_per_image=100,
    image_size_range=(600, 1500),
    min_box_size=20,
    max_box_size=200,
    min_predictions=1  # 确保每张图片至少有一个预测
):
    
    # 生成类别
    categories = [
        {"id": i+1, "name": f"category_{i+1}"} 
        for i in range(num_categories)
    ]
    
    # 生成图片
    images = []
    for i in range(num_images):
        width = random.randint(image_size_range[0], image_size_range[1])
        height = random.randint(image_size_range[0], image_size_range[1])
        images.append({
            "id": i+1,
            "height": height,
            "width": width
        })
    
    # 生成标注
    annotations = []
    ann_id = 1
    
    # 为每张图片生成标注
    for img in images:
        num_objects = random.randint(min_objects_per_image, max_objects_per_image)
        
        for _ in range(num_objects):
            # 随机选择类别
            category_id = random.randint(1, num_categories)
            
            # 生成随机框的尺寸
            box_w = random.randint(min_box_size, min(max_box_size, img["width"]//3))
            box_h = random.randint(min_box_size, min(max_box_size, img["height"]//3))
            
            # 确保框在图片范围内
            x = random.randint(0, img["width"] - box_w)
            y = random.randint(0, img["height"] - box_h)
            
            area = box_w * box_h
            
            annotations.append({
                "id": ann_id,
                "image_id": img["id"],
                "category_id": category_id,
                "bbox": [x, y, box_w, box_h],
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1
    
    # 生成预测结果
    predictions = {}
    for img in images:
        img_anns = [ann for ann in annotations if ann["image_id"] == img["id"]]
        # 确保至少有min_predictions个预测框
        num_preds = max(min_predictions, len(img_anns) + random.randint(-1, 2))
        
        boxes = []
        scores = []
        labels = []
        
        # 对真实框添加随机扰动
        for ann in img_anns[:num_preds-1]:
            x, y, w, h = ann["bbox"]
            # 添加随机偏移和尺度变化
            noise_scale = 0.1
            dx = w * random.uniform(-noise_scale, noise_scale)
            dy = h * random.uniform(-noise_scale, noise_scale)
            dw = w * random.uniform(-noise_scale, noise_scale)
            dh = h * random.uniform(-noise_scale, noise_scale)
            
            boxes.append([
                x + dx, 
                y + dy, 
                x + w + dx + dw, 
                y + h + dy + dh
            ])
            scores.append(random.uniform(0.5, 1.0))
            labels.append(ann["category_id"])
        
        # 如果预测框数量不足，添加随机预测框
        while len(boxes) < num_preds:
            box_w = random.randint(min_box_size, min(max_box_size, img["width"]//3))
            box_h = random.randint(min_box_size, min(max_box_size, img["height"]//3))
            x = random.randint(0, img["width"] - box_w)
            y = random.randint(0, img["height"] - box_h)
            
            boxes.append([x, y, x + box_w, y + box_h])
            scores.append(random.uniform(0.3, 0.7))
            labels.append(random.randint(1, num_categories))
        
        predictions[img["id"]] = {
            'boxes': torch.tensor(boxes),
            'scores': torch.tensor(scores),
            'labels': torch.tensor(labels)
        }
    
    gt_data = {
        "images": images,
        "categories": categories,
        "annotations": annotations
    }
    
    return gt_data, predictions

def save_test_data(output_file="test/gt_data.py"):
    gt_data, predictions = generate_random_test_data()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("import torch\n\n")
        f.write("gt_data = ")
        f.write(str(gt_data).replace("], ", "],\n    ").replace("}, ", "},\n    "))
        f.write("\n\n")
        f.write("predictions = {\n")
        for img_id, pred in predictions.items():
            f.write(f"    {img_id}: {{\n")
            f.write(f"        'boxes': torch.tensor({pred['boxes'].tolist()}),\n")
            f.write(f"        'scores': torch.tensor({pred['scores'].tolist()}),\n")
            f.write(f"        'labels': torch.tensor({pred['labels'].tolist()})\n")
            f.write("    },\n")
        f.write("}")

if __name__ == "__main__":
    save_test_data()