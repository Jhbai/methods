import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "D://LLM//grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"----- Loading model to {device}... -----")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("----- Model loaded successfully -----")

def capture(image: Image.Image, text: str):

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    draw = ImageDraw.Draw(image)

    # 確保有偵測到物件才畫圖
    print("輸出結果:", results)
    if len(results) > 0 and "boxes" in results[0]:
        for box, label in zip(results[0]["boxes"], results[0]["labels"]):
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), label, fill="red")
    return image
