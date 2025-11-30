from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
from io import BytesIO
import model
import os

app = FastAPI()

@app.post("/grounding_dino/")
async def grounding_dino_process_image(file: UploadFile = File(...)):
    # 讀取上傳的圖片
    image_data = await file.read()
    try:
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        return Response(content=f"Error decoding image: {str(e)}", status_code=400)

    processed_image = model.capture(image, "Please detect any human")

    # 將處理後的圖片轉換為 BytesIO 物件
    output_buffer = BytesIO()
    processed_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return Response(content=output_buffer.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
