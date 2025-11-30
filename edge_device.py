import requests
from PIL import Image
import io
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# FastAPI 服務的 URL
FASTAPI_URL = "http://192.168.50.10:8000/grounding_dino/"

def main():
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(camera_config)
    picam2.start()

    print(f"連接到 FastAPI 服務：{FASTAPI_URL}")
    print("按下 'q' 鍵或關閉視窗以停止。")

    try:
        while True:
            # 從相機獲取一幀圖片
            array = picam2.capture_array()
            image = Image.fromarray(array)

            # 將 PIL Image 轉換為 BytesIO，以便傳送到 FastAPI
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}

            try:
                # 呼叫 FastAPI API
                response = requests.post(FASTAPI_URL, files=files, timeout=10) # 設置 timeout
                response.raise_for_status() # 如果響應狀態碼不是 200，則拋出 HTTPError

                # 接收處理後的圖片
                processed_image_data = response.content
                processed_image = Image.open(io.BytesIO(processed_image_data))

                # 將 PIL Image 轉換為 OpenCV 格式以便顯示
                processed_cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)

                # 顯示處理後的圖片
                cv2.imshow("Grounding DINO Processed Image", processed_cv_image)

            except requests.exceptions.Timeout:
                print("API 請求超時，請檢查 FastAPI 服務是否正常運行且響應速度足夠快。")
            except requests.exceptions.RequestException as e:
                print(f"API 請求失敗: {e}")
            except Exception as e:
                print(f"處理錯誤: {e}")

            # 等待按鍵，如果按下 'q' 則退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01) # 避免 CPU 佔用過高

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("程式已停止。")

if __name__ == "__main__":
    main()
