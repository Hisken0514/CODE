import cv2
import pytesseract
from PIL import Image
import concurrent.futures
import os

# 設置 Tesseract 可執行文件的路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 圖片資料夾路徑
image_folder = 'writingtest'
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# 迭代資料夾中的所有圖片檔案
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: 無法加載圖片 {image_file}。")
        continue

    # 獲取原圖的寬和高
    orig_height, orig_width = image.shape[:2]

    # 設定最大寬度或高度
    max_dim = 1000

    # 計算縮放比例，保持寬高比例
    scale_factor = max_dim / max(orig_width, orig_height)

    # 根據計算出的縮放比例來調整圖片大小
    dim = (int(orig_width * scale_factor), int(orig_height * scale_factor))
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 將圖片轉換為灰度圖
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # 使用自適應閾值處理
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 找到輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 設定最小面積閾值
    min_area = 2500

    # 處理每個輪廓的函數
    def process_contour(contour):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            return None  # 忽略面積小於最小值的輪廓
        
        roi = resized_image[y:y + h, x:x + w]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(roi_pil, lang='chi_tra')
        return x, y, w, h, text.strip()

    results = []
    y_coords = []  # 用來儲存行的 Y 座標

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_contour, contour) for contour in contours]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 排序輪廓根據 Y 座標
    results.sort(key=lambda x: x[1])

    # 調整行之間的距離參數來控制裁切到第 4 行
    for x, y, w, h, text in results:
        if not y_coords or abs(y_coords[-1] - y) > 50:  # 使用50px作為初始閾值
            y_coords.append(y)

    # 如果行數不等於 4，調整閾值並重新進行裁切
    max_iterations = 10  # 最大重試次數
    margin = 5
    while len(y_coords) != 5 and max_iterations > 0:  # line4 應該有 5 個Y座標點
        if len(y_coords) > 5:  # 如果行數超過 4，減小行間距
            adjustment = -10  # 減小行間距
        else:  # 如果行數不足 4，增加行間距
            adjustment = 10  # 增加行間距
        
        y_coords = []  # 清空 y 座標列表
        for x, y, w, h, text in results:
            if not y_coords or abs(y_coords[-1] - y) > (50 + adjustment):  # 動態調整行間距
                y_coords.append(y)
        
        max_iterations -= 1

    # 確保我們只裁切到 4 行
    y_coords = y_coords[:5]

    # 根據行座標進行裁切並儲存
    for i in range(len(y_coords) - 1):
        cropped_line = gray[y_coords[i] - margin:y_coords[i + 1] + margin, 0:gray.shape[1]]
        cropped_image_path = os.path.join(output_folder, f'{image_file}_line_{i + 1}.png')
        cv2.imwrite(cropped_image_path, cropped_line)

    # 儲存畫線結果
    output_image_path = os.path.join(output_folder, f"processed_{image_file}")
    cv2.imwrite(output_image_path, cropped_line)

cv2.destroyAllWindows()
