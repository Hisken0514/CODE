#這支程式將圖片畫線並將結果存下
import cv2
import pytesseract
from PIL import Image
import concurrent.futures

# 設置 Tesseract 可執行文件的路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 使用相對路徑
image_path = r'line_3.png'
image2_path = r'image.png'

# 使用 OpenCV 加載圖片
image = cv2.imread(image_path)
image2 = cv2.imread(image2_path)

# 檢查圖片是否成功加載
if image is None:
    print("Error: 無法加載圖片。請檢查路徑是否正確以及文件是否存在。")
else:
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
    min_area = 2500  # 需要調整這個值

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_contour, contour) for contour in contours]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 初始化邊界
    xmin1, xmax1, ymin1, ymax1 = 1000, 0, 1000, 0
    xmin2, xmax2, ymin2, ymax2 = 1000, 0, 1000, 0

    for x, y, w, h, text in results:
        print(f"方框座標和大小: x={x}, y={y}, w={w}, h={h}, 識別的字: {text}")

        # 更新邊界
        if xmin1 > x:
            xmin1, ymin1 = x, y
            xmin2, ymin2 = x, y + h
        if xmax1 < x + w:
            xmax1, ymax1 = x + w, y
            xmax2, ymax2 = x + w, y + h

    # 顯示結果

    cv2.line(image2, (xmin1, ymin1), (xmax1, ymax1), (0, 0, 255), 3)
    cv2.line(image2, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 255), 3)

    # 顯示圖片
    #cv2.imshow('Result', gray)
    #cv2.waitKey(0)
    cv2.imshow('Result2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
