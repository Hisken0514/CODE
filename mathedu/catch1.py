#這支程式將圖片轉為灰階並將詩詞切成四行
import cv2
import pytesseract
from PIL import Image
import concurrent.futures

# 設置 Tesseract 可執行文件的路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 使用相對路徑
image_path = r'writingtest\00045_5431421.jpg'

# 使用 OpenCV 加載圖片
image = cv2.imread(image_path)

# 檢查圖片是否成功加載
if image is None:
    print("Error: 無法加載圖片。請檢查路徑是否正確以及文件是否存在。")
else:
    # 設定縮小比例
    orig_height, orig_width = image.shape[:2]
    max_dim = 1000
    scale_factor = max_dim / orig_width if orig_width > orig_height else max_dim / orig_height
    width = int(orig_width * scale_factor)
    height = int(orig_height * scale_factor)
    dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 2500

    def process_contour(contour):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            return None
        roi = resized_image[y:y + h, x:x + w]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(roi_pil, lang='chi_tra')
        return x, y, w, h, text.strip()

    results = []
    y_coords = []  # 用來儲存行的 Y 座標

    # 使用多執行緒加快文字區域處理
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_contour, contour) for contour in contours]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 將結果按 Y 座標進行排序
    results.sort(key=lambda x: x[1])

    # 將每一行的 Y 座標儲存下來
    for x, y, w, h, text in results:
        print(f"方框座標和大小: x={x}, y={y}, w={w}, h={h}, 識別的字: {text}")
        #cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # 判斷與上一個 Y 座標的差異來決定是否是新的一行
        if not y_coords or abs(y_coords[-1] - y) > 50:
            y_coords.append(y)

    # 根據最後一行的高度來調整裁切範圍
    if results:
        last_y, last_h = results[-1][1], results[-1][3]  # 最後一行的 y 和高度 h
        last_y_end = last_y + last_h
        y_coords.append(min(last_y_end, gray.shape[0]))  # 添加最後一行結束的位置

    # 對每行進行裁切
    cropped_lines = []
    for i in range(len(y_coords) - 1):
        # 設定邊距
        margin = 5
        cropped_line = gray[y_coords[i] - margin:y_coords[i + 1] + margin, 0:gray.shape[1]]
        cropped_lines.append(cropped_line)
        # 儲存裁切後的圖片
        cv2.imwrite(f'line_{i + 1}.png', cropped_line)

    # 顯示滑鼠座標的回調函數
    def show_mouse_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            temp_image = gray.copy()
            cv2.putText(temp_image, f"X: {x}, Y: {y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Result', temp_image)

    # 設置滑鼠回調
    cv2.namedWindow('Result')
    cv2.setMouseCallback('Result', show_mouse_coordinates)

    # 顯示行裁切結果
    print("行 Y 座標：", y_coords)
    cv2.imshow('Result', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
