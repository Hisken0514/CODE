import cv2
import pytesseract
from PIL import Image
import concurrent.futures
import os

# 設置 Tesseract 可執行文件的路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

output_folder = 'output_images4'  # 拿出資料夾
image_path = 'white.png'  # 背景圖
save_folder = 'final_output41'  # 存放資料夾
os.makedirs(save_folder, exist_ok=True)

# 迭代 output_images 文件夹中的所有图片文件
for image_file in os.listdir(output_folder):
    image_file_path = os.path.join(output_folder, image_file)
    image = cv2.imread(image_file_path)

    if image is None:
        print(f"Error: 無法加載圖片 {image_file}。")
        continue

    # 重新加載背景图片，确保每次迭代时使用独立的背景图片
    background_image = cv2.imread(image_path)
    if background_image is None:
        print("Error: 無法加載背景圖片。請檢查路徑是否正確以及文件是否存在。")
        break

    orig_height, orig_width = image.shape[:2]

    # 設置圖片最大值
    max_dim = 1000

    # 計算缩放比例
    scale_factor = max_dim / max(orig_width, orig_height)

    # 調整比例
    dim = (int(orig_width * scale_factor), int(orig_height * scale_factor))
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 轉灰階
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # 自適應二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 設定域值
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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_contour, contour) for contour in contours]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 初始化边界
    xmin1, xmax1, ymin1, ymax1 = 1000, 0, 1000, 0
    xmin2, xmax2, ymin2, ymax2 = 1000, 0, 1000, 0

    for x, y, w, h, text in results:
        # 更新边界
        if xmin1 > x:
            xmin1, ymin1 = x, y
            xmin2, ymin2 = x, y + h
        if xmax1 < x + w:
            xmax1, ymax1 = x + w, y
            xmax2, ymax2 = x + w, y + h

    # 在背景图像上绘制框线
    cv2.line(background_image, (xmin1, ymin1), (xmax1, ymax1), (0, 0, 255), 3)
    cv2.line(background_image, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 255), 3)

    # 保存每个处理后的图像
    output_image_path = os.path.join(save_folder, f'test1_{image_file}')
    cv2.imwrite(output_image_path, background_image)
    print(f"保存完成 {output_image_path}")

cv2.destroyAllWindows()
