# # 載入相關套件
# !sudo apt-get install tesseract-ocr
# !sudo apt-get install tesseract-ocr-chi-tra
# !pip3 install pytesseract
import cv2 
import pytesseract
import matplotlib.pyplot as plt

# 載入圖檔
image = cv2.imread('./images_ocr/receipt.png')

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,6))
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# 參數設定
custom_config = r'--psm 6'
# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))

# 參數設定，只辨識數字
custom_config = r'--psm 6 outputbase digits'
# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))

# 參數設定白名單，只辨識有限字元
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6'
# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))

# 參數設定黑名單，只辨識有限字元
custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz --psm 6'
# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))

# 載入圖檔
image = cv2.imread('./images_ocr/chinese.png')

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,6))
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# 辨識多國文字，中文繁體、日文及英文
custom_config = r'-l chi_tra+jpn+eng --psm 6'
# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))


# 載入圖檔
image = cv2.imread('./images_ocr/chinese_2.png')

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,6))
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# 辨識多國文字，中文繁體、日文及英文
custom_config = r'-l chi_tra+jpn+eng --psm 6'
# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))