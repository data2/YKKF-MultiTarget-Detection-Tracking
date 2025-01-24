from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
# 使用中文字体路径
font_path = "C:\\Windows\\Fonts\\msyh.ttc"  # 替换为你系统中的中文字体路径
font = ImageFont.truetype(font_path, size=30)

# 示例：加载图像并绘制中文文本
img = np.zeros((500, 500, 3), dtype=np.uint8)  # 创建一个黑色背景图像
img_pil = Image.fromarray(img)  # 转换为PIL图像
draw = ImageDraw.Draw(img_pil)
draw.text((50, 50), "目标检测", font=font, fill=(255, 255, 255))  # 在图像上绘制中文文本

# 显示图像
img = np.array(img_pil)  # 将PIL图像转换回OpenCV格式
cv2.imshow("Image with Chinese Text", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
