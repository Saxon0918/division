# 标记区域
import pandas as pd
import cv2
from skimage import measure, color

path = "images/NEWNYC/NYC_thinned.bmp"
img = cv2.imread(path)
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_temp = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
labels, number = measure.label(img_temp, connectivity=1, return_num=True)
print("number:" + str(number))
# print("labels:" + labels)
dst = color.label2rgb(labels, bg_label=0)  # bg_label=0要有，不然会有警告

for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j] == 1:
            dst[i][j] = 1
# print(labels[620,995])

# cv2.imshow("666", dst)
cv2.imwrite("images/NEWNYC/NYC_label.bmp", dst * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# label = pd.DataFrame(labels)
# label.to_excel("data/NYC/label.xlsx")