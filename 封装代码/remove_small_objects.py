from skimage import morphology
import cv2
import os

dir = r"submit\0.9527\infers"
img_save_path="infers"
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)
for img_name in os.listdir(dir):
    img = cv2.imread(os.path.join(dir, img_name), -1)
    img = morphology.remove_small_objects(img, min_size=50)
    cv2.imwrite(os.path.join(img_save_path, img_name), img)
