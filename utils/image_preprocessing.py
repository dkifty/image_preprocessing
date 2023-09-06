import os, glob
import numpy as np
import cv2

def mask2binary(MASK_DIR = 'mask_img', ext = 'png', resize = (500,500)): ## input은 누끼딴 png 이미지~
    DIR = os.getcwd()
    BI_DIR = os.path.join(DIR, MASK_DIR).replace(MASK_DIR, 'bi_img')
    if not os.path.exists(BI_DIR):
        os.mkdir(BI_DIR)
    
    mask_img = glob.glob(os.path.join(DIR, MASK_DIR, '/*.', ext))
    mask_img.sort()
    
    for a in mask_img:
        img = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_LINEAR)
        img = np.where(img>0, 255, img)
        cv2.imwrite(a.replace(MASK_DIR, 'bi_img'), img)
        
def img_zoom(img, zoom_factor=0):
    if zoom_factor == 0:
        return img
    
    height, width = img.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])

    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant', constant_values=0)
    assert result.shape[0] == height and result.shape[1] == width
    return result
