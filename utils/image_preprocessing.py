import os, glob
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
