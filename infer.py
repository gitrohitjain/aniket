from objdet.yolov5.detect import run
from classification.trainer import classify
import os

if __name__ == '__main__':
    weights = '/Users/rohitjain/Desktop/aniket/objdet/aoi_model/best.pt' #crop detection
    imgsz = (640, 640)
    conf_thres = 0.25
    max_det = 1
    save_crop = True
    # source = '/Users/rohitjain/Desktop/aniket/data/images/'
    source ='/Users/rohitjain/Desktop/aniket/data/images/opencv_09-01-2023_S41_True_pre-pHAdj.png'
    
    mycroppath = run(weights=weights, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, save_crop=save_crop, source=source)
    print('cropped image path: ', mycroppath)    
    result = classify(mycroppath)
    print('classification result: ', 'True' if result else 'False')
    os.remove(mycroppath)

    