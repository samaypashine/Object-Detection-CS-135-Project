import os
import cv2
import pandas as pd
from IOU.IOU import *
from bb_box.box import *

def preprocessing(sed_detector, images_path, images_annotations):
    num = 0
    train_images, train_labels = [], []
    for e, i in enumerate(os.listdir(images_annotations)):
        if num < 10:
            try:
                if i.startswith("airplane"):
                    
                    filename = i.split(".")[0]+".jpg"
                    
                    image = cv2.imread(os.path.join(images_path, filename))
                    DF = pd.read_csv(os.path.join(images_annotations, i))
                    
                    
                    bb_values = bounding_box_from_DF(DF, img = None, draw = False, cummulate = True)


                    sed_detector.setBaseImage(image)
                    sed_detector.switchToSelectiveSearchFast()
                    detector_results = sed_detector.process()
                    
                    imout = image.copy()

                    counter = 0
                    falsecounter = 0
                    flag = 0
                    fflag = 0
                    bflag = 0
                    
                    for e,result in enumerate(detector_results):
                        if e < 2000 and flag == 0:
                            for bb_value in bb_values:
                                
                                x,y,w,h = result
                                iou = IOU(bb_value, {"x1" : x, "x2" : x + w, "y1" : y, "y2" : y + h })
                                
                                if counter < 30:
                                    if iou > 0.70:
                                        timage = imout[y : y + h, x : x + w]
                                        resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                                        train_images.append(resized)
                                        train_labels.append(1)
                                        counter += 1
                                else :
                                    fflag =1
                                
                                if falsecounter < 30:
                                    if iou < 0.3:
                                        timage = imout[y:y+h,x:x+w]
                                        resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                                        train_images.append(resized)
                                        train_labels.append(0)
                                        falsecounter += 1
                                else :
                                    bflag = 1
                            
                            
                            if fflag == 1 and bflag == 1:
                                # print("inside")
                                flag = 1
                num += 1
            
            except Exception as e:
                print("[ERROR]. Encountered Error {} : ".format(e), filename)
        
    return train_images, train_labels
