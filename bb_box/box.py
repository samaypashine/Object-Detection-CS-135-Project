import cv2

def bounding_box_from_DF(df, img = None, draw = False, cummulate = False):
    values = []
    for row in df.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])

        if cummulate == True:
            values.append({"x1" : x1, "x2" : x2, "y1" : y1, "y2" : y2})

        if draw == True and img != None:
            try:
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0), 2)
            except Exception as e:
                print("[ERROR]. Cannot Draw on the Image.")
    return values
