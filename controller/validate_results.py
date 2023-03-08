
import cv2
import pybboxes as pbx


def draw_yolo(image, labels):
    H, W = image.shape[:2]    
    for label in labels:                
        yolo_normalized = label[1:]
        box_voc = pbx.convert_bbox(tuple(yolo_normalized), from_type="yolo", to_type="voc", image_size=(W,H))
        cv2.rectangle(image, (box_voc[0], box_voc[1]), 
                    (box_voc[2], box_voc[3]), (0, 0, 255), 1)
    cv2.imwrite("output_vis.png", image)
    cv2.imshow("output_vis", image)
    cv2.waitKey(0)
