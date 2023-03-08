import cv2
import os


def save_aug_lab(transformed_bboxes, lab_pth, lab_name):      
    # print("transformed_bboxes: ", transformed_bboxes)      
    lab_out_pth = os.path.join(lab_pth, lab_name)
    with open(lab_out_pth, 'w') as output:
        for bbox in transformed_bboxes:
            print("transformed_bboxes_bbox:", bbox)
            updated_bbox = str(bbox).replace(',', ' ').replace('[', '').replace(']', '')
            output.write(updated_bbox + '\n')


def save_aug_image(transformed_image, out_img_pth, img_name):    
    out_img_path = os.path.join(out_img_pth,img_name)
    cv2.imwrite(out_img_path, transformed_image)
