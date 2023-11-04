from controller.album_to_yolo_bb import multi_obj_bb_yolo_conversion
from controller.album_to_yolo_bb import single_obj_bb_yolo_conversion
from controller.save_augs import save_aug_image, save_aug_lab
from controller.validate_results import draw_yolo
import albumentations as A


def apply_aug(image, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes):
    transform = A.Compose([
        A.RandomCrop(width=300, height=300),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=-1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),        
        A.Resize(300, 300)
    ], bbox_params=A.BboxParams(format='yolo'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']        
    tot_objs = len(bboxes)
    if tot_objs != 0:        
        if tot_objs > 1:
            transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        else:
            transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0], classes)]
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".png")             
        draw_yolo(transformed_image, transformed_bboxes)
    else:
        print("label file is empty")        
