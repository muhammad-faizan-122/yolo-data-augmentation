
def single_obj_bb_yolo_conversion(transformed_bboxes, class_names):       
    if len(transformed_bboxes):                
        class_num = class_names.index(transformed_bboxes[-1])           
        bboxes = list(transformed_bboxes)[:-1] # .insert(0, '0')
        bboxes.insert(0, class_num)
    else:
        bboxes = []
    return bboxes


def multi_obj_bb_yolo_conversion(aug_labs, class_names):
    yolo_labels = []
    for aug_lab in aug_labs:        
        bbox = single_obj_bb_yolo_conversion(aug_lab, class_names)
        yolo_labels.append(bbox)
    return yolo_labels
