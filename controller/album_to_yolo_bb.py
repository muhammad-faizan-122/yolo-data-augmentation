
def single_obj_bb_yolo_format(transformed_bboxes):       
    if len(transformed_bboxes):
        if transformed_bboxes[-1] == 'obj1':            
            class_ = 0
        elif transformed_bboxes[-1] == 'obj2':            
            class_ = 1            
        elif transformed_bboxes[-1] == 'obj3':            
            class_ = 2        
        bboxes = list(transformed_bboxes)[:-1] # .insert(0, '0')
        bboxes.insert(0, class_)
    else:
        bboxes = []
    return bboxes


def multi_obj_bb_yolo_conversion(aug_labs):
    yolo_labels = []
    for aug_lab in aug_labs:        
        bbox = single_obj_bb_yolo_format(aug_lab)
        yolo_labels.append(bbox)
    return yolo_labels
