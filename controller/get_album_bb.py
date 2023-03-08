def get_album_bb_list(yolo_bbox, class_names):    
    album_bb =[]
    str_bbox_list = yolo_bbox.split(' ')        
    for index, value in enumerate(str_bbox_list):        
        if index == 0:  # class number is placed at index 0
            class_name = class_names[int(value)]            
        else:
            album_bb.append(float(value))    
    album_bb.append(class_name)  # [x_center, y_center, width, height, class_name]
    return album_bb


def get_album_bb_lists(yolo_str_labels, classes):
    album_bb_lists = []
    yolo_list_labels = yolo_str_labels.split('\n')
    for yolo_str_label in yolo_list_labels:               
        if len(yolo_str_label) > 0:
            album_bb_list = get_album_bb_list(yolo_str_label, classes)        
            album_bb_lists.append(album_bb_list)            
    return album_bb_lists


def get_bboxes_list(inp_lab_pth, classes):
    yolo_str_labels = open(inp_lab_pth, "r").read()     
    if yolo_str_labels:
        if "\n" in yolo_str_labels:
            print("multi-objs")
            album_bb_lists = get_album_bb_lists(yolo_str_labels, classes)        
        else:        
            print("single line ")
            album_bb_lists = get_album_bb_list(yolo_str_labels, classes)
            album_bb_lists = [album_bb_lists]  # require 2d list in alumbentation function
    else:
        print("No object")
        album_bb_lists = []
    return album_bb_lists
