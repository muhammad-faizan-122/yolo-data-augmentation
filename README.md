# How to apply data augmentation on YOLOv5 or YOLOv8 dataset and save the augmented results.
## Input 
![input image](input-ds/images/image_1.jpg)
## Output
![input label](out-aug-ds/images/image_1_aug_out.png)
## Output Visualization
![input label](output_vis.png)
- **input-ds** contain the input of YOLOv8 and YOLOv5 which are following directories.
    - Images directory contains the images
    - labels directory contains the .txt files. Each .txt file contains the normalized bounding boxes in a following format.
- **out-aug-ds** contain the augmented output contains following directories.
    - Images directory contains the augmented images.
    - labels directory contains the augmented labels.
- **controller** contain following scripts.
    - **apply_album_aug.py** contain the augmentated operations.
    - **validate_results.py** draw the augmented labels on augmented image to visualize the results.
    - **album_to_yolo_bb.py** is used to convert to labels in albumentation format to yolo format
    - **get_album_bb.py** is used to get labels in albumentation format from input yolo format.
    - **workflow.py** contain the pipeline to get the desired results.
    - **save_augs.py** to save the augmented results.
- **CONSTANT.yaml** contain following contants need to update on according to your case.
    - inp_img_pth for input images path
    - inp_lab_pth for input labels path
    - out_img_pth for output image path
    - out_lab_pth for output labels path
    - transformed_file_name: use to name augmented output to differentiate from other input dataset.
    - CLASSES: list of input class name according to class number. 
- step to apply augmentation on your own dataset.
    - install requirements using ```pip install -r requirements.txt```
    - provide the input and output path in **CONSTANT.yaml** file.
    - update the name of transformed_file_name in CONSTANT.yaml
    - Update the list of classes list in CONSTANT.yaml
    - update the classes name in according to your case in controller/get_album_bb.py and controller/get_album_bb.py
    - run the ```script python3 run.py```