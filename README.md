# How to apply data augmentation on YOLOv5 or YOLOv8 dataset and save the augmented results.
## input 
![input image](input-ds/images/image_1.jpg)
![input label](input-ds/images/image_1.jpg)
## output
![output]()


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
- step to apply augmentation on dataset.
    - install requirements using ```pip install -r requirements.txt```
    - provide the input and output path in **CONSTANT.yaml** file.
    - provide the provide transformed