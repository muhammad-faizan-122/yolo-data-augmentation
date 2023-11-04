# YOLOv5/YOLOv8 Data Augmentation with Albumentations
This GitHub repository offers a solution for augmenting datasets for YOLOv8 and YOLOv5 using the Albumentations library. Albumentations is a Python package designed for image augmentation, providing a simple and flexible approach to perform various image transformations. For more detail you can refer my medium [article](https://medium.com/red-buffer/apply-data-augmentation-on-yolov5-yolov8-dataset-958e89d4bc5d).

## Input 
![input image](input-ds/images/image_1.jpg)
## Output
![input label](bb_image/image_1_aug_out.png)

## Directories description
- **input-ds** contain the input of YOLOv8 and YOLOv5 which are following directories.
    - Images directory contains the images
    - labels directory contains the .txt files. Each .txt file contains the normalized bounding boxes in a following format.
- **out-aug-ds** contain the augmented output contains following directories.
    - Images directory contains the augmented images.
    - labels directory contains the augmented labels.
- **bb_image** This directory contains images with bounding boxes for visualizing the results of augmented data. This is for validation; bounding boxes should be correctly drawn on the objects of interest.
- **utils.py**: Contains all user-defined helper function.
- **main.py**: Contains Yolo dataset augmentor pipeline
- **CONSTANT.yaml** contain following contants need to update on according to your case.
    - inp_img_pth for input images path
    - inp_lab_pth for input labels path
    - out_img_pth for output image path
    - out_lab_pth for output labels path
    - transformed_file_name: use to name augmented output to differentiate from other input dataset.
    - CLASSES: list of input class name according to class number. 
## Usage
- step to apply augmentation on your own dataset.
    - Create Virtual Environment.
    - Install requirements using 
        ```
        pip install -r requirements.txt
        ```
    - Provide the input and output path in **CONSTANT.yaml** file.
    - Update the name of transformed_file_name in CONSTANT.yaml otherwise code will overwrite last augmentations.
    - Provide the list of name of  classes in CONSTANT.yaml in a same sequence as used to assign class numbers in the YOLO dataset labeling.
        - For example, if you provided a class list as ```['obj1', 'obj2', 'obj3']```, the class number used for 'obj1' in the label file should be 0, similarly for 'obj2', the class number should be 1, and so on.
    - Run the pipeline using 
        ```
        python3 run.py
        ```


