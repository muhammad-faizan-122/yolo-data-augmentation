from utils import *


def run_yolo_augmentor():
    """
    Run the YOLO augmentor on a set of images.

    This function processes each image in the input directory, applies augmentations,
    and saves the augmented images and labels to the output directories.

    """
    imgs = [img for img in os.listdir(CONSTANTS["inp_img_pth"]) if is_image_by_extension(img)]

    for img_num, img_file in enumerate(imgs):
        print(f"{img_num}-image is processing...\n")
        image, gt_bboxes, aug_file_name = get_inp_data(img_file)
        apply_augmentation(image, gt_bboxes, aug_file_name)


if __name__ == "__main__":
    run_yolo_augmentor()