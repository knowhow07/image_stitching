from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def stitch_image():
    from stitch_img import stitchImg
    img_center = np.array(Image.open('data/mountain_center.png')) / 255.0
    img_left = np.array(Image.open('data/mountain_left.png')) / 255.0
    img_right = np.array(Image.open('data/mountain_right.png')) / 255.0

    # You are free to change the order of input arguments
    stitched_img = stitchImg(img_center, img_left, img_right)

    # Save the stitched image
    stitched_img = Image.fromarray((stitched_img).astype(np.uint8))
    stitched_img.save('outputs/stitched_img.png')


if __name__ == '__main__':
    stitch_image()