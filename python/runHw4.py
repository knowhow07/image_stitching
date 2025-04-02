import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def runHw4():
    # runHw4 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw4('all') 
    # without any error.
    #
    # Usage:
    # python runHw4.py                  : list all the registered functions
    # python runHw4.py 'function_name'  : execute a specific test
    # python runHw4.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        # 'honesty': honesty, 
        'challenge1a': challenge1a, 
        'challenge1b': challenge1b, 
        'challenge1c': challenge1c, 
        'challenge1d': challenge1d, 
        'challenge1e': challenge1e, 
        'challenge1f': challenge1f,
    }
    run_tests(args.function_name, fun_handles)

# # Academic Honesty Policy
# def honesty():
#     from signAcademicHonestyPolicy import sign_academic_honesty_policy
#     # Type your full name and uni (both in string) to state your agreement 
#     # to the Code of Academic Integrity.
#     sign_academic_honesty_policy('Dev Mehrotra', 'dmehrotra')

# Tests for Challenge 1: Panoramic Photo App

# Test homography
def challenge1a():
    from helpers import ImageClicker
    from hw4_challenge1 import computeHomography, applyHomography, showCorrespondence

    orig_img = Image.open('data/portrait.png')
    orig_img = np.array(orig_img)

    warped_img = Image.open('data/portrait_transformed.png')
    warped_img = np.array(warped_img)

    clicker = ImageClicker('data/portrait.png', 4)
    clicker.run()
    src_pts_nx2 = clicker.get_points()
    print("Source image points", src_pts_nx2)

    clicker = ImageClicker('data/portrait_transformed.png', 4)
    clicker.run()
    dest_pts_nx2 = clicker.get_points()
    print("Destination image points", dest_pts_nx2)

    # verification the shape and points
    print("src_pts_nx2 type:", type(src_pts_nx2), "shape:", np.array(src_pts_nx2).shape)
    if src_pts_nx2 is None or dest_pts_nx2 is None:
        raise ValueError("You must select 4 points on each image.")
    src_pts_nx2 = np.array(src_pts_nx2)
    dest_pts_nx2 = np.array(dest_pts_nx2)
    if src_pts_nx2.shape != (4, 2) or dest_pts_nx2.shape != (4, 2):
        raise ValueError(f"Incorrect shape for selected points: got {src_pts_nx2.shape} and {dest_pts_nx2.shape}, expected (4, 2) each.")

    H_3x3 = computeHomography(src_pts_nx2, dest_pts_nx2)

    clicker = ImageClicker('data/portrait.png', 4)
    clicker.run()
    src2_pts_nx2 = clicker.get_points()
    if src2_pts_nx2 is None or len(src2_pts_nx2) < 4:
        raise ValueError("You must select 4 test points.")
    src2_pts_nx2 = np.array(src2_pts_nx2)

    print("Test points", src2_pts_nx2)

    test_pts_nx2 = src2_pts_nx2
    dest_pts_nx2 = applyHomography(H_3x3, test_pts_nx2)

    result_img = showCorrespondence(orig_img, warped_img, test_pts_nx2, dest_pts_nx2)
    result_img = Image.fromarray(result_img.astype(np.uint8))
    result_img.save('outputs/homography_result.png')



# Test wrapping
def challenge1b(): 
    from helpers import ImageClicker
    from matplotlib import pyplot as plt
    from hw4_challenge1 import computeHomography, backwardWarpImg
        
    bg_img = np.array(Image.open('data/Osaka.png')) / 255.0
    portrait_img = np.array(Image.open('data/portrait_small.png')) / 255.0

    # Estimate homography
    # bg_pts = np.array([[xs1, ys1], [xs2, ys2], [xs3, ys3], [xs4, ys4]])
    # # portrait_pts = np.array([[xd1, yd1], [xd2, yd2], [xd3, yd3], [xd4, yd4]])
    # clicker = ImageClicker('data/Osaka.png', 4)
    # clicker.run()
    # bg_pts = clicker.get_points()
    # print("Background points", bg_pts)
    # clicker = ImageClicker('data/portrait_small.png', 4)
    # clicker.run()
    # portrait_pts = clicker.get_points()
    bg_pts = np.array([(99, 19), (276, 72), (84, 433), (279, 420)])
    portrait_pts = np.array([(0, 0), (portrait_img.shape[1], 0), (0, portrait_img.shape[0]), (portrait_img.shape[1], portrait_img.shape[0])])
    print("Portrait points", portrait_pts)
    H_3x3 = computeHomography(portrait_pts, bg_pts)

    # Warp the portrait image
    dest_canvas_shape = bg_img.shape[:2]
    mask, dest_img = backwardWarpImg(portrait_img, np.linalg.inv(H_3x3), dest_canvas_shape)
    # mask should be of the type logical
    mask = ~mask
    # Superimpose the image
    print(mask.shape, bg_img.shape, dest_img.shape)
    result = bg_img * np.stack([mask, mask, mask], axis=2) + dest_img
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save('outputs/Van_Gogh_in_Osaka.png')

    plt.figure()
    plt.imshow(result)
    plt.title('Van Gogh in Osaka')
    plt.show()
# Test RANSAC -- outlier rejection
def challenge1c():
    from helpers import genSIFTMatches
    from hw4_challenge1 import showCorrespondence, runRANSAC
    img_src = np.array(Image.open('data/mountain_left.png').convert('RGB'))
    img_dst = np.array(Image.open('data/mountain_center.png').convert('RGB'))

    xs, xd = genSIFTMatches(img_src, img_dst)
    # xs and xd are the centers of matched frames
    # xs and xd are nx2 matrices, where the first column contains the x
    # coordinates and the second column contains the y coordinates
    xs = xs[:, [1, 0]]
    xd = xd[:, [1, 0]]
    # Assuming showCorrespondence is a function defined elsewhere in your code
    print(xs, xd)
    before_img = showCorrespondence(img_src, img_dst, xs, xd)
    before_img = Image.fromarray((before_img).astype(np.uint8))
    before_img.save('outputs/before_ransac.png')

    plt.figure()
    plt.imshow(before_img)
    plt.title('Before RANSAC')
    plt.show()
    np.random.seed(0)
    # Use RANSAC to reject outliers
    ransac_n = 100 # Max number of iterations
    ransac_eps = 30  # Acceptable alignment error 
    # Assuming runRANSAC is a function defined elsewhere in your code
    inliers_id, _ = runRANSAC(xs, xd, ransac_n, ransac_eps)
    after_img = showCorrespondence(img_src, img_dst, xs[inliers_id, :], xd[inliers_id, :])
    after_img = Image.fromarray((after_img).astype(np.uint8))

    after_img.save('outputs/after_ransac.png')

    plt.figure()
    plt.imshow(after_img)
    plt.title('After RANSAC')
    plt.show()

# Test image blending
def challenge1d():
    from hw4_challenge1 import blendImagePair
    

    fish = np.array(Image.open('data/escher_fish.png').convert('RGBA'))
    fish, fish_mask = fish[:, :, :3], fish[:, :, 3]

    horse = np.array(Image.open('data/escher_horsemen.png').convert('RGBA'))
    horse, horse_mask = horse[:, :, :3], horse[:, :, 3]

    blended_result = blendImagePair(fish, fish_mask, horse, horse_mask, 'blend')
    blended_result = Image.fromarray((blended_result))
    blended_result.save('outputs/blended_result.png')

    overlay_result = blendImagePair(fish, fish_mask, horse, horse_mask, 'overlay')
    overlay_result = Image.fromarray((overlay_result))
    overlay_result.save('outputs/overlay_result.png')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(fish); axs[0, 0].set_title('Escher Fish')
    axs[0, 1].imshow(horse); axs[0, 1].set_title('Escher Horse')
    axs[1, 0].imshow(blended_result); axs[1, 0].set_title('Blended')
    axs[1, 1].imshow(overlay_result); axs[1, 1].set_title('Overlay')
    plt.show()

# Test image stitching
def challenge1e():
    from hw4_challenge1 import stitchImg
    img_center = np.array(Image.open('data/mountain_center.png')) / 255.0
    img_left = np.array(Image.open('data/mountain_left.png')) / 255.0
    img_right = np.array(Image.open('data/mountain_right.png')) / 255.0

    # You are free to change the order of input arguments
    stitched_img = stitchImg(img_center, img_left, img_right)

    # Save the stitched image
    stitched_img = Image.fromarray((stitched_img).astype(np.uint8))
    stitched_img.save('outputs/stitched_img.png')

# Test image stitching
def challenge1f():
    from hw4_challenge1 import stitchImg
    # Take three photos of a scene, and stitch them together to form a panorama
    # Take three photos of a scene, and stitch them together to form a panorama
    img_left = np.array(Image.open('data/Image1.jpg')) / 255.0
    img_center = np.array(Image.open('data/Image2.jpg')) / 255.0
    img_right = np.array(Image.open('data/Image3.jpg')) / 255.0

    stitched_img = stitchImg(img_center, img_left, img_right)

    # Save the stitched image
    stitched_img = Image.fromarray((stitched_img).astype(np.uint8))
    stitched_img.save('outputs/scene_panorama.png')

if __name__ == '__main__':
    runHw4()