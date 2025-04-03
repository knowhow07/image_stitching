from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List
from scipy.ndimage import distance_transform_edt
from helpers import genSIFTMatches


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    
    src = np.asarray(src_pts_nx2)
    dst = np.asarray(dest_pts_nx2)
    x, y, u, v = src[:,0], src[:,1], dst[:,0], dst[:,1]
    A = np.zeros((9,9))
    j = 0
    for i in range(4):
        A[j,:] = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*u[i], y[i]*u[i], u[i]])
        A[j+1,:] = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*v[i], y[i]*v[i], v[i]])
        j += 2
    
    sym_mat = np.dot(A.T,A)
    eig_val, eig_vec = np.linalg.eig(sym_mat)

    idx = np.argsort(eig_val)[0] 
    eig_vec = eig_vec[:,idx]
    H = np.reshape(eig_vec,(3,3))
    #print(H)
    return H
    


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    src = np.asarray(src_pts_nx2)
    src = np.hstack((src, np.ones((src.shape[0],1))))
    dest = H_3x3 @ src.T
    dest = dest / dest[2,:]
    dest = dest.T
    return dest[:,:2]


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
   

    # Create new images to draw lines on
    result_width = img1.shape[1] + img2.shape[1]
    result_height = max(img1.shape[0], img2.shape[0])
    result = Image.new('RGB', (result_width, result_height))

    # Paste the original images onto the result image
    result.paste(img1_pil, (0, 0))
    result.paste(img2_pil, (img1.shape[1], 0))

    # Create a draw object
    draw = ImageDraw.Draw(result)
    

    # Draw lines between corresponding points
    for pt1, pt2 in zip(pts1_nx2, pts2_nx2):
        pt2a = pt2[0]+ img1.shape[1]
        print(pt1,pt2)
        
        draw.line((pt1[0],pt1[1],pt2a,pt2[1]), fill='green', width=2)

    result_arr = np.array(result)

    return result_arr


def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    src_img = np.array(src_img)
    dest_img = np.zeros((canvas_shape[0], canvas_shape[1], 3))
    dest_mask = np.zeros((canvas_shape[0], canvas_shape[1]), dtype=np.uint8)
    
    for x in range(canvas_shape[1]):
        for y in range(canvas_shape[0]):
            dest = np.array([x,y,1])
            src = np.dot(destToSrc_H,dest)
            src = src / src[2]
            src = src[:2]
            src = src.astype(int)
            if src[0] >= 0 and src[0] < src_img.shape[1] and src[1] >= 0 and src[1] < src_img.shape[0]:
                dest_img[y,x] = src_img[src[1],src[0]]
                dest_mask[y,x] = 1
                

    
    
    
    
    return  dest_mask.astype(bool),dest_img


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    mask1 = np.array(mask1) > 0
    mask2 = np.array(mask2) > 0
    
   
    
    
    if mode == 'overlay':
        overlayed = img1.copy()
        overlayed[mask2] = img2[mask2]
        return overlayed.astype(np.uint8)
        
        
    elif mode == 'blend':
        out_img = np.zeros_like(img1)
        mask1 = distance_transform_edt(mask1)
        mask2 = distance_transform_edt(mask2)
        out_img = (img1 * mask1[..., np.newaxis]+ img2 * mask2[..., np.newaxis]) / (mask1 + mask2)[..., np.newaxis]
        return out_img.astype(np.uint8)

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    max_inliers_id = np.array([])
    H_max = np.zeros((3,3))
    for i in range(ransac_n):
        idx = np.random.choice(src_pt.shape[0], 4, replace=False,)
        H = computeHomography(src_pt[idx], dest_pt[idx])
        dest_pt_pred = applyHomography(H, src_pt)
        dist = np.linalg.norm(dest_pt - dest_pt_pred, axis=1)
        inliers_id = np.where(dist < eps)[0]
        if len(inliers_id) > len(max_inliers_id):
            max_inliers_id = inliers_id
            H_max = H
    
  #  print(max_inliers_id, H_max)
            
    return max_inliers_id, H_max
#https://kushalvyas.github.io/stitching.html
def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    initial_image = np.array(args[0])
    args = args[1:]
    for index, image_np in enumerate(args):
        current_image = image_np
        
        
        corners = np.array([[0, 0], [current_image.shape[1] -1, 0], [current_image.shape[1] -1, current_image.shape[0]-1], [0, current_image.shape[0] -1]])

        
        
        source_keypoints, destination_keypoints = genSIFTMatches(current_image, initial_image)
        source_keypoints = source_keypoints[:, [1, 0]] 
        destination_keypoints = destination_keypoints[:, [1, 0]]
        
        np.random.seed(549)
        inliers, homography_matrix = runRANSAC(source_keypoints, destination_keypoints, ransac_n=800, eps=0.3)
        
        
        newcorners = applyHomography(homography_matrix, corners)  
        
        width, height, top_x, top_y = findCorners(newcorners, initial_image)

        canvas = np.zeros((height, width, 3))
        canvas[top_y:initial_image.shape[0] + top_y, top_x:initial_image.shape[1] + top_x, :] = initial_image

        canvas_mask = np.any(canvas != 0, axis=-1).astype(int)
        
        T= np.array([[1, 0, top_x], [0, 1, top_y], [0, 0, 1]])
        updated_homography_matrix = T @ homography_matrix
        warped_mask, warped_image = backwardWarpImg(current_image, np.linalg.inv(updated_homography_matrix), [height, width])
        
        canvas_mask = canvas_mask.squeeze()
        warped_mask = warped_mask.squeeze()
        print(canvas_mask.shape, warped_mask.shape, canvas.shape, warped_image.shape)
        blended_image = blendImagePair((canvas*255).astype(np.uint8), (canvas_mask*255).astype(np.uint8), (warped_image*255).astype(np.uint8), (warped_mask*255).astype(np.uint8), mode='blend')
        
        initial_image = blended_image / 255.0
        
        return blended_image
        

def findCorners(corners, initial_image):
    min_x = np.min(corners[:, 0])
    min_y = np.min(corners[:, 1])
    if min_x < 0:
        top_x = int(-min_x)
    else:
        top_x = 0
    
    if min_y < 0:
        top_y = int(-min_y)
    else:
        top_y = 0
    
    
    max_width  = max(int((np.max(corners[:, 0]) + top_x)), initial_image.shape[1] + top_x)
    max_height = max(int((np.max(corners[:, 1]) + top_y)), initial_image.shape[0] + top_y)
    
    return max_width, max_height, top_x, top_y

    

   
    
