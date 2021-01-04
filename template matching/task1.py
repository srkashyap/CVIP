import argparse
import copy
import os
import cv2
import numpy as np

import utils

# Prewitt operator
prewitt_x = [[1, 0, -1]] * 3
prewitt_y = [[1] * 3, [0] * 3, [-1] * 3]

# Sobel operator
sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

def flip_x(img):
    """Flips a given image along x axis."""
    flipped_img = copy.deepcopy(img)
    center = int(len(img) / 2)
    for i in range(center):
        flipped_img[i] = img[(len(img) - 1) - i]
        flipped_img[(len(img) - 1) - i] = img[i]
    return flipped_img
def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    
    parser.add_argument(
        "--img_path", type=str, default="`",
        help="path to the image used for edge detection")
    
    parser.add_argument(
        "--kernel", type=str, default="sobel",
        choices=["prewitt", "sobel", "Prewitt", "Sobel"],
        help="type of edge detector used for edge detection")
    
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    
    args = parser.parse_args()
    return args
img_path = "data/proj1-task1.jpg"
kernel = "sobel"
rs_directory = "./results/"


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if not img.dtype() == np.uint8:
#            pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img
    
def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 255, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 *img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")

    cv2.imwrite(img_saving_path, img)
    
def zero_pad(img, pwx, pwy):
    """Pads a given image with zero at the border."""
    padded_img = copy.deepcopy(img)
    for i in range(pwx):
        padded_img.insert(0, [0 for value in enumerate(padded_img[i])])
        padded_img.insert(len(padded_img), [0 for value in enumerate(padded_img[-1])])
    for i, row in enumerate(padded_img):
        for j in range(pwy):
            row.insert(0, 0)
            row.insert(len(row), 0)
    return padded_img
def convolve2d(img, kernel):
    
    img1 = np.asarray(flip_x(img))
    pwx, pwy = 2, 2
    az = np.zeros((img1.shape[0]+pwx, img1.shape[1]+pwy))
    az[pwx-1:-pwx+1, pwy-1:-pwy+1] = img1
    
    kernel = np.asarray(kernel)
    
    img2=np.zeros((3,3))
    img3=np.zeros((256,256))
    for i in range(1,255):
        for j in range(1,255):
            img2=img1[i-1:i+2,j-1:j+2]
            img3[i-1][j-1] = np.dot(img2.reshape(1,-1), kernel.reshape(-1,1)) 
    img_conv=img3
    # TODO: implement this function.
  #  raise NotImplementedError
    return img_conv
    
    
def normalize(img):

    img = np.asarray(img)
    img = (img - img.min())/(img.max() - img.min())
    img *= 255
    img =  img.astype(int)
    print("Image")
    return img
    
def detect_edges(img, kernel, norm=True):
    """Detects edges using a given kernel.

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel used to detect edges.
        norm (bool): whether to normalize the image or not.

    Returns:
        img_edge: nested list (int), image containing detected edges.
    """
    print("Inside detect_edges")
    img_edges = convolve2d(img,kernel)
    if norm:
        print("Detect edges: Image normalized")
        img_edges = normalize(img)
    # TODO: detect edges using convolve2d and normalize the image containing detected edges using normalize.
    # raise NotImplementedError
    return img_edges
    
def edges(img):
    img = np.asarray(img)
    kx = np.asarray([[1,0,-1],[1,0,-1],[1,0,-1]])
    ky = np.asarray([[1,1,1],[0,0,0],[-1,-1,-1]])
    edge_x = detect_edges(img,kx)
    edge_y = detect_edges(img,ky)
    return edge_x,edge_y
    
    
def edge_magnitude(edge_x, edge_y):
    """Calculate magnitude of edges by combining edges along two orthogonal directions.

    Hints:
        Combine edges along two orthogonal directions using the following equation:

        edge_mag = sqrt(edge_x ** 2 + edge_y ** 2).

        Make sure that you normalize the edge_mag, so that the maximum pixel value is 1.

    Args:
        edge_x: nested list (int), image containing detected edges along one direction.
        edge_y: nested list (int), image containing detected edges along another direction.

    Returns:
        edge_mag: nested list (int), image containing magnitude of detected edges.
    """
    
    edge_mag = normalize(np.sqrt((edge_x**2) + (edge_y**2)))
    
    # TODO: implement this function.
#     raise NotImplementedError
    return edge_mag
    
def main():
    args = parse_args()

    img = read_image(args.img_path)

    if args.kernel in ["prewitt", "Prewitt"]:
        kernel_x = prewitt_x
        kernel_y = prewitt_y
    elif args.kernel in ["sobel", "Sobel"]:
        kernel_x = sobel_x
        kernel_y = sobel_y
    else:
        raise ValueError("Kernel type not recognized.")

    if not os.path.exists(args.rs_directory):
        os.makedirs(args.rs_directory)

    img_edge_x = detect_edges(img, kernel_x, False)
    img_edge_x = np.asarray(img_edge_x)
    write_image(normalize(img_edge_x), os.path.join(args.rs_directory, "{}_edge_x.jpg".format(args.kernel.lower())))

    img_edge_y = detect_edges(img, kernel_y, False)
    img_edge_y = np.asarray(img_edge_y)
    write_image(normalize(img_edge_y), os.path.join(args.rs_directory, "{}_edge_y.jpg".format(args.kernel.lower())))

    img_edges = edge_magnitude(img_edge_x, img_edge_y)
    write_image(img_edges, os.path.join(args.rs_directory, "{}_edge_mag.jpg".format(args.kernel.lower())))


if __name__ == "__main__":
    %tb
    main()
