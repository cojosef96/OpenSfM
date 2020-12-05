from opensfm import config
from opensfm.features import extract_features, build_flann_index, denormalized_image_coordinates
from opensfm.matching import match_flann, match_flann_symmetric
import cv2
import numpy as np
import matplotlib.pyplot as plt

CONFIG_FILE = "data/berlin/config.yaml"
IMAGE_1 = 'data/berlin/images/day.png'
IMAGE_2 = 'data/berlin/images/night.png'
FEATURE_EXTRACTORS = ['AKAZE', 'SIFT', 'HAHOG', 'ORB']


def load_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2


def transform_keypoints_to_cv_points(p1, p2):
    # transform into cv2 keypoints
    cv_kp1 = [cv2.KeyPoint(x=p1[i, 0], y=p1[i, 1], _size=20) for i in range(len(p1))]
    cv_kp2 = [cv2.KeyPoint(x=p2[i, 0], y=p2[i, 1], _size=20) for i in range(len(p2))]
    return cv_kp1, cv_kp2


def draw_matches(img1, kp1, img2, kp2, matches, color=None, feat="SIFT"):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
            c = (int(c[0]), int(c[1]), int(c[2]))
            c = tuple(c)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m[0]].pt).astype(int))
        end2 = tuple(np.round(kp2[m[1]].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

    plt.figure(figsize=(9, 4))
    plt.imshow(new_img)
    plt.savefig("fig_{}.png".format(feat))


def plot_features(img2, cv_kp2, feat='sift'):
    img3 = np.array([])
    img3 = cv2.drawKeypoints(img2, cv_kp2, img3, color=(0, 0, 255))
    plt.imshow(img3)
    plt.savefig("image_{}.png".format(feat))


def denormalize_points(p1, p2, shape1, shape2):
    p1 = denormalized_image_coordinates(p1, shape1[1], shape1[0])
    p2 = denormalized_image_coordinates(p2, shape2[1], shape2[0])
    return p1, p2


if __name__ == '__main__':
    config_sfm = config.load_config(CONFIG_FILE)
    symmetric_matching = config_sfm['symmetric_matching']
    img1, img2 = load_images(IMAGE_1, IMAGE_2)
    shape1 = img1.shape
    shape2 = img2.shape
    for feat in FEATURE_EXTRACTORS:
        print("start testing feature extraction and matching with {}".format(feat))
        config_sfm["feature_type"] = feat
        p1, f1, c1 = extract_features(img1, config_sfm)
        p2, f2, c2 = extract_features(img2, config_sfm)
        print("size of keypoints_1 = %d" % len(p1))
        print("size of keypoints_2 = %d" % len(p2))
        i1 = build_flann_index(f1, config_sfm)
        i2 = build_flann_index(f2, config_sfm)
        if symmetric_matching:
            matches = match_flann_symmetric(f1, i1, f2, i2, config_sfm)
        else:
            matches = match_flann(i1, f2, config_sfm)
        matches = np.array(matches, dtype=int)
        print("size of matches = %d" % len(matches))
        p1, p2 = denormalize_points(p1, p2, shape1, shape2)
        cv_kp1, cv_kp2 = transform_keypoints_to_cv_points(p1, p2)
        # draw the N first matches
        N = 50
        draw_matches(img1, cv_kp1, img2, cv_kp2, matches[:N], feat=feat)
        plot_features(img2, cv_kp2, feat=feat)
        print("\n")
