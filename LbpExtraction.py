import cv2
from skimage import feature


def calc_lbp(img):
    lbp = feature.local_binary_pattern(img, 8, 1, method='default')
    lbp = lbp.astype('uint8')
    # lbp = feature.local_binary_pattern(hsv[:, :, 1], 8,
    #                                    1, method="default")
    # lbp = cv2.resize(lbp, (128, 128))
    return lbp