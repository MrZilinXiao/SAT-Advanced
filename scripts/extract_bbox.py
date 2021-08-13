from scipy import ndimage
import cv2
import numpy as np
import argparse


def get_bbox_id(instance_img: np.ndarray):
    """

    :param instance_img: should be read by cv2.imread with `-1` option
    :return:
    """
    assert len(instance_img.shape) == 2
    bbox_list, instance_id_list = [], []
    for instance_id in np.unique(instance_img):
        # need exclude instance_id == 0?
        # Instance images: 8-bit .png where each pixel stores an integer value
        # per distinct instance (0 corresponds to unannotated or no depth).
        if instance_id == 0:
            continue

        s = ndimage.generate_binary_structure(2, 2)  ## 8-way connect, instead of the default 4-way
        labeled_array, numpatches = ndimage.label(np.array(instance_img == instance_id, dtype=int),
                                                  s)  ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        largest_cc_mask_xy = np.where(labeled_array == (np.bincount(labeled_array.flat)[1:].argmax() + 1))
        x_min, x_max = largest_cc_mask_xy[1].min(), largest_cc_mask_xy[1].max()
        y_min, y_max = largest_cc_mask_xy[0].min(), largest_cc_mask_xy[0].max()
        bbox_list.append([x_min, y_min, x_max, y_max])
        instance_id_list.append(instance_id)  # directly save instance_id? yes: instance_id = obj_id + 1

    return bbox_list, instance_id_list


if __name__ == '__main__':
    pass