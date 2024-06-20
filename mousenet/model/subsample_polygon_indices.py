import re

import numpy as np

from mousenet.config.config import INPUT_CORNER, INPUT_SIZE, get_resolution
from mousenet.model.data.Data import Data


def get_subsample_indices(layer):
    data = Data()

    source_area = re.split("[^[a-zA-Z]]*", layer.source_name)[0]
    source_depth = layer.source_name[len(source_area) :]
    source_resolution = get_resolution(source_area, source_depth)
    target_area = re.split("[^[a-zA-Z]]*", layer.target_name)[0]

    # calculate visual field corners of input assuming input image is one pixel per degree
    image_height = INPUT_SIZE[3]
    image_width = INPUT_SIZE[2]  # TODO: height should be [2] and width [3]

    input_field = [
        INPUT_CORNER[0],
        INPUT_CORNER[1],
        INPUT_CORNER[0] + image_width,
        INPUT_CORNER[1] + image_height,
    ]

    if source_area == "input" or source_area == "LGNd":
        source_field = input_field
    else:
        source_field = data.get_visual_field_shape(source_area)

    if target_area == "LGNd":
        target_field = input_field
    else:
        target_field = data.get_visual_field_shape(target_area)

    # indices in source frame of reference at image resolution
    left = target_field[0] - source_field[0]
    width = target_field[2] - target_field[0]
    bottom = target_field[1] - source_field[1]
    height = target_field[3] - target_field[1]
    left, width, bottom, height = [
        int(np.floor(source_resolution * x)) for x in [left, width, bottom, height]
    ]
    if source_area != "input":
        # if True:
        source_right = int(source_field[2] - source_field[0])
        source_top = int(source_field[3] - source_field[1])
        # WIP: moddify subfield capturing from source layer to reduce padding
        kernel = layer.params.kernel_size
        # w_out = (w_in +2*padding[0]+ (kernel-1)-1)/stride + 1 -> padding=0
        w_extra_cov = kernel - 1
        left_extra, right_extra = w_extra_cov // 2, w_extra_cov // 2
        right = left + width
        if np.mod(w_extra_cov, 2) != 0:
            right_extra += 1
        padding_left = left_extra - left if left - left_extra < 0 else 0
        padding_right = (
            right + right_extra - source_right
            if right + right_extra >= source_right
            else 0
        )
        new_left = max(0, left - left_extra)
        new_right = min(right + right_extra, source_right)
        new_width = new_right - new_left

        # h_out = (h_in +2*padding[1]+ (kernel-1)-1)/stride + 1 -> padding=0
        h_extra_cov = kernel - 1
        bottom_extra, top_extra = h_extra_cov // 2, h_extra_cov // 2
        top = bottom + height
        if np.mod(h_extra_cov, 2) != 0:
            top_extra += 1
        padding_bottom = bottom_extra - bottom if bottom - bottom_extra < 0 else 0
        padding_top = (
            top + top_extra - source_top if top + top_extra >= source_top else 0
        )
        new_bottom = max(0, bottom - bottom_extra)
        new_top = min(top + top_extra, source_top)
        new_height = new_top - new_bottom
        left, width, bottom, height = new_left, new_width, new_bottom, new_height
        # given that input format for padding is: (N,C,H,W)
        # It takes padding of order: padding_left, padding_right, padding_top, padding_bottom
        # Formula is
        # H_out = H_in + padding_top + padding_bottom
        # W_out = W_in + padding_left + padding_right
        layer.params.padding = [
            int(x) for x in [padding_left, padding_right, padding_bottom, padding_top]
        ]

    # indices in source frame of reference at source resolution
    left, width, bottom, height = [
        int(np.floor(source_resolution * x)) for x in [left, width, bottom, height]
    ]
    return left, width, bottom, height
