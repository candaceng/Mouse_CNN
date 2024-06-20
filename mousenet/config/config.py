DEBUG = False

# C, D, H, W
INPUT_SIZE = (3, 21, 53, 81)  # adjusted for visual subfields, h >= 53, w >= 81
OUTPUT_LGN_MODEL = (5, 51, 79)
# INPUT_SIZE = (3, 79, 51)

BATCH_SIZE = 2
KERNEL_SIZE = (3, 3, 3)


# LGN parameters:
PATH_PARAM_FILTER_LGN_CSV = "../mousenet/config/data_lgn/lgn_filters_mean.csv"
PATH_NUM_OF_NEURONS_PER_FILTER_LGN_YAML = (
    "../mousenet/config/data_lgn/num_neurons_per_filter_debug.yaml"
)

INPUT_CORNER = (0, 0)  # min azimuth, min elevation of input image
# INPUT_CORNER = (0, -50) # min azimuth, min elevation of input image

NUM_CLASSES = 1000
HIDDEN_LINEAR = 2048

EDGE_Z = 1  # Z-score (# standard deviations) of edge of kernel
INPUT_GSH = 1  # Gaussian height of input to LGNv
INPUT_GSW = 4  # Gaussian width of input to LGNv

# OUTPUT_AREAS = ['VISpor5']
OUTPUT_AREAS = ["VISp5", "VISl5", "VISrl5", "VISli5", "VISpl5", "VISal5", "VISpor5"]

SUBFIELDS = True  # use area-specific visual subfields


def get_out_sigma(source_area, source_depth, target_area, target_depth):
    source_resolution = get_resolution(source_area, source_depth)
    target_resolution = get_resolution(target_area, target_depth)
    return target_resolution / source_resolution


def get_resolution(area, depth):
    """
    :param area: cortical visual area name
    :param depth: layer name
    :return: model resolution in pixels per degree visual angle
    """
    return 1

    # if area == 'VISp' or area == 'LGNd' or area == 'input':
    #     return 1
    # else:
    #     return 0.5
