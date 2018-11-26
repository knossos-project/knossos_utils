from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import numpy as np
import itertools
import scipy
from numba import jit


def create_label_overlay_img(labels, save_path, background=None, cvals=None,
                             save_raw_img=True):
    """
    Needs recatoring. Super RAM and time intensive...
    """
    if cvals is None:
        cvals = {}
    else:
        assert isinstance(cvals, dict)

    np.random.seed(0)

    label_prob_dict = {}

    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        if unique_label == 0:
            continue
        label_prob_dict[unique_label] = (labels == unique_label).astype(np.int8)

        if not unique_label in cvals:
            cvals[unique_label] = [np.random.rand() for _ in range(3)] + [1]

    if len(label_prob_dict) == 0:
        print("No labels detected! No overlay image created")
    else:
        create_prob_overlay_img(label_prob_dict, save_path,
                                background=background, cvals=cvals,
                                save_raw_img=save_raw_img)


def create_prob_overlay_img(label_prob_dict, save_path, background=None,
                            cvals=None, save_raw_img=True):
    """
    Needs recatoring. Super RAM and time intensive... Combin with 'create_label_overlay_img'
    """
    assert isinstance(label_prob_dict, dict)
    if cvals is not None:
        assert isinstance(cvals, dict)

    np.random.seed(0)

    label_prob_dict_keys = label_prob_dict.keys()
    sh = label_prob_dict[label_prob_dict_keys[0]].shape[:2]
    comp = np.zeros([sh[0], sh[1], 4], dtype=np.float32)
    for key in label_prob_dict_keys:
        label_prob = np.array(label_prob_dict[key])

        label_prob = label_prob.squeeze()

        if key in cvals:
            cval = cvals[key]
        else:
            cval = [np.random.rand() for _ in range(3)] + [1]

        this_img = np.zeros([sh[0], sh[1], 4], dtype=np.float32)
        this_img[label_prob > 0] = np.array(cval) * 255
        this_img[:, :, 3] = label_prob * 100
        comp = alpha_composite(comp, this_img)
    if background is None:
        background = np.ones(comp.size)
        background[:, :, 3] = np.ones(sh)
    elif len(np.shape(background)) == 2:
        t_background = np.zeros(np.asarray(comp).shape)
        for ii in range(3):
            t_background[:, :, ii] = background
        t_background[:, :, 3] = np.ones(background.squeeze().shape) * 255
        background = t_background
    elif len(np.shape(background)) == 3:
        background = np.array(background)[:, :, 0]
        background = np.array([background, background, background,
                               np.ones_like(background) * 255])

    if np.max(background) <= 1:
        background *= 255.
    else:
        background = np.array(background, dtype=np.float)

    comp = alpha_composite(comp, background)

    if save_path is not None:
        scipy.misc.imsave(save_path, comp)

    if save_raw_img and background is not None:
        raw_save_path = "".join(save_path.split(".")[:-1]) + "_raw." + save_path.split(".")[-1]
        scipy.misc.imsave(raw_save_path, background)


def create_composite_img(labels, background, max_alpha_raw=0.8, max_alpha_ol=1.0, cvals=None):
    """

    :param labels:
    :param background:
    :param cvals:
    :return:
    """
    unique_labels = np.unique(labels)
    if cvals is None:
        cvals = {}
        np.random.seed(0)
        for unique_label in unique_labels:
            if unique_label == 0:
                cvals[unique_label] = np.array([0, 0, 0, 0], dtype=np.uint8)
            else:
                cvals[unique_label] = np.array([np.random.rand() * 255 for _ in range(3)]
                                               + [max_alpha_ol * 255], dtype=np.uint8)
    else:
        assert isinstance(cvals, dict)

    if len(unique_labels) == 0:
        print("No labels detected! No overlay image created")
        if len(background.shape) == 3:
            assert background.shape[2] == 1, "Only 2D images allowed."
            background = background[..., 0]
        comp = Image.fromarray(background, 'L')
    else:
       comp = create_overlay_img(labels, background, cvals=cvals, max_alpha_raw=max_alpha_raw)
    return comp


def create_overlay_img(labels, background, cvals, n_dils=0, max_alpha_raw=0.8):
    """
    :param label_prob_dict:
    :param background:
    :param cvals:
    :return:
    """
    assert isinstance(cvals, dict)
    if 0 <= np.max(background) <= 1.0 and not background.dtype.kind in ('u', 'i'):
        background = (background * 255).astype(np.uint8)
    labels = labels.squeeze()
    labels = multi_dilation(labels, n_dils)
    sh = labels.shape
    target_img = np.zeros([sh[0], sh[1], 4], dtype=np.uint8)
    # vectorized double for loop should perform better than single for-loop which contains fancy indexing
    #  of the whole array for each label (if number of labels is high)
    # import tqdm
    # pbar = tqdm.tqdm(total=np.prod(labels.shape[:2]), mininterval=5)
    # for i, j in itertools.product(np.arange(labels.shape[0]),
    #                               np.arange(labels.shape[1])):
    #     target_img[i, j] = cvals[labels[i, j]]
    #     pbar.update(1)
    target_img = label_mapping(labels, target_img, cvals)
    if background is not None:
        if len(np.shape(background)) == 2:
            background = background[..., None]
        background = np.concatenate([background, background, background,
                                     np.ones_like(background) * 255 * max_alpha_raw], axis=2)
        target_img = alpha_composite(target_img, background)
    else:
        target_img = Image.fromarray(target_img, 'RGBA')
    return target_img


@jit
def label_mapping(labels, target_img, cvals):
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            target_img[i, j] = cvals[labels[i, j]]
    return target_img


def alpha_composite(src, dst):
    ''' http://stackoverflow.com/questions/3374878/with-the-python-imaging-library-pil-how-does-one-compose-an-image-with-an-alp/3375291#3375291
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    src = np.asarray(src)
    dst = np.asarray(dst)
    out = np.empty(src.shape, dtype = 'float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)
    out[alpha] *= 255
    np.clip(out, 0, 255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype('uint8')
    out = Image.fromarray(out, 'RGBA')
    return out


def multi_dilation(overlay, n_dilations):
    if n_dilations == 0:
        return overlay
    unique_ixs = np.unique(overlay)
    for ix in unique_ixs:
        if ix == 0:
            continue
        binary_mask = (overlay == ix).astype(np.int)
        binary_mask = binary_dilation(binary_mask, iterations=n_dilations)
        overlay[binary_mask == 1] = ix
    return overlay