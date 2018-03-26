from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import numpy as np
import itertools
import scipy


def create_label_overlay_img(labels, save_path, background=None, cvals=None,
                             save_raw_img=True):
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
        label_prob_dict[unique_label] = (labels == unique_label).astype(np.int)

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
    assert isinstance(label_prob_dict, dict)
    if cvals is not None:
        assert isinstance(cvals, dict)

    np.random.seed(0)

    label_prob_dict_keys = label_prob_dict.keys()
    sh = label_prob_dict[label_prob_dict_keys[0]].shape[:2]
    imgs = []

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
        imgs.append(this_img)
    if background is None:
        background = np.ones(imgs[0].shape)
        background[:, :, 3] = np.ones(sh)
    elif len(np.shape(background)) == 2:
        t_background = np.zeros(imgs[0].shape)
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

    comp = imgs[0]
    # cnt = 0
    for img in imgs[1:]:
        # cnt += 1
        comp = alpha_composite(comp, img)
        # scipy.misc.imsave(save_path + "%d.tif" % cnt, comp)

    comp = alpha_composite(comp, background)

    if save_path is not None:
        scipy.misc.imsave(save_path, comp)

    if save_raw_img and background is not None:
        raw_save_path = "".join(save_path.split(".")[:-1]) + "_raw." + save_path.split(".")[-1]
        scipy.misc.imsave(raw_save_path, background)


def create_composite_img(labels, background, cvals=None):
    """

    :param labels:
    :param background:
    :param cvals:
    :return:
    """
    if cvals is None:
        cvals = {}
    else:
        assert isinstance(cvals, dict)

    np.random.seed(0)
    max_alpha = 100./255
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        if unique_label == 0:
            continue
        if not unique_label in cvals:
            cvals[unique_label] = [np.random.rand() for _ in range(3)] + [max_alpha]

    if len(unique_labels) == 0:
        print("No labels detected! No overlay image created")
        if len(background.shape) == 3:
            assert background.shape[2] == 1, "Only 2D images allowed."
            background = background[..., 0]
        comp = Image.fromarray(background, 'L')
    else:
       comp = create_overlay_img(labels, background, cvals=cvals)
    return comp


def create_overlay_img(labels, background, cvals=None):
    """

    :param label_prob_dict:
    :param background:
    :param cvals:
    :return:
    """
    if cvals is not None:
        assert isinstance(cvals, dict)
    else:
        cvals = {}
        np.random.seed(0)
    labels = labels.squeeze()
    sh = labels.shape
    target_img = np.zeros([sh[0], sh[1], 4], dtype=np.uint8)
    # vectorized double for loop should perform better than single for-loop which contains fancy indexing
    #  of the whole array

    for i, j in itertools.product(np.arange(labels.shape[0]),
                                  np.arange(labels.shape[1])):
        l = labels[i, j]
        target_img[i, j] = np.array(np.array(cvals[l]) * 255, dtype=np.uint8)
    if background is not None:
        if len(np.shape(background)) == 2:
            background = background[..., None]
        background = np.concatenate([background, background, background, np.ones_like(background) * 255], axis=2)
        target_img = alpha_composite(target_img, background)
    else:
        target_img = Image.fromarray(target_img, 'RGBA')
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
    np.clip(out,0,255)
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