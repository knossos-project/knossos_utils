from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import numpy as np


def multi_dilation(overlay, n_dilations):
    if n_dilations == 0:
        return
    unique_ixs = np.unique(overlay)
    for ix in unique_ixs:
        if ix == 0:
            continue
        binary_mask = (overlay == ix).astype(np.int)
        binary_mask = binary_dilation(binary_mask, iterations=n_dilations)
        overlay[binary_mask == 1] = ix


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
        if len(background.shape) == 3:
            assert background.shape[2] == 1, "Only 2D images allowed."
            background = background[..., 0]
        comp = Image.fromarray(background, 'L')
    else:
       comp = create_prob_overlay_img(label_prob_dict, background, cvals=cvals)
    return comp


def create_prob_overlay_img(label_prob_dict, background, cvals=None):
    """

    :param label_prob_dict:
    :param background:
    :param cvals:
    :return:
    """
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
    for img in imgs[1:]:
        comp = alpha_composite(comp, img)

    comp = alpha_composite(comp, background)

    return comp


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