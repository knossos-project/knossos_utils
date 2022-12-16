from __future__ import absolute_import, division, print_function
# builtins is either provided by Python 3 or by the "future" module for Python 2 (http://python-future.org/)
from builtins import range, int
import warnings
import numpy as np


def subobject_map_from_mergelist(mergelist_content):
    """
    Extracts a single object layer from a mergelist and returns a map of subobject ID > object ID.
    If one subobject is contained in more than one object, the last object is kept.
    :param mergelist_content:
        the mergelist content as a string
    """
    subobjects_map = {}
    for line in mergelist_content.split("\n")[0::4]:
        elems = line.split()
        if len(elems) > 0:
            object_id = elems[0]
            for subobject_id in elems[3:]:
                if (len(elems) > 4):
                    subobjects_map[np.uint64(subobject_id)] = np.uint64(object_id)
                elif not np.uint64(subobject_id) in subobjects_map:
                    subobjects_map[np.uint64(subobject_id)] = np.uint64(object_id)
    return subobjects_map


def apply_mergelist(segmentation, mergelist_content, background_id=0, pad=0, missing_subobjects_to_background=False):
    """
    Merges subobjects using a dictionary of (subobject, object) pairs. So each subobject can only be member of one object.
    The resulting segmentation for each merged group contains only the first ID of that group
    :param segmentation:
        3D array containing the subobject IDs
    :param mergelist_content:
        the mergelist content as a string
    :param pad:
        optional padding that is excluded from mergelist application
    """
    subobject_map = subobject_map_from_mergelist(mergelist_content)
    width = segmentation.shape[0]
    height = segmentation.shape[1]
    depth = segmentation.shape[2]

    object_map = {}

    for z in range(pad, depth - pad):
        for y in range(pad, height - pad):
            for x in range(pad, width - pad):
                subobject_id = segmentation[x, y, z]
                if subobject_id == background_id:
                    continue
                # ghost ID's in overlay cubes
                try:
                    object_id = subobject_map[subobject_id]
                except KeyError:
                    warnings.warn("Found label (%d) in overlay which is not "
                                  "contained in mergelist." % subobject_id,
                                  RuntimeWarning)
                    segmentation[x, y, z] = background_id
                    continue
                if object_id == background_id and missing_subobjects_to_background:
                    segmentation[x, y, z] = background_id
                    continue
                new_subobject_id = subobject_id
                if object_id in object_map.keys():
                    new_subobject_id = object_map[object_id]
                else:
                    object_map[object_id] = subobject_id

                segmentation[x, y, z] = new_subobject_id

    return segmentation


def gen_mergelist_from_segmentation(segmentation, background_id=0, pad=0, offsets=np.array([0, 0, 0], dtype=np.uint64)):
    """
    Generates a mergelist from a segmentation in which each subobject is contained in its own object.
    The object's coordinate is the first coordinate of the subobject.
    :param segmentation:
        3D array containing the subobject IDs
    :background_id:
        The background id will be skipped
    :pad:
        optional padding that is excluded from mergelist generation
    :offsets:
        the voxel coordinate closest to 0, 0, 0 of the whole dataset, used to give objects their correct coordinate
    """
    width = segmentation.shape[0]
    height = segmentation.shape[1]
    depth = segmentation.shape[2]
    so_cache = background_id

    seen_subobjects = set()
    new_mergelist = ""
    for z in range(pad, depth - pad):
        for y in range(pad, height - pad):
            for x in range(pad, width - pad):
                next_id = segmentation[x, y, z]
                if next_id == so_cache or next_id == background_id or next_id in seen_subobjects:
                    continue
                so_cache = next_id
                seen_subobjects.add(next_id)
                new_mergelist += "{0} 0 0 {0}\n{1} {2} {3}\n\n\n".format(next_id, int(offsets[0]+x), int(offsets[1]+y), int(offsets[2]+z))
    return new_mergelist

def gen_mergelist_from_objects(objects):
    new_mergelist = ""
    for obj, (subobjects, coords) in objects.items():
        sub_obj_str = ""
        for subobj_id in subobjects:
            sub_obj_str += "{} ".format(subobj_id)
        new_mergelist += "{} 0 0 {}\n".format(obj.first, sub_obj_str[:-1]) # remove trailing white space
        new_mergelist += "{} {} {}\n\n\n".format(coords[0], coords[1], coords[2])
    return new_mergelist


def parse_mergelist(mergelist: str, return_todo=False, return_immutable=False, return_supervoxel_ids=False, return_position=False, return_color=False, return_category=False, return_comment=False):
    import re
    lines = mergelist.split('\n')
    idx = 0
    object_ids, todos, immutables, supervoxel_ids, positions, colors, categories, comments = [[] for _ in range(8)]
    while(idx < len(lines)-1):
        try:
            line_split = lines[idx].split(' ')
            obj_id, todo, immutable = (int(val) for val in line_split[:3])
            object_ids.append(obj_id)
            if return_todo: todos.append(todo)
            if return_immutable: immutables.append(immutable)
            if return_supervoxel_ids: supervoxel_ids.append([int(val) for val in line_split[3:]])
            idx += 1;
            hits = re.search(r'(-?\d+) (-?\d+) (-?\d+) ((\d+) (\d+) (\d+))?', lines[idx]).groups()
            if return_position: positions.append(tuple(int(val) for val in hits[:3]))
            if return_color: colors.append(None if hits[3] is None else tuple(int(val) for val in hits[4:]))
            idx += 1;
            if return_category: categories.append(lines[idx])
            idx += 1;
            if return_comment: comments.append(lines[idx])
            idx += 1;
        except (IndexError, ValueError) as e:
            raise Exception(f'Parsing mergelist failed at line {idx+1}: {lines[idx]}\nerror: {e}')
    return object_ids, todos, immutables, supervoxel_ids, positions, colors, categories, comments
