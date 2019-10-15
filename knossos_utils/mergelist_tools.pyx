from __future__ import absolute_import, division, print_function
# builtins is either provided by Python 3 or by the "future" module for Python 2 (http://python-future.org/)
from builtins import range, map, zip, filter, round, next, input, bytes, hex, oct, chr, int
from functools import reduce

cimport cython
import networkx as nx
import numpy as np
cimport numpy as np
from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

seg_dtype = cython.fused_type(np.uint8_t, np.uint16_t, np.uint32_t, np.uint64_t)


@cython.boundscheck(False)
def objects_from_mergelist(mergelist_content):
    """
    Reads mergelist into a vector of objects (= sets of subobject ids)
    :param mergelist_content:
        the mergelist content as a string
    """
    cdef vector[unordered_set[np.uint64_t]] obj_list
    for line in mergelist_content.split("\n")[0::4]:
        elems = line.split()
        if len(elems) > 0:
            obj_list.push_back(<unordered_set[np.uint64_t]>[np.uint64(elem) for elem in elems[3:]])
    return obj_list


@cython.boundscheck(False)
def subobject_map_from_mergelist(mergelist_content):
    """
    Extracts a single object layer from a mergelist and returns a map of subobject ID > object ID.
    If one subobject is contained in more than one object, the last object is kept.
    :param mergelist_content:
        the mergelist content as a string
    """
    cdef unordered_map[np.uint64_t, np.uint64_t] subobjects_to_objects_map
    for line in mergelist_content.split("\n")[0::4]:
        elems = [np.uint64(elem) for elem in line.split()]
        if len(elems) > 0:
            object_id = elems[0]
            for subobject_id in elems[3:]:
                subobjects_to_objects_map[subobject_id] = object_id
    return subobjects_to_objects_map


@cython.boundscheck(False)
def apply_mergelist(seg_dtype[:,:,:] segmentation, mergelist_content, seg_dtype background_id=0, seg_dtype pad=0, bool missing_subobjects_to_background=False):
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
    cdef unordered_map[seg_dtype, seg_dtype] subobject_map = subobject_map_from_mergelist(mergelist_content)
    cdef int width = segmentation.shape[2]
    cdef int height = segmentation.shape[1]
    cdef int depth = segmentation.shape[0]
    cdef Py_ssize_t x, y, z
    cdef seg_dtype subobject_id
    cdef seg_dtype object_id
    cdef seg_dtype new_subobject_id

    cdef unordered_map[seg_dtype, seg_dtype] object_map
    for it in subobject_map:
        if it.first != background_id:
            object_map[it.second] = min(it.first, object_map[it.second])

    for z in range(pad, depth - pad):
        for y in range(pad, height - pad):
            for x in range(pad, width - pad):
                subobject_id = segmentation[z, y, x]
                if subobject_map.find(subobject_id) == subobject_map.end():
                    if missing_subobjects_to_background and not subobject_id == background_id:
                        segmentation[z, y, x] = background_id
                    continue

                object_id = subobject_map[subobject_id]
                object_map_it = object_map.find(object_id)
                new_subobject_id =  dereference(object_map_it).second

                segmentation[z, y, x] = new_subobject_id

    return segmentation


@cython.boundscheck(False)
def gen_mergelist_from_segmentation(seg_dtype[:,:,:] segmentation, seg_dtype background_id=0, seg_dtype pad=0, np.ndarray[np.uint64_t, ndim=1] offsets=np.array([0, 0, 0])):
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
    cdef int width = segmentation.shape[2]
    cdef int height = segmentation.shape[1]
    cdef int depth = segmentation.shape[0]
    cdef Py_ssize_t x, y, z
    cdef seg_dtype next_id
    cdef seg_dtype so_cache = background_id

    cdef unordered_set[seg_dtype] seen_subobjects
    new_mergelist = ""
    for z in range(pad, depth - pad):
        for y in range(pad, height - pad):
            for x in range(pad, width - pad):
                next_id = segmentation[z, y, x]
                if next_id == background_id or next_id == so_cache or seen_subobjects.find(next_id) != seen_subobjects.end():
                    continue
                so_cache = next_id
                seen_subobjects.insert(next_id)
                new_mergelist += "{0} 0 0 {0}\n{1} {2} {3}\n\n\n".format(next_id, offsets[0]+x, offsets[1]+y, offsets[2]+z)
    return new_mergelist


@cython.boundscheck(False)
def gen_mergelist_from_objects(unordered_map[seg_dtype, pair[unordered_set[seg_dtype], vector[seg_dtype]]] objects):
    new_mergelist = ""
    for obj in objects:
        sub_obj_str = ""
        for subobj_id in obj.second.first:
            sub_obj_str += "{} ".format(subobj_id)
        new_mergelist += "{} 0 0 {}\n".format(obj.first, sub_obj_str[:-1]) # remove trailing white space
        coord = obj.second.second
        new_mergelist += "{} {} {}\n\n\n".format(coord[0], coord[1], coord[2])
    return new_mergelist


def merge_objects(vector[unordered_set[seg_dtype]] objects):
    G = nx.Graph()
    for obj in objects:
        first = dereference(obj.begin())
        obj.erase(obj.begin())
        G.add_node(first)
        while not obj.empty():
            second = dereference(obj.begin())
            obj.erase(obj.begin())
            G.add_edge(first, second)
    return nx.connected_components(G)
