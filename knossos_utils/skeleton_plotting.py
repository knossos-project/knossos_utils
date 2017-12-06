################################################################################
#  This file provides a functions and classes for working with synapse annotations.
#  and writing raw and overlay data.
#
#  (C) Copyright 2017
#  Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.
#
#  skeleton_plotting.py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 of
#  the License as published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#
################################################################################


"""
Class and function definitions that allow the plotting of SkeletoAnnotation objects and Synapse objects.
"""

try:
    import mayavi.mlab as mlab
except:
    print("mayavi not installed")
import numpy as np
import matplotlib as mplt
import random


def add_spheres_to_mayavi_window(sphere_coords,
                                 radii,
                                 color = (0.0, 1.0, 0.0, 1.0),
                                 resolution = 20.,
                                 mode='sphere'):
    '''

    Adds spheres to the current mayavi window.

    '''


    ### DISCLAIMER: IT IS NOT CLEAR WHETHER THE DEFINED
    ### SPHERE SIZE IS A RADIUS OR DIAMETER OR SOME MYSTERIOUS "SIZE";
    ### The documentation is not conclusive

    try:
        _ = (e for e in radii)
    except TypeError:
        radii = np.ones(len(sphere_coords)/3) * radii

    coords = np.array(sphere_coords)
    sc = np.hsplit(coords, 3)

    try:
        x = [el[0] for el in sc[0].tolist()]
        y = [el[0] for el in sc[1].tolist()]
        z = [el[0] for el in sc[2].tolist()]
    except:
        x = [el for el in sc[0].tolist()]
        y = [el for el in sc[1].tolist()]
        z = [el for el in sc[2].tolist()]

    #raise()
    mlab.points3d(x, y, z, radii, color = color[0:3],
                  scale_factor = 1.0, resolution = 20, opacity = color[3],
                  mode=mode)
    return


def add_synapses_to_mayavi_window(syns,
                                  synapse_location = 'pre',
                                  color=(0.0, 1.0, 0.0, 1.0),
                                  diameter_scale = 1.,
                                  all_same_size = 0.):

    '''

    Adds human generated synapse objects to the current mayavi window.
    Synapses are rendered as spheres, with syn.AZlen used as diameter.

    Parameters
    ----------

    synapses : Iterable of Synapse instances
        E.g. a list of synapse lists, where every list contains the synapses
        extracted from the same annotation.


    synapse_location : str
        Either 'pre', 'post', 'pre_post_average' or 'az_average'.

    color : tuple
        rgba values between 0 and 1

    diameter_scale : float
        scaling factor to apply to the az_len (interpreted as diameter)
         to radius conversion; this can be helpful for small synapses


    '''


    if synapse_location == 'pre':
        coords = [syn.preNodeCoord_scaled for syn in syns]
    elif synapse_location == 'post':
        coords = [syn.postNodeCoord_scaled for syn in syns]
    elif synapse_location == 'pre_post_average':
        coords = [syn.avgPrePostCoord_scaled for syn in syns]
    elif synapse_location == 'az_average':
        coords = [syn.az_center_of_mass for syn in syns]
    else:
        raise Exception('Unsupported synapse_location given: '
                        + synapse_location)
    if not all_same_size > 0.:
        radii = [syn.az_len for syn in syns]
        radii = (np.array(radii) / 2.) * diameter_scale
    else:
        radii = [all_same_size] * len(syns)

    add_spheres_to_mayavi_window(coords, radii, color)



    return


def get_different_rgba_colors(num_colors,
                              rgb_only = False,
                              set_alpha = 1.0):

    """
    Parameters
    ----------

    num_colors : int
       Number of randomly picked colors to return (not necessarily unique)

    set_alpha : float
        alpha value to add to the rgb colors

        Returns

    list of random rgba colors


    """

    # create matplotlib color converter object
    cc_mplt = mplt.colors.ColorConverter()

    # this is a bit ugly, the matplotlib converter actually supports all web
    # colors, see eg http://www.w3schools.com/html/html_colornames.asp Here I
    #  just handpicked a few ones, would be better to make all of them
    # available programmatically, or to actually implement a proper
    # perception-based color selection module

    hand_picked_colors = ['LightPink',
                          'DeepPink',
                          'Crimson',
                          'DarkRed',
                          'OrangeRed',
                          'DarkOrange',
                          'Yellow',
                          'DarkKhaki',
                          'MediumTurquoise',
                          'MediumBlue',
                          'LightSkyBlue',
                          'Magenta',
                          'Thistle',
                          'DarkOliveGreen',
                          'MediumSpringGreen']


    colors = []
    for _ in xrange(num_colors):
        if rgb_only:
            colors.append(cc_mplt.to_rgb(random.choice(hand_picked_colors)))
        else:
            colors.append(cc_mplt.to_rgba(random.choice(hand_picked_colors)))

    return colors


def visualize_anno_with_synapses(anno, syns):

    """
    Visualizes an annotation together with synapses. If syns is an iterable
    of iterables of Synapse objects, the iterables will be given an
    inidividual color (useful to look at redundant synapse annotations on
    the same annotation)

        Parameters
    ----------

    anno : Iterable of Synapse instances
        E.g. a list of synapse lists, where every list contains the synapses
        extracted from the same annotation.


    syns : iterable or iterable of iterables of Synapse objects

    """

    visualize_annotation(anno)

    if is_iterable_of_iterables(syns):
        for same_syns in syns:
            add_synapses_to_mayavi_window(same_syns, color = ())
    else:
        add_synapses_to_mayavi_window(syns, color = (0.0, 0.0, 1.0, 0.5))


    return

def is_iterable_of_iterables(item):
    try:
        t = (e for e in item)
        tt = (e for e in t)
        return True
    except TypeError:
        return False


def add_anno_to_mayavi_window(anno,
                              node_scaling = 1.0,
                              override_node_radius = 500.,
                              edge_radius = 250.,
                              show_outline = False,
                              dataset_identifier='',
                              opacity=1):
    '''

    Adds an annotation to a currently open mayavi plotting window

    Parameters: anno: annotation object
    node_scaling: float, scaling factor for node radius
    edge_radius: float, radius of tubes for each edge

    '''

    # plot the nodes
    # x, y, z are numpy arrays, s as well

    if type(anno) == list:
        nodes = []
        for this_anno in anno:
            nodes.extend(this_anno.getNodes())
        color = anno[0].color
    else:
        nodes = list(anno.getNodes())
        color = anno.color

    coords = np.array([node.getCoordinate_scaled() for node in nodes])\
    #* node_scaling

    sc = np.hsplit(coords, 3)

    # separate x, y and z; mlab needs that
    #datasetDims = np.array(anno.datasetDims)


    x = [el[0] for el in sc[0].tolist()]
    y = [el[0] for el in sc[1].tolist()]
    z = [el[0] for el in sc[2].tolist()]

    if override_node_radius > 0.:
        s = [override_node_radius]*len(nodes)
    else:
        s = [node.getDataElem('radius') for node in nodes]

    s = np.array(s)
    s = s * node_scaling
    #s[0] = 5000
 #extent=[1, 108810, 1, 106250, 1, 115220]
    #raise
    pts = mlab.points3d(x, y, z, s, color=color, scale_factor=1.0,
                        opacity=opacity)

    # dict for faster lookup, nodes.index(node) adds O(n^2)
    nodeIndexMapping = {}
    for nodeIndex, node in enumerate(nodes):
        nodeIndexMapping[node] = nodeIndex

    edges = []
    for node in nodes:
        for child in node.getChildren():
            try:
                edges.append((nodeIndexMapping[node], nodeIndexMapping[child]))
            except:
                print('Phantom child node, annotation object inconsistent')


    # plot the edges
    pts.mlab_source.dataset.lines = np.array(edges)
    pts.mlab_source.update()
    tube = mlab.pipeline.tube(pts, tube_radius = edge_radius)
    mlab.pipeline.surface(tube, color = anno.color)

    if show_outline:
        if dataset_identifier == 'j0126':
            mlab.outline(extent=(0,108810,0,106250,0,115220), opacity=0.5,
                         line_width=5.)
        elif dataset_identifier == 'j0251':
            mlab.outline(extent=(0,270000,0,270000,0,387350), opacity=0.5,
                         line_width=5.)
        elif dataset_identifier == 'j0256':
            mlab.outline(extent=(0,166155,0,166155,0,77198), opacity=0.5,
                         line_width=1., color=(0.5,0.5,0.5))
        else:
            print('Please add a dataset identifier string')

    return

def visualize_annotation(anno,
                         node_scaling = 1.0,
                         override_node_radius = 500.,
                         edge_radius = 250.0,
                         bg_color = (1.0, 1.0, 1.0),
                         dataset_identifier='',
                         show_outline=True,
                         figure_size_px = (600, 600)):
    '''

    Creates a new mayavi window and adds the annotation to it.
    Make sure that the edge radius is half of the node radius to avoid ugly skeleton renderings.

    '''

    figure_size = figure_size_px

    mlab.figure(None, bg_color,
                fgcolor = (0.0, 0.0, 0.0),
                size=(600, 600))
    mlab.clf()

    #if type(anno) == list:
    #    for cur_anno in anno:
    #        add_anno_to_mayavi_window(cur_anno, node_scaling, edge_radius)
    #else:

    add_anno_to_mayavi_window(anno,
                              node_scaling=node_scaling,
                              override_node_radius=override_node_radius,
                              edge_radius=edge_radius,
                              show_outline=show_outline,
                              dataset_identifier=dataset_identifier)

    #mlab.view(49, 31.5, 52.8, (4.2, 37.3, 20.6))
    #mlab.xlabel('x')
    #mlab.ylabel('y')
    #mlab.zlabel('z')
