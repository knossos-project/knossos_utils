"""
Class and function definitions that allow the extraction and
analysis of synapse annotations on skeleton annotations.
"""

import re as re
import numpy as np
import networkx as nx
import itertools as itools
from skeleton import Skeleton, SkeletonAnnotation
from skeleton_utils import split_by_connected_component,\
    annotation_from_nodes, get_reachable_nodes, merge_annotations,\
    get_nodes_with_comment, get_largest_annotation
from skeleton_utils import EnhancedAnnotation, KDtree
from skeleton_utils import average_coordinate, Coordinate
from skeleton_utils import gen_random_pass
from random import randint
import skeleton_utils as au
import matplotlib.pyplot as mplot
import math
import copy
import sys
try:
    import Levenshtein
    levenshtein_available = True
except:
    levenshtein_available = False

class AnnotationFormatException(Exception):
    pass

class HeuristicMatchingException(Exception):
    pass


def norm(vector):
    """ Returns the norm (length) of the vector."""
    return np.sqrt(np.dot(vector, vector))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.

    Example
    -------

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.1415926535897931
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if math.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle


class ConsolidatedSynapse(object):
    """
    Represents a set of Synapse objects that were lumped into a consensus
    synapse, i.e. that represent the same physical synapse.
    """

    @property
    def active_zone_average_length(self):
        az_lens = [x.az_len for x in self.synapse_annotations]
        return sum(az_lens) / len(az_lens)

    @property
    def source_annotations(self):
        return [x.source_annotation for x in self.synapse_annotations]

    @property
    def redundancy_level(self):
        return len(set(self.source_annotations))

    @property
    def pre_node_center_of_mass(self):
        try:
            return self._pre_node_center_of_mass
        except AttributeError:
            pre_node_coords = []
            for cur_s in self.synapse_annotations:
                pre_node_coords.append(cur_s.preNodeCoord)
            self._pre_node_center_of_mass = average_coordinate(
                pre_node_coords)
            return self._pre_node_center_of_mass

    @property
    def pre_node_center_of_mass_scaled(self):
        try:
            return self._pre_node_center_of_mass_scaled
        except AttributeError:
            pre_node_coords = []
            for cur_s in self.synapse_annotations:
                pre_node_coords.append(cur_s.preNodeCoord_scaled)
            self._pre_node_center_of_mass_scaled = average_coordinate(
                pre_node_coords)
            return self._pre_node_center_of_mass_scaled

    def __init__(self, source_synapses=None, collection_id=None):
        self.synapse_annotations = set()

        if source_synapses is not None:
            for cur_s in source_synapses:
                self.synapse_annotations.add(cur_s)
                cur_s.consolidated_synapse = self

        self.collection_id = collection_id

    def __len__(self):
        return len(self.synapse_annotations)


class Synapse(object):
    """
    Represents a synapse annotation by one annotator.
    """

    @property
    def source_annotation(self):
        try:
            return self._source_annotation
        except AttributeError:
            return self.sourceAnnoFile

    @source_annotation.setter
    def source_annotation(self, val):
        self._source_annotation = val

    @property
    def preNodeCoord(self):
        try:
            return self._preNodeCoord
        except AttributeError:
            self._preNodeCoord = self.preNode.getCoordinate()
            return self._preNodeCoord

    @preNodeCoord.setter
    def preNodeCoord(self, val):
        # For backwards compatibility, but coordinates should always be
        # taken from self.preNode anyway
        pass

    @property
    def postNodeCoord(self):
        try:
            return self._postNodeCoord
        except AttributeError:
            self._postNodeCoord = self.postNode.getCoordinate()
            return self._postNodeCoord

    @postNodeCoord.setter
    def postNodeCoord(self, val):
        pass

    @property
    def preNodeCoord_scaled(self):
        try:
            return self._preNodeCoord_scaled
        except AttributeError:
            self._preNodeCoord_scaled = self.preNode.getCoordinate_scaled()
            return self._preNodeCoord_scaled

    @property
    def postNodeCoord_scaled(self):
        try:
            return self._postNodeCoord_scaled
        except AttributeError:
            self._postNodeCoord_scaled = self.postNode.getCoordinate_scaled()
            return self._postNodeCoord_scaled

    @property
    def avgPrePostCoord(self):
        return average_coordinate([self.preNodeCoord, self.postNodeCoord])

    @property
    def avgPrePostCoord_scaled(self):
        return average_coordinate([self.preNodeCoord_scaled,
                                   self.postNodeCoord_scaled])

    @property
    def AZlen(self):
        return self.az_len

    @property
    def az_anno(self):
        try:
            return self._az_anno
        except AttributeError:
            if len(self.AZnodes):
                self._az_anno = annotation_from_nodes(self.AZnodes,
                    self.AZnodes[0].annotation, connect=True)
                return self._az_anno
            else:
                return None
    @property
    def az_len(self):
        try:
            return self._az_len
        except AttributeError:
            if self.az_anno is not None:
                self._az_len = self.az_anno.physical_length()
                return self._az_len
            else:
                return None

    @property
    def AZcenterOfMass(self):
        return self.az_center_of_mass

    @property
    def az_center_of_mass(self):
        try:
            return self._az_center_of_mass
        except AttributeError:
            if len(self.AZnodes):
                self._az_center_of_mass = [int(y) for y in average_coordinate(
                    [x.getCoordinate() for x in self.az_anno.getNodes()])]
                return self._az_center_of_mass
            else:
                return None

    @property
    def az_center_of_mass_scaled(self):
        try:
            return self._az_center_of_mass_scaled
        except AttributeError:
            if len(self.AZnodes):
                self._az_center_of_mass_scaled = [int(y) for y in
                                              average_coordinate(
                    [x.getCoordinate_scaled() for x in self.az_anno.getNodes()])]
                return self._az_center_of_mass_scaled
            else:
                return None

    def __init__(self, pre_node=None, post_node=None, az_nodes=None,
            temp_id=None, source_anno=None):
        self.consolidated_synapse = None

        self.location_tags = []
        self.type_tags = []
        self.no_tags = []

        self.tags = set()

        self.type_vote = None
        self.loc_vote = None
        self.no_vote = None

        self.preAnnoSeedID = None
        self.postAnnoSeedID = None

        self.sourceAnnoFile = ''
        self.sourceUsername = ''
        self.rawSourceComments = []

        self.preNode = pre_node
        self.postNode = post_node

        if source_anno is not None:
            self._source_annotation = source_anno

        if az_nodes is None:
            self.AZnodes = [] # connectivity defined over node children
        else:
            self.AZnodes = list(az_nodes)

        self.parentSpine = None

        self.collection_id = None

        self.ID = None

        # contains as / sy / dus, f / m / g / vus
        self.combinedClassificationString = ''
        self.vesicles = ''
        self.density = ''
        self.postPos = '' # either p0,1,2,3 or x for not yet known

        # Temporary ID assigned during annotation
        self.temp_id = temp_id

        self.type = None

        return

    def hasPrePost(self):
        npPostC = np.array(self.postNodeCoord)
        npPreC = np.array(self.preNodeCoord)
        if npPostC.sum() and npPreC.sum():
            return True
        else:
            print 'Warning: Pre - or post coord not properly set!'
            return False

    def euclDistToOtherSyn(self, otherSyn):
        return np.linalg.norm(np.array(self.getAvgPrePostCoord()) -
                              np.array(otherSyn.getAvgPrePostCoord()))

    def angleToOtherSyn(self, otherSyn):
        npPostC = np.array(self.postNodeCoord)
        npPreC = np.array(self.preNodeCoord)

        npOtherPostC = np.array(otherSyn.postNodeCoord)
        npOtherPreC = np.array(otherSyn.preNodeCoord)

        angle = math.degrees(angle_between((npPostC - npPreC),
            (npOtherPostC - npOtherPreC)))
        return angle

    def getAvgPrePostCoord(self):
        return self.avgPrePostCoord

    def __str__(self):
        return 'Synapse %s --%s--> %s, type %s, vesicles %s' % (
            str(self.preNodeCoord), str(self.az_center_of_mass),
            str(self.postNodeCoord), str(self.type), str(self.vesicles),)


class Spine(object):
    def __init__(self):
        self.annoSeedID = None
        # set of synapse objects that are attached to this spine
        self.synapses = set()

        # lists of nodes that were tagged as p1,p2,p3
        self.p1 = []
        self.p2 = []
        self.p3 = []

        self.redundantSpines = []

        return



def getAZLens(annotations):
    # find all AZ annotations
    AZannos = au.getAnnosByCommentRegex(r's\d+[abcdef]?[_-]?', annotations)
    AZannos = au.setAnnotationStats(AZannos)

    AZmatcher = re.compile(r'(?P<synID>s\d+[abcde]?)')
    AZs = {}
    # put lengths into a dictionary with synID key
    for AZanno in AZannos:
        mobj = AZmatcher.search(AZanno.getComment())
        if mobj:
            AZs[mobj.group('synID').lower()] = AZanno.pathLen
    return AZs


def get_matching_az(node, probable_az_kd=None, search_radius=1000.):
    """
    Return active zone annotation for node.

    First, assume that there are separate connected components in the same
    tree, with no other components present in the tree. One represents the
    single node synapse annotation, the other the active zone.

    If that fails, try to m   atch by proximity to a putative az passed as as
    KDtree structure.

    Also return the synapse ID that may have been added by the annotator.

    Parameters
    ----------

    node : SkeletonNode instance

    probable_az_kd : KDtree instance that contains SkeletonAnnotation
        instances representing active zones, indexed by the location of their
        center of gravity (average coordinate).

    search_radius : float
        Radius in which to match active zone by proximity

    Returns
    -------

    active_zone : SkeletonAnnotation instance
        A new SkeletonAnnotation consisting of the nodes making up the active
        zone connected component.

    extracted_id : str
        String representing a manually annotated synapse ID.
    """

    active_zone_numbered_ex = r'az\s(?P<syn_id>\d+)'
    extracted_id = ''

    az_and_synapse = list(split_by_connected_component(node.annotation))

    if len(az_and_synapse) == 2:
        # This should represent a synapse plus active zone annotation.
        # Check the assumption that one component is the single synapse
        # node and the other has several nodes for active zone annotation.
        if not 1 in [len(x) for x in az_and_synapse] or \
                not True in [len(x) > 1 for x in az_and_synapse]:
            raise AnnotationFormatException('Invalid synapse and active '
                'zone lumping at %s' % (node.getCoordinate(),))
        active_zone = az_and_synapse[[len(x) > 1 for x in az_and_synapse].
            index(True)]
    elif len(az_and_synapse) == 1:
        # Attempt to fall back to matching by proximity
        if not probable_az_kd is None:
            matched_az = probable_az_kd.query_ball_point(
                node.getCoordinate_scaled(), search_radius)
            if len(matched_az) == 1:
                active_zone = matched_az[0]
            elif len(matched_az) > 1:
                raise AnnotationFormatException('No matching by tree '
                    'possible and proximity fallback ambiguous for synapse at '
                    '%s' % (node.getCoordinate()))
            else:
                raise AnnotationFormatException('No matching by tree possible '
                    'and proximity fallback yielded no matches for synapse at '
                    '%s.' % (node.getCoordinate(),))
        else:
            raise AnnotationFormatException('No matching by tree possible for'
                'synapse at %s' % (node.getCoordinate(),))

    elif len(az_and_synapse) > 2:
        raise AnnotationFormatException('More than two compartments '
            'where synapse / active zone pair was expected (synapse at %s)' %
            (node.getCoordinate(),))
    else:
        raise AnnotationFormatException('Connected component splitting '
                                        'returned no annotations. This is '
                                        'probably a bug.')

    for cur_n in active_zone.getNodes():
        try:
            cur_comment = cur_n.getComment()
            if cur_comment is not None:
                extracted_id = re.search(active_zone_numbered_ex,
                    cur_n.getComment(),
                    re.IGNORECASE).group('syn_id')
        except AttributeError:
            pass

    return active_zone, extracted_id

def get_probable_az(s, length_cutoff=10000.):
    """
    Take a skeleton and return a set of annotations that are probably active
    zone annotations, based on the heuristic that they are relatively short
    and linear.

    Parameters
    ----------

    s : Skeleton instance

    length_cutoff : float
        Discard annotations that have physical lengths exceeding this value.

    Returns
    -------

    probable_az : set of SkeletonAnnotation instances

    az_centers_kd : KDTree of centers (average coordinates) of annotations.
    """

    probable_az = list()
    az_centers = list()

    for cur_a in s.getAnnotations():
        for cur_cc in split_by_connected_component(cur_a):
            if True in [x.degree() > 2 for x in cur_cc.getNodes()]:
                continue
            if 0 in [x.degree() for x in cur_cc.getNodes()]:
                continue
            if cur_cc.physical_length() > length_cutoff:
                continue

            probable_az.append(cur_a)

    for cur_az in probable_az:
        az_centers.append(average_coordinate([x.getCoordinate_scaled() for
            x in cur_az.getNodes()]))

    az_centers_kd = KDtree(probable_az, az_centers)

    return probable_az, az_centers_kd

def load_consolidated_synapses(synapses):
    """
    Generate dictionary that maps fold-consensus (int) to list of
    ConsolidatedSynapse objects from list of Synapse objects, where the
    Synapse annotations already include collection id and source_anno id tags.
    These tags are generated by consensus_synapses_to_skeleton based on the
    data output from generate_consensus_synapses and can be manually modified
    in Knossos to fix errors in automatic synapse matching.

    Parameters
    ----------

    synapses : list of Synapse instances
        This function will set the source_anno and collection_id
        attributes on the synapse instances, which can only succeed if the
        corresponding comments are set on the annotation, as done by the
        function consensus_synapses_to_skeleton.

    Returns
    -------

    consensus_synapses_by_redundancy : dict
        Maps fold-consensus (int) to list of ConsolidatedSynapse objects
    """

    collection_ex = r'collection\s*(?P<collection_id>\d+)'
    source_anno_ex = r'source_anno\s*(?P<source_anno_id>.+)'

    # Extract the comments necessary to set the source_anno and collection_id
    # attributes on the Synapse instances
    #

    for cur_s in synapses:
        cur_az = cur_s.az_anno
        collection_and_source_id_set = [False, False]

        if len(cur_az.getNodes()) < 2:
            raise AnnotationFormatException('Incomplete active zone '
                'annotation at %s.' % (str(cur_s.az_center_of_mass),))

        for cur_n in cur_az.getNodes():
            cur_comment = cur_n.getComment()
            if cur_comment is None:
                continue

            m = re.search(collection_ex, cur_n.getComment(), re.IGNORECASE)
            try:
                cur_collection_id = m.group('collection_id')
                cur_s.collection_id = cur_collection_id
                collection_and_source_id_set[0] = True
            except AttributeError:
                # No match
                pass

            # synapses_from_f0097 can set source_anno already if the
            # source_anno keyword argument is passed to it. We must overwrite
            # it here to get the source_anno string set by
            # consensus_synapses_to_skeleton
            m = re.search(source_anno_ex, cur_n.getComment(), re.IGNORECASE)
            try:
                cur_source_anno = m.group('source_anno_id')
                cur_s.source_anno = cur_source_anno
                collection_and_source_id_set[1] = True
            except AttributeError:
                # No match
                pass

        if not all(collection_and_source_id_set):
            raise AnnotationFormatException('Synapse annotation at %s does '
                'not have collection id and source_anno id.' %
                (str(cur_s.az_center_of_mass),))
        else:
            cur_s.collection_id = cur_collection_id
            cur_s.source_annotation = cur_source_anno

    # Collect the synapses by their collection id and source_annotation into
    # the final consensus_synapses data structure.
    synapses_by_collection = {}
    consensus_synapses_all = set()
    consensus_synapses_by_redundancy = {}

    for cur_synapse in synapses:
        synapses_by_collection.setdefault(cur_synapse.collection_id,
            []).append(cur_synapse)
    for cur_collection_id, cur_synapses in synapses_by_collection.iteritems():
        consensus_synapses_all.add(ConsolidatedSynapse(cur_synapses,
            collection_id=cur_collection_id))
    for cur_consolidated_synapse in consensus_synapses_all:
        consensus_synapses_by_redundancy.setdefault(
            cur_consolidated_synapse.redundancy_level, []).append(
            cur_consolidated_synapse)

    # Fill up gaps in redundancy for easier processing later
    for cur_redundancy in range(1,
                                max(consensus_synapses_by_redundancy.keys())):
        if not cur_redundancy in consensus_synapses_by_redundancy:
            consensus_synapses_by_redundancy[cur_redundancy] = []

    return consensus_synapses_by_redundancy

def synapses_from_f0097(skeleton, synapse_direction='to', source_anno='dummy',
        check_if_in_original_tracing=False, load_collections=False):
    """
    Generate a list of Synapse objects using the f0097 synapse annotation
    convention (and its variations).

    In this convention, the various synaptic features are annotated in the
    following way:

    Postsynaptic compartment: A node that contains the string 'postsynapse',
    possibly followed by an integer synapse ID.

    Active zone: A set of linearly connected nodes that are part of the
    same annotation (= tree) as the postsynapse node. One of the nodes can
    contain the string 'AZ' followed by an integer synapse ID. As a fallback,
    spatial proximity to the postsynapse node is used if the two are not
    matched by common tree.

    Presynaptic compartment: A node that contains the string 'presynapse',
    possibly followed by an integer synapse ID.

    Parameters
    ----------

    skeleton : Skeleton object

    synapse_direction : str
        If 'to', the postsynapse node must be an unconnected node and the
        presynapse node must be connected. (No other values currently
        supported)

    source_anno : str
        String that identifies the source annotation for the provided
        skeleton. Set this to the nml file name from which skeleton was read,
        for example.

    check_if_in_original_tracing : bool
        If True, check whether either the presynapse or the postsynapse
        annotated node is also part of the original tracing. Which of the two is
        correct depends on the value of synapse_direction.

    load_collections : bool
        If True, assume that the skeleton contains collection IDs and
        source_anno tags, as written by consensus_synapses_to_skeleton.


    Raises
    ------

    AnnotationFormatException
        A mistake in the annotation format requires manual correction.

    HeuristicMatchingException
        Spatial proximity based matching failed, requires manual correction
        (i.e. manual numbering of the ambiguous synapse).

    Returns
    -------

    synapses_out : Set of Synapse instances

    consensus_synapses_out : dict or None
        If load_collections is True, maps fold-consensus (int) to list of
        ConsolidatedSynapse objects, None otherwise

    Notes
    -----

    The synapse_direction parameter should be extended in the future to
    allow analysis of from-synapses and bidirectional synapses.

    Currently, this function operates on the assumption that synapses are
    annotated in just one tracing. It would also be possible to annotate
    synapses between two tracings, but this is currently not supported.

    The approach in this function is to first attempt to match presynapses,
    postsynapses and active zones based on their IDs. If there are pre- and
    postsynapses that match by ID, but for which no numbered active zone can
    be found, the active zone is matched based on its presence in the same
    tree as the pre- or postsynapse, depending on synapse_direction. If that
    fails, it is matched by spatial proximity.
    If there are pre- or postsynapses (depending on synapse_direction) for
    which no numbering is present, their matching active zone is detected
    based on its presence in the same tree (or spatial proximity as
    fallback) and the matching other compartment is detected by spatial
    proximity of a labeled, non-numbered node to the active zone.
    """

    other_compartment_distance_threshold = 650.
    postsynapse_numbered_ex = r'postsynapse\s*(?P<syn_id>\d*)'
    presynapse_numbered_ex = r'(presynapse|persynapse)\s*(?P<syn_id>[\d,\. ]*)'
    active_zone_numbered_ex = r'az\s*(?P<syn_id>\d*)'

    if synapse_direction == 'to':
        synapse_ex = postsynapse_numbered_ex
        other_compartment_ex = presynapse_numbered_ex
    else:
        raise Exception('Direction %s not implemented' % (synapse_direction,))

    # Find the compartment that represents the original tracing
    largest_compartment = get_largest_annotation(skeleton)

    compartments_numbered = {'synapse': {}, 'other': {}, 'active_zone': {}}
    compartments_unassigned = {'synapse': [], 'other': [], 'active_zone': []}
    compartments = [('synapse', synapse_ex),
        ('other', other_compartment_ex),
        ('active_zone', active_zone_numbered_ex)]

    for cur_n in skeleton.getNodes():
        for cur_compartment, cur_compartment_ex in compartments:
            cur_comment = cur_n.getComment()
            if cur_comment is None:
                continue
            try:
                cur_id = re.search(cur_compartment_ex, cur_comment,
                    re.IGNORECASE).group('syn_id')
            except AttributeError:
                # Does not have a matching comment
                continue

            if check_if_in_original_tracing and cur_compartment == \
                    'other' and synapse_direction == 'to':
                if cur_n.annotation is not largest_compartment[0]:
                    raise AnnotationFormatException('Node at %s should be '
                        'part of the original tracing, but isn\'t.' % (str(
                        cur_n.getCoordinate()),))

            if cur_id == '':
                # No synapse ID provided: Collect this compartment into the
                # "unassigned" category
                compartments_unassigned.setdefault(
                    cur_compartment, []).append(cur_n)
            else:
                # Synapse ID provided: Collect this compartment into the
                # "numbered" category. Synapse ID strings may be comma and /
                # or whitespace separated lists of actual integer IDs,
                # since one presynapse can have several postsynapses.
                all_ids = [x for x in cur_id.replace(',', ' ').replace(
                    '.', ' ').split(' ') if x]
                if len(all_ids) > 1:
                    for cur_id in all_ids:
                        cur_n_copy = copy.copy(cur_n)
                        if cur_id in compartments_numbered[cur_compartment]:
                            raise AnnotationFormatException('Duplicate id %s '
                                'at %s' % (cur_id, str(cur_n.getCoordinate()),))
                        compartments_numbered[cur_compartment][cur_id] = cur_n_copy
                        compartments_numbered[cur_compartment][cur_n_copy] = cur_id
                elif len(all_ids) == 1:
                    cur_id = all_ids[0]
                    if cur_id in compartments_numbered[cur_compartment]:
                        raise AnnotationFormatException('Duplicate id %s '
                            'at %s' % (cur_id, str(cur_n.getCoordinate()),))
                    compartments_numbered[cur_compartment][cur_id] = cur_n
                    compartments_numbered[cur_compartment][cur_n] = cur_id
                else:
                    raise AnnotationFormatException('Illegal synapse '
                        'numbering at %s.' % (cur_n.getCoordinate()))

    if len(compartments_numbered['synapse']) != \
            len(compartments_numbered['other']):
        raise AnnotationFormatException('There must be the same number of '
                                        'numbered synapse and postsynapse '
                                        'annotations.')

    if len(compartments_numbered['active_zone']) < \
            len(compartments_numbered['synapse']):
        print('Not all synapses have a corresponding active zone label, '
              'but some do. Will attempt to match the missing ones by '
              'presence in the same tree / proximity.')

    # Now, find active zones and match to synapses. We start by extracting
    # putative active zone annotations from the skeleton.
    probable_az, az_centers_kd = get_probable_az(skeleton)

    # Find the active zones for which we can assign a number,
    # because their matching synapse has a number
    for cur_s, cur_id in compartments_numbered['synapse'].iteritems():
        # compartments_numbered stores both the IDs as keys and the
        # actual node objects, we are only interested in the actual node
        # objects here
        if isinstance(cur_s, str):
            continue

        # We already have a matching active zone based on numbering,
        # so we don't have to fall back to matching by common tree / proximity
        if cur_id in compartments_numbered['active_zone']:
            continue

        active_zone, extracted_id = get_matching_az(cur_s,
            probable_az_kd=az_centers_kd)
        active_zone_node = list(active_zone.getNodes())[0]
        compartments_numbered['active_zone'][cur_id] = active_zone_node

    # There must now be the same number of numbered active zones, synapses
    # and other compartments and all IDs must be matched across the three.
    # Get the number of synapse / other / active zones detected with number
    lengths = [len([y for y in x.keys() if isinstance(y, str)]) for x in \
        compartments_numbered.values()]

    # Check that all numbers in lengths are equal
    if len(set(lengths)) is not 1:
        raise AnnotationFormatException('Number of numbered presynapses (%d), '
            'postsynapses (%d) and matched active zones (%d) is not '
            'identical.' % (lengths[0], lengths[1], lengths[2],))

    # Check that the same IDs are present for the numbered synapses, active
    # zones and other compartments.
    for cur_id in compartments_numbered['synapse']:
        if not isinstance(cur_id, str):
            continue
        if not cur_id in compartments_numbered['other'] or not \
                    cur_id in compartments_numbered['active_zone']:
            raise AnnotationFormatException('IDs not matched over numbered '
                'synapses, postsynapses and active zones.')

    # Match the unnumbered synapse compartments to active
    # zones based on tree, or, if that fails, by proximity.
    non_numbered_matches = {'synapse': {}, 'active_zone': {}, 'other': {}}
    for cur_s in compartments_unassigned['synapse']:
        active_zone, extracted_id = get_matching_az(cur_s,
            probable_az_kd=az_centers_kd)
        if extracted_id is not '':
            # An ID was manually added to the active zone annotation
            raise AnnotationFormatException('The active zone for synapse at '
                '%s has manually set ID %s, but the matching synapse doesn\'t.'
                % (cur_s.getCoordinate(), active_zone[1]))
        non_numbered_matches['synapse'][cur_s] = active_zone
        non_numbered_matches['active_zone'][active_zone] = cur_s

    # Now get all the non-numbered other compartments based on proximity to
    # active zones

    # Create structure for easy and efficient spatial lookup. We need to pass
    # an annotation to annotation_from_nodes so that the scaling can be
    # preserved.
    if len(compartments_unassigned['other']) > 0:
        unassigned_other = annotation_from_nodes(
            compartments_unassigned['other'],
            annotation=compartments_unassigned['other'][0].annotation)
        unassigned_other = EnhancedAnnotation(unassigned_other)

        for cur_az, cur_s in non_numbered_matches['active_zone'].iteritems():
            az_avg_coordinate = average_coordinate(
                [x.getCoordinate_scaled() for x in cur_az.getNodes()])
            az_avg_coordinate_dataset = [int(y) for y in average_coordinate(
                [x.getCoordinate() for x in cur_az.getNodes()])]
            other_match = unassigned_other.kd_tree_nodes.query_ball_point(
                az_avg_coordinate, other_compartment_distance_threshold)
            if len(other_match) > 1:
                raise HeuristicMatchingException('Other compartment match '
                    'ambiguous for active zone at %s' % (
                    str(az_avg_coordinate_dataset),))
            if len(other_match) == 0:
                raise HeuristicMatchingException('No matching other '
                    'compartment for active zone at %s within distance '
                    'threshold %f.' % (str(az_avg_coordinate_dataset),
                    other_compartment_distance_threshold,))
            other_match = other_match[0]
            if other_match in non_numbered_matches['other']:
                raise HeuristicMatchingException('Duplicate matching other '
                    'compartment at %s for active zone at %s.' %
                    (str(other_match.getCoordinate()),
                    str(az_avg_coordinate_dataset)))
            non_numbered_matches['other'][other_match] = (cur_az, cur_s)

        # There might still be other compartments that were not found by
        # matching from active zone but still had an annotation.
        for cur_other in compartments_unassigned['other']:
            if not cur_other in non_numbered_matches['other']:
                raise AnnotationFormatException(
                    'Other compartment was annotated, '
                    'but no matching synapse was found at %s' %
                    (cur_other.getCoordinate(),))

    # We collected everything we have now. For the numbered case, we have the
    # compartments_numbered dict that maps
    #   compartment type (str) -> synapse id -> SkeletonNode in compartment
    # For the non-numbered case, we have the non_numbered_matches dict
    # which maps
    #   'other' -> other compartment (SkeletonNode) ->
    #       (active zone (SkeletonAnnotation), synapse (SkeletonNode))

    synapses_out = set()

    for cur_other, (cur_az, cur_syn_node) in \
            non_numbered_matches['other'].iteritems():
        new_synapse = Synapse(az_nodes=[x for x in cur_az.getNodes()],
            pre_node=cur_other,
            post_node=cur_syn_node,
            source_anno=source_anno)
        synapses_out.add(new_synapse)

    for cur_other, cur_syn_id in compartments_numbered['other'].iteritems():
        if not isinstance(cur_syn_id, str):
            continue
        cur_az = compartments_numbered['active_zone'][cur_syn_id]
        cur_syn_node = compartments_numbered['synapse'][cur_syn_id]
        new_synapse = Synapse(az_nodes=get_reachable_nodes(cur_az),
            post_node=cur_syn_node,
            pre_node=cur_other,
            source_anno=source_anno)
        synapses_out.add(new_synapse)

    if load_collections == False:
        return synapses_out, None
    else:
        consensus_synapses_out = load_consolidated_synapses(synapses_out)
        return synapses_out, consensus_synapses_out

def consensus_synapses_to_skeleton(consensus_synapses):
    """
    Convert a consensus synapse dictionary, as generated by
    generate_consensus_synapses to an NML file that can be used for manual
    correction and completion of the automatic lumping of independently
    identified synapses into consensus synapses.

    Parameters
    ----------

    consensus_synapses : dict
        Maps integer redundancy level to a list of ConsolidatedSynapse
        instances.

    Returns
    -------

    s : Skeleton instance
    """

    s = Skeleton()
    cur_id = 1

    for redundancy, consensus_syns in consensus_synapses.iteritems():
        for cur_consensus_syn in consensus_syns:
            if redundancy == 1:
                identifier_node_comment = '(todo) collection %d' % (cur_id,)
            else:
                identifier_node_comment = 'collection %d' % (cur_id,)

            cur_anno = consensus_synapse_to_annotation(cur_consensus_syn,
                identifier_node_comment=identifier_node_comment)
            s.add_annotation(cur_anno)

            cur_id += 1

    return s

def consensus_synapse_to_annotation(consensus_syn, identifier_node_comment):
    """
    Convert a ConsolidatedSynapse instance to SkeletonAnnotation.

    The SkeletonAnnotation will contain numbered synapse tracings readable by
    synapses_from_f0097. Additionally to the usual format, the node
    containing the "az <id>" comment will contain a "source_anno <id>"
    comment, that identifies the annotation from which that particular
    synapse was loaded. A different node in the active zone tracing will
    contain a comment "collection <id>", possibly with an additional "(todo)"
    label. This id is used to match synapses from different annotations.

    Parameters
    ----------

    consensus_syn : ConsolidatedSynapse instance

    identifier_node_comment : str

    Returns
    -------

    new_anno : SkeletonAnnotation instance
    """

    new_anno = SkeletonAnnotation()

    for cur_s in consensus_syn.synapse_annotations:
        cur_anno = synapse_to_annotation(cur_s)

        # Add source annotation identifier string to active zone comment node
        az_comment_node = get_nodes_with_comment(cur_anno, r'az',
            re.IGNORECASE)
        if not len(az_comment_node) == 1:
            # print az_comment_node
            # print [x.getComment() for x in az_comment_node]
            raise Exception('Something bad happened.')
        az_comment = az_comment_node[0].getComment()
        az_comment = az_comment + ' source_anno ' + cur_s.source_annotation
        az_comment_node[0].setComment(az_comment)

        # Get any other node from the AZ annotation
        for cur_n in get_reachable_nodes(az_comment_node[0]):
            if cur_n is not az_comment_node[0]:
                break
        if cur_n == az_comment_node[0]:
            raise Exception('Something bad happened: %s, %s.' % (str(cur_n
                .getCoordinate()), str(get_reachable_nodes(az_comment_node[0]))))
        cur_n.setComment(identifier_node_comment)

        new_anno = merge_annotations(new_anno, cur_anno)

    return new_anno

def synapse_to_annotation(synapse,
                          syn_id=None,
                          clear_comments=True,
                          syn_style='fish',
                          add_todo_label=False):
    """
    Generate annotation representing Synapse object.

    Parameters
    ----------

    synapse : Synapse instance

    syn_id : int or None
        Synapse ID to match presynapse, postsynapse and active zone
        compartment. If None, a random ID will be assigned.
    clear_comments : Bool
        Clears existing comments in the AZ nodes.

    Returns
    -------

    new_anno : SkeletonAnnotation instance
    """

    if syn_id == None:
        syn_id = randint(0, 99999999999)

    new_anno = copy.copy(synapse.az_anno)
    az_comment_node = get_nodes_with_comment(new_anno, r'az',
        re.IGNORECASE)
    if len(az_comment_node) > 1:
        #print [x.getComment() for x in az_comment_node]
        if len(set([x.getComment() for x in az_comment_node])) > 1:
            print [x.getComment() for x in az_comment_node]
            #raise Exception('Something bad happened.')
    if len(az_comment_node) == 0:
        az_comment_node = list(new_anno.getNodes())[0]
    else:
        az_comment_node = az_comment_node[0]

    if clear_comments:
        [node.setComment('') for node in new_anno.getNodes()]

    if add_todo_label:
        for az_node in new_anno.getNodes():
            # find AZ node without the comment label
            if not (az_node == az_comment_node):
                az_node.setComment('todo')
                break

    if syn_style == 'fish':
        az_comment_node.setComment('AZ %s' % (syn_id,))
    elif syn_style == 'bird':
        az_comment_node.setComment('s%03d-az' % (syn_id,))

    cur_pre = copy.copy(synapse.preNode)
    if syn_style == 'fish':
        cur_pre.setComment('Presynapse %s' % (syn_id,))
    elif syn_style == 'bird':
        cur_pre.setComment('s%03d-p4' % (syn_id,))
    else:
        raise Exception('syn_style not supported')

    cur_post = copy.copy(synapse.postNode)
    if syn_style == 'fish':
        cur_post.setComment('Postsynapse %s' % (syn_id,))
    elif syn_style == 'bird':
        cur_post.setComment('s%03d-px' % (syn_id,))
    else:
        raise Exception('syn_style not supported')

    new_anno.addNode(cur_pre)
    new_anno.addNode(cur_post)

    return new_anno

def synapses_to_skeleton(synapses,
                         syn_style='fish',
                         add_todo_label=False):
    """
    Create Skeleton object that contains synapse annotations only,
    exported from Synapse objects.

    Parameters
    ----------

    synapses : Iterable of Synapse instances or Synapse instance

    Returns
    -------

    s : Skeleton instance
    """

    if isinstance(synapses, Synapse):
        synapses = [synapses]

    s = Skeleton()

    cur_id = 0

    for cur_s in synapses:
        cur_id += 1
        cur_a = synapse_to_annotation(cur_s, cur_id,
                                      syn_style=syn_style,
                                      add_todo_label=add_todo_label)
        s.add_annotation(cur_a)

    return s

def j0126_axon_syn_analysis(path_to_dir):
    """

    This is a large "meta" method that runs a more or less complete analysis.

    1) Lists all final j0126 synTaskA nml files
    2) Identifies and groups redundant annotations (on a filename basis)
    3) Tries to extract synapses out of each file (j0126 annotation format)
    4) Tries to generate consensus synapses
    5) Loads in skeleton consensus files and links the consensus synapses
       with these consensus skeletons

    Parameters
    ----------

    path_to_dir : str


    Returns
    -------

    detected_synapses : Set of Synapse instances

    Notes

    """

    # used to find the latest final version of a submission
    same_annotator = dict()

    # used to find redundant annotations
    grouped_nmls = dict()

    task_extractor = \
        re.compile(r'(?P<task>.*)-(?P<annotator>.*)-\d*-\d*-final\.nml$')

    all_NML_files = [file for file in os.listdir(path_to_dir)
        if file.lower().endswith('.nml')]

    for nml_file in all_NML_files:
        mobj = task_extractor.search(nml_file)

        if mobj:
            try:
                grouped_nmls[mobj.group('task')][mobj.group('annotator')] = \
                                                                    nml_file
            except:
                grouped_nmls[mobj.group('task')] = \
                                        {mobj.group('annotator'): nml_file}

    #for same_task, same_annotator in grouped_nmls:
    #    for s

    for nmlfile in allNMLfiles:
        print 'loading ' + nmlfile
        annos = au.loadj0126NML(os.path.join(path_to_dir, nmlfile))

    return


def synapses_from_j0126(annotation,
                        extractionMode='taskA',
                        enable_heuristic=False,
                        log_file=''):
    """
    Extracts synapse annotations from annotation objects.
    Each synapse node is labeled with a unique int identifier, multiple
    synapse nodes belonging to the same synapse can be grouped together
    by this approach. The extraction works as follows:

        - Find all nodes with a synapse id comment
        - Group nodes with the same synapse id with a dict
        - Parse nodes belonging to the same synapse and create a synapse obj

    Since human annotators make many mistakes, several sanity checks were
    implemented to identify incomplete annotations and heuristics are applied to
    correct the most obvious mistakes.

    Parameters
    ----------

    annotation : SkeletonAnnotation object

    extractionMode : 'complete' || 'taskA'
        'taskA': Expects pre, post node and active zone annotation,
                 and performs the possible sanity checks based on this
                 expectation
        'complete': Expects annotations from taskA + taskB
                    i.e. complete spine annotations, density, vesicles

    log_file : String


    Returns
    ---------

    list of found Synapse objects

    Raises
    ------

    Nothing so far.

    Notes
    -----

    -

    """

    if log_file != '':
        logf = open(log_file,'a')
        sys.stdout = logf

    #if type(annotation) == list:
    #     if len(annotation) == 1:
    #         annotation = annotation[0]
    #     else:
    #         raise Exception('Please only pass a single annotation')

    #print("Start parsing of: {0}".format(annotation.filename))

    if not levenshtein_available:
        enable_heuristic = False
        print("Heuristic disabled against request,"
              "please install Levenshtein distance string comparison.")


    # Make a deep copy of the annotation - this is necessary due to some
    # weird in-place modifications that happen somewhere downstream and
    # change the actual annotation passed (nodes are fragmented into multiple
    # annotation objects). This is something todo.

    #annotation = copy.deepcopy(annotation)

    #if len(set([node.annotation for node in annotation.getNodes()])) > 1:
    #    raise Exception('The annotation has inconsistent node->annotation
    # refs')

    # prepare a few data structures for later use
    nxG = au.annoToNXGraph(annotation, merge_annotations_to_single_graph=True)

    synapseNodes =\
        au.getNodesByCommentRegex(r'^(s|bb)\d+[abcdefghijklmnopqr]?[_\-.]?',
                                  annotation)

    # put synapses into a dictionary
    synapses = dict()
    #synapsematcher = re.compile(r'(?P<synID>s\d+[a-m]?)[_\-.]p[x01234]')
    synapsematcher = re.compile(r'(?P<synID>(s|bb)\d+[a-m]?).*')
    for synNode in synapseNodes:
        mobj = synapsematcher.search(synNode.getPureComment())
        if mobj:
            try:
                synapses[mobj.group('synID').lower()].append(synNode)
            except:
                synapses[mobj.group('synID').lower()] = [synNode]


    synObjs = []
    syns_with_parsing_problem = []

    synNodeMatcher = re.compile(
        r'(?P<synid>(s|bb)\d+[a-m]?)[_\-.]?(?P<spos>p\d)?[_\-.]?('
        r'?P<vesicles>f?m?g?'
        r'(vus)?)'
        r'?[_\-.]?(?P<density>(as)?(sy)?(dus)?)?')

    postPxMatcher = re.compile(r'px', re.IGNORECASE)
    postPNonxMatcher = re.compile(r'p[0123]', re.IGNORECASE)
    preMatcher = re.compile(r'p4', re.IGNORECASE)
    AZMatcher = re.compile(r'^(s|bb)\d+[a-m]?(-az)?[?]?$', re.IGNORECASE)


    all_comment_nodes = au.get_all_comment_nodes(annotation)
    comment_nodes_kd = au.KDtree(all_comment_nodes)

    # create a synapse object for each key
    for synID, synNodes in synapses.iteritems():
        if len(synNodes) > 4:
            print("Unplausible number of synapse nodes for: {0}".format(synID))
            if enable_heuristic == False: continue

        currSyn = Synapse()


        currSyn.rawSourceComments.append(synID)
        try:
            currSyn.sourceAnnoFile = annotation.filename
            currSyn.sourceUsername = annotation.username
        except:
            currSyn.sourceAnnoFile = annotation[0].filename
            currSyn.sourceUsername = annotation[0].username

        for synNode in synNodes:

            mobj = synNodeMatcher.search(synNode.getPureComment())

            postxmobj = postPxMatcher.search(synNode.getPureComment())
            postnonxmobj = postPNonxMatcher.search(synNode.getPureComment())
            premobj = preMatcher.search(synNode.getPureComment())
            azmobj = AZMatcher.search(synNode.getPureComment())

            #print synNode.getPureComment()
            if azmobj:
                # this COULD be a node indicating the AZ, further tests
                # must be performed to be sure

                putativeAZnodes = nx.node_connected_component(nxG, synNode)
                pLens = nx.single_source_dijkstra_path_length(nxG, synNode)

                # take the longest possible path length and see if its too
                # long; this means that an annotator made a mistake
                putativeAZlen = sorted([length for key, length \
                    in pLens.items()], reverse=True)[0]

                putativeAZlen /= 1000. # in microns
                #print synNode.getPureComment()
                #print 'found AZ ' + str(
                #    putativeAZlen) + ' originating at node ' + synNode\
                #    .getPureComment()
                if putativeAZlen < 10.:
                    # AZs longer than 10 um should not exist
                    #currSyn.AZlen = putativeAZlen
                    currSyn.AZnodes = list(putativeAZnodes)

            if premobj:
                # this node is a presynaptic node
                currSyn.preNode = synNode
                currSyn.preNodeCoord = synNode.getCoordinate_scaled()
            if postxmobj:

                # this node is a postsynaptic node, without
                # further manual location specification
                currSyn.postNode = synNode
                currSyn.postNodeCoord = synNode.getCoordinate_scaled()
            elif postnonxmobj:
                # this node is a postsynaptic node, with
                # further manual location specification
                # this means that vesicles and the density
                # were also annotated

                #print 'in postnonxmobj' + synNode.getPureComment()
                if mobj.group('vesicles') and mobj.group('density'):
                    # this node contains vesicle and density annotations
                    currSyn.density = mobj.group('density')
                    currSyn.vesicles = mobj.group('vesicles')
                    currSyn.postNode = synNode
                    currSyn.postNodeCoord = synNode.getCoordinate_scaled()
                    currSyn.postPos = mobj.group('spos')

                #if not mobj.group('vesicles') and not mobj.group('density'):
                    # this must be a spine annotation node, we therefore
                    # create a spine object and attach it to the synapse obj

                    #print 'Found putative spine annotation for synapse: ' + \
                    #      currSyn.rawSourceComments[0] + ' from file ' + \
                    #      currSyn.sourceAnnoFile

                    # todo create spine obj

        # sanity checks & heuristic to fix annotator errors follow
        # this depends on the extraction mode
        # build a kd-tree with all nodes that have a comment
        # try to fix tracer errors

        if extractionMode == 'taskA':
            # Each synapse obj should have a
            # pre AND post node AND an active zone
            synIsSane = True

            # test if pre node is missing and attempt to fix
            if not currSyn.preNode and currSyn.postNode and currSyn.AZlen > 0.:

                # Try spatial lookup to fix the attention problem of the
                # annotator. This needs to be done very conservative.

                # query tree at post node coord for nodes with a comment

                # this should yield exactly 3 nodes: AZ node, missing pre
                # node and the post node. the missing preNode needs at least
                # contain 's' or a single digit in its comment to be accepted

                #print 'Could not identify pre node for ' + \
                #      currSyn.rawSourceComments[0] + ' from file ' + \
                #      currSyn.sourceAnnoFile + ' trying heuristic fix.'
                print("Could not identify pre node for: {0}".format(synID))

                synIsSane = False
                # searches in the proximity of 500 nm around the post
                # synapse annotation node for a putative pre synapse node
                found_nodes = comment_nodes_kd.query_ball_point(
                    currSyn.postNode.getCoordinate_scaled(), 1000.)

                for curr_node in found_nodes:
                    if (curr_node != currSyn.postNode) and not \
                            (curr_node in currSyn.AZnodes):
                        if Levenshtein.distance(curr_node.getPureComment(),
                                                synID + '-p4') == 1:
                            if enable_heuristic:
                                synIsSane = True
                                currSyn.preNode = curr_node

                                print("Fixed pre node for: {0}".format(synID))
                                print("New pre node is {0} at {1}".
                                      format(curr_node.getPureComment(),
                                             str(curr_node)))

            # test if post node is missing and attempt to fix
            if not currSyn.postNode and currSyn.preNode and currSyn.AZlen > 0.:

                print("Could not identify post node for: {0}".format(synID))

                synIsSane = False
                # searches in the proximity of 500 nm around the pre
                # synapse annotation node for a putative post synapse node
                found_nodes = comment_nodes_kd.query_ball_point(
                    currSyn.preNode.getCoordinate_scaled(), 1000.)

                for curr_node in found_nodes:
                    if (curr_node != currSyn.preNode) and not \
                        (curr_node in currSyn.AZnodes):

                        if Levenshtein.distance(curr_node.getPureComment(),
                                                synID + '-px') == 1:
                            if enable_heuristic:
                                synIsSane = True
                                currSyn.postNode = curr_node

                                print("Fixed post node for: {0}".format(synID))
                                print("New post node is {0} at {1}".
                                      format(curr_node.getPureComment(),
                                             str(curr_node)))

            if not currSyn.AZlen > 0. and currSyn.postNode and currSyn.preNode:
                synIsSane = False
                print("Could not identify az nodes for: {0}".format(synID))

                found_nodes = comment_nodes_kd.query_ball_point(
                    currSyn.postNode.getCoordinate_scaled(), 1000.)

                if len(found_nodes) > 2 and len(found_nodes) < 5:
                    for curr_node in found_nodes:
                        if (curr_node != currSyn.preNode) and \
                                (curr_node != currSyn.postNode):

                            # this is a candidate AZ seed node, test whether
                            # further safety criteria are fulfilled

                            putativeAZnodes = nx.node_connected_component(nxG,
                                curr_node)

                            if len(putativeAZnodes) < 2:
                                break

                            pLens = nx.single_source_dijkstra_path_length(nxG,
                                curr_node)

                            # take the longest possible path length and see
                            # if its too long;
                            putativeAZlen = sorted([length for key, length \
                                in pLens.items()], reverse=True)[0]

                            putativeAZlen /= 1000. # in microns

                            if putativeAZlen < 10.:
                                if synID in curr_node.getPureComment():
                                    # AZs longer than 10 um should not exist

                                    if enable_heuristic:
                                        print("Fixed AZ for: {0}".format(synID))
                                        print("New AZ start is {0} at {1}".
                                            format(curr_node.getPureComment(),
                                                   str(curr_node)))

                                        currSyn.AZnodes = putativeAZnodes
                                        synIsSane = True

                #if synIsSane:
                #    print 'Heuristic fix successful.'
                #else:
                #    print 'Heuristic fix failed.'

            if not currSyn.preNode or not currSyn.postNode or not \
                    currSyn.AZnodes:
                synIsSane = False

            if synIsSane:
                if currSyn.preNode.distance_scaled(currSyn.postNode) / 1000. \
                        > 5.:

                    synIsSane = False
                    print("Pre- post node too far away for: {0}".format(synID))
                    #print 'Pre and post nodes too far away for ' + \
                    #      currSyn.rawSourceComments[0] + ' from file ' + \
                    #      currSyn.sourceAnnoFile

                if currSyn.postNode in currSyn.AZnodes or \
                                currSyn.preNode in currSyn.AZnodes:

                    print("Pre / post in AZ for: {0}".format(synID))
                    #print 'Pre or post node in AZ nodes for ' + \
                    #      currSyn.rawSourceComments[0] + ' from file ' + \
                    #      currSyn.sourceAnnoFile
                    synIsSane = False

                if currSyn.preNode.getComment() == \
                        currSyn.postNode.getComment():
                    print("Pre/post have same comments for: {0}".format(synID))
                    #print 'Pre - and post nodes have same comments.'
                    synIsSane = False

            if synIsSane:
                synObjs.append(currSyn)
            else:
                syns_with_parsing_problem.append(currSyn)
                #else if extractionMode == 'complete':
                #    if currSyn.preNode and not currSyn.postNode:
                #        print 'Could not identify post node for synapse \
                #            (but found pre node): ' + currSyn
                # .rawSourceComments[0]+\
                #            ' from file ' + currSyn.sourceAnnoFile
                #
                #    elif currSyn.preNode and currSyn.postNode:
                #        synObjs.append(currSyn)



    # Another pass over all found synapses to identify type, location or "no"
    # labels


    new_synObjs = []
    for syn in synObjs:
        syn_nodes = list(syn.AZnodes) # make a copy
        syn_nodes.append(syn.preNode)
        syn_nodes.append(syn.postNode)
        for snode in syn_nodes:
            # Normal labeling (label in az node only)
            if Levenshtein.distance(snode.getPureComment(),'spine-head') <= 1:
                syn.location_tags.append('spine-head')
            elif Levenshtein.distance(snode.getPureComment(),'spine-neck') <= 1:
                syn.location_tags.append('spine-neck')
            elif Levenshtein.distance(snode.getPureComment(),'shaft') <= 1:
                syn.location_tags.append('shaft')
            elif Levenshtein.distance(snode.getPureComment(),'soma') <= 1:
                syn.location_tags.append('soma')
            elif Levenshtein.distance(snode.getPureComment(),'as') <= 1:
                syn.type_tags.append('as')
            elif Levenshtein.distance(snode.getPureComment(),'sy') <= 1:
                syn.type_tags.append('sy')
            elif Levenshtein.distance(snode.getPureComment(),'no') <= 1:
                syn.no_tags.append('no')

            # Dimitar style labeling
            if 'head' in snode.getPureComment():
                syn.location_tags.append('spine-head')
            elif 'shaft' in snode.getPureComment():
                syn.location_tags.append('shaft')
            elif 'soma' in snode.getPureComment():
                syn.location_tags.append('soma')
            elif 'neck' in snode.getPureComment():
                syn.location_tags.append('spine-neck')
            if 'black' in snode.getPureComment():
                syn.tags.add('bb')
            if 'bb' in snode.getPureComment():
                syn.tags.add('bb')
            if 'onfirmed' in snode.getPureComment():
                syn.tags.add('confirmed')
            if 'axoax' in snode.getPureComment():
                syn.tags.add('axo-axo')
            if 'as' in snode.getPureComment():
                syn.type_tags.append('as')
            elif 'sy' in snode.getPureComment():
                syn.type_tags.append('sy')
            if 'correct' in snode.getPureComment():
                syn.tags.add('correct')
            elif 'remove' in snode.getPureComment():
                syn.tags.add('remove')
            elif 'unclear' in snode.getPureComment():
                syn.tags.add('unclear')


        new_synObjs.append(syn)

    synObjs = new_synObjs

    total_syns_found = len(synObjs) + len(syns_with_parsing_problem)
    print("Total synapses successfully parsed: {0}".format(total_syns_found))

    if len(synObjs) > 0:
        print("{0}% bad synapses".format(float(len(syns_with_parsing_problem))/
                                     float(total_syns_found)*100.))

    print('\n')

    if log_file != '':
        sys.stdout = sys.__stdout__
        logf.close()

    return synObjs, syns_with_parsing_problem


def getAllSpines(annotation):
    synapses = synapses_from_j0126(annotation)
    spines = dict()
    for synID, nodes in synapses.items():
        if len(nodes) > 2: #spine synapse found
            spines[synID] = nodes

    return spines


def calcHeadNeckRatio(spines):
    """spines needs to be a dictionary of synapse IDs and nodes"""
    ratios = dict()
    for synID, nodes in spines.items():
        ratios[synID] = lambda: 0
        # get p2 and p3
        for node in nodes:
            if 'p2' in node.getPureComment().lower():
                ratios[synID].p2 = node.getDataElem('radius')
            elif 'p3' in node.getPureComment().lower():
                ratios[synID].p3 = node.getDataElem('radius')
        try:
            ratios[synID].p2
            ratios[synID].p3
            ratios[synID] = ratios[synID].p3 / ratios[synID].p2
        except:
            ratios[synID] = float('NaN')
    return ratios


def getSpineSomaDistance(annotations, spines):
    """Returns for each spine its distance to the soma node"""

    somaAnno = au.getAnnosByCommentRegex(r'Soma', annotations)

    return


def correlateAZLenHeadDia(annotation):
    return


def calcSynapseDisagreement(annotations):
    """All annotations passed should only contain annotator variability"""

    return


def plotAnnotationSynProperties(annotation):
    """Plots the properties of all synapses of 98"""
    return

# 1) synapse locations are labeled with pX-syn-properties and p4
# (3x redundancy)
# => how to merge? majority vote? all agree that there IS a synapse,
# x)

def genConsenusSynapsesTaskA(redundantAnnos, reqRedundancy, spotlightRadius):
    # redundantAnnos must contain single annotations describing the same tree
    # individual synapses are matched by:
    # directionality of pre- post nodes; spatial proximity of pre- post nodes

    # a consenus synapse is generated by spatial averaging of the pre and
    # post nodes
    # in case that at least reqRedundancy annotators found a synapse at
    # the "same"
    # location

    # first get all synapses out of all redundant annos
    # insert the nodes into kd-trees for fast spatial lookups



    return consensusSynapses


# missing for task generators:
# fake knossos version
# include locking for B2a
# include branch node generation for B and C tasks


def genSynTaskC_NML(consSynapsesA, consAnno):
    # Task C: AZ measurement
    # source: list of redundant
    # NML file contains for each synapse:
    # tree with 2 nodes (pre - and post node), post node as branch node

    # collects all pre (p4) - and postsynapse (p0,1,2,3 with additional
    # synapse tag) nodes
    synapseNodes = getAllSynapses(annotations)

    skel = ns.Skeleton()

    return filename


def genSynTaskB2a_NML(consSynapsesA, seedID, path):
    skel = au.genj0126SkelObj()

    # generate one artifical anno for each synapse
    # each artifical anno contains the post node, tagged as branch point
    # with the locking comment 'radiuslocked'
    currNodeID = 1
    currAnnoID = 1
    for consSynapse in consSynapsesA:
        # create anno
        currAnno = ns.SkeletonAnnotation()
        currAnno.annotation_ID = currAnnoID
        # create node with comment
        newNodePost = ns.SkeletonNode()
        newNodePost.from_scratch(currAnno, consSynapse.postNodeCoord[0],
            consSynapse.postNodeCoord[1],
            consSynapse.postNodeCoord[2],
            inVp=1, inMag=1, time=0,
            ID=currNodeID, radius=1.0)
        newNodePost.setPureComment('radiuslocked')

        currAnno.addNode(newNodePost)

        # add anno to skel
        skel.annotations.add(currAnno)
        currNodeID += 1
        currAnnoID += 1
        skel.branchNodes.append(currNodeID)

    # the nml file is written with the locking options

    filename = path + skel.experimentName + '-' + seedID + '-synB2a.000.nml'
    skel.toNml(filename)
    return


def genSynTaskB_NML(consSynapsesA, consAnno, path):
    skel = au.genj0126SkelObj()



    # add a branch point into each


    # remove all irrelevant properties from the consensus anno
    # => THIS MODIFIES NODES IN PLACE, NO DEEP COPY!
    [node.setPureComment('') for node in consAnno.getNodes()]
    [node.setDataElem("radius", 2.0) for node in consAnno.getNodes()]

    skel.annotations.add(consAnno)

    filename = path + skel.experimentName + '-' + \
               consAnno.seedID + '-synA.000.nml'
    skel.toNml(filename)

    return


def massGenSynTaskA_NML(consAnnos, pathForNMLtaskFiles):
    return


def genSynTaskA_NML(consAnno, path):
    # Task A: find presynapse and postsynapse nodes (tag with px:
    # postsynapse and p4: presynapse)
    skel = au.genj0126SkelObj()

    consnxG = annoToNXGraph(consAnno)
    # get all branch nodes
    bNodes = [node for node, deg in consnxG.degree().items() if deg > 2]

    for bNode in bNodes:
        edges = zip([node.distance_scaled(child) for child in
            bnode.getChildren()], bnode.getChildren())
        edges.sort(reverse=True) # sorts by length, longest first
        # remove all but longest edge
        [bNode.removeChild(child) for length, child in edges[1:]]

        # refresh nxGraph with now removed edges
    consnxG = annoToNXGraph(consAnno)

    # find connected components
    frags = nx.connected_components(consnxG)
    fragsnxG = nx.connected_component_subgraphs(consnxG)
    fragsWithLens = zip(frags, [fragnxG.size(weight='weight') / 1000
        for fragnxG in fragsnxG])

    # remove all irrelevant properties from the consensus anno
    # => THIS MODIFIES NODES IN PLACE, NO DEEP COPY!
    #[node.setPureComment('') for node in consAnno.getNodes()]
    #[node.setDataElem("radius", 2.0) for node in consAnno.getNodes()]

    skel.annotations.add(consensusAnno)

    filename = path + skel.experimentName + '-' + \
               consensusAnno.seedID + '-synA.000.nml'
    skel.toNml(filename)
    return


def synapsesToKDtree(synapses, synapse_location='pre_post_average',
        scaling=(10.0, 10.0, 20.0)):
    """

    Uses the KD-tree wrapper class in annotationUtils and creates
    a KD-tree for efficient spatial synapse searches.

    Parameters
    ----------

    synapses : iterable of Synapse instances

    synapse_location : str
        'pre_post_average' or 'az_average'

    scaling : iterable of float
        Scaling factor between voxels and physical units

    """

    if synapse_location == 'pre_post_average':
        coords=[synapse.avgPrePostCoord_scaled for synapse in synapses]
    elif synapse_location == 'az_average':
        coords=[synapse.az_center_of_mass_scaled for synapse in synapses]
    elif synapse_location == 'post':
        coords=[synapse.postNode.getCoordinate_scaled() for synapse in synapses]
    elif synapse_location == 'pre':
        coords=[synapse.preNode.getCoordinate_scaled() for synapse in synapses]
    else:
        raise Exception('synapse_location parameter must be \'az_average\' or'
            '\'pre_post_average\' or \'post\' or \'pre\'')

    synTree = au.KDtree(synapses, coords)

    return synTree



def synapse_discrepancy_analyzer(synapses):
    '''
    Parameters:
        synapses: iterable of iterables of synapse objects




    Returns:
    -


    '''

    # histogram of closest synapse from other redundant annotation

    # histogram of closest synapse from same annotation


    return

def generate_consensus_synapses(synapses,
                                spotlight_radius=205,
                                scaling=(10.0, 10.0, 20.0),
                                synapse_location='az_average'):
    """
    Lump multiple independent synapse annotations into a consensus by
    matching synapses between annotations by proximity.

    Parameters
    ----------

    synapses : Iterable of iterable of Synapse instances
        E.g. a list of synapse lists, where every list contains the synapses
        extracted from the same annotation.

    spotlight_radius : int
        Radius in which to lump synapses

    scaling : Iterable of float
        Scaling factor between voxels and physical units

    synapse_location : str
        Either 'pre_post_average' or 'az_average'. In the first case, use the
        average of pre- and postsynapse positions as synapse location. In the
        latter case, use the active zone average position.

    Returns
    -------

    consensus_synapses : dict
        Maps fold-consensus (int) to list of ConsolidatedSynapse objects

    same_matches : int
        Number of synapses that matched a synapse in the same annotation
        (these are discarded from further processing, but the number can give
        you an idea of how well your spotlight_radius is chosen.)
    """

    # The Synapse objects need an identifier that says what annotation they
    # are from. We use the username, if available. If it isn't, we assign
    # random string as identifier.


    for cur_syns in synapses:
        cur_usernames = list(set([x.sourceUsername for x in cur_syns]))

        try:
            if len(cur_usernames) != 1 or cur_usernames[0] == '' \
                    or cur_usernames[0] == None:
                arbitrary_id = gen_random_pass(8)
                for cur_s in cur_syns:
                    cur_s.sourceUsername = arbitrary_id
        except KeyError:
            print('Warning: One set of synapses is empty')

    # Flatten synapse lists
    all_syns = list(itools.chain.from_iterable(synapses))

    # Prepare spatial lookups, just put all synapses into the KDtree
    all_syns_kd = synapsesToKDtree(all_syns, synapse_location='az_average',
        scaling=scaling)

    synsWithPartners = {}

    # Query all against all, the result is a list of synapse lists,
    # for each synapse query
    if synapse_location == 'pre_post_average':
        locations = [Coordinate(syn.avgPrePostCoord) * Coordinate(scaling) for
            syn in all_syns]
    elif synapse_location == 'az_average':
        locations = [Coordinate(syn.az_center_of_mass) * Coordinate(scaling)
            for syn in all_syns]
    else:
        raise Exception('synapse_location parameter setting not supported.')

    # Find all synapses that lie within spotlight_radius for every synapse
    found_syns = all_syns_kd.query_ball_point(locations, spotlight_radius)

    # Since we added all synapses irrespective of origin into the same KD
    # Tree, we now need to remove those matches that came from the same
    # original annotation.
    same_matches = 0
    for synIndex, queryResult in enumerate(found_syns):
        synsWithPartners[all_syns[synIndex]] = []
        query_result_distinct = [x for x in queryResult if x.sourceUsername
                                 != all_syns[synIndex].sourceUsername]
        query_result_same = [x for x in queryResult if x.sourceUsername
                             == all_syns[synIndex].sourceUsername]

        synsWithPartners[all_syns[synIndex]] = query_result_distinct

        if len(query_result_same) > 1:
            same_matches += 1
            #print('Same-annotation match: Presynapse at %s with '
            #      'presynapse(s) at %s.' %
            #      (str(all_syns[synIndex].preNodeCoord),
            #       str([x.preNodeCoord for x in query_result_same])))

    # We now lump the matches into consensus synapses. For this to work
    # correctly, we require that all synapses in a consensus synapses are
    # matched to all other synapses in the same consensus (that they form a
    # maximal clique) and that there is no ambiguity in what clique to use
    # (that the clique is equal to a connected component in the graph).
    # Further, all nodes in the clique must represent synapses from distinct
    # original annotations.
    # In all other cases, we consider the synapses unmatched (1-redundant).
    #

    consensus_synapses = dict()
    handled_synapses = set()
    match_graph = nx.from_dict_of_lists(synsWithPartners)
    cliques = list(nx.find_cliques(match_graph))
    for cur_clique in cliques:
        corresponding_component = nx.node_connected_component(match_graph,
                                                              cur_clique[0])
        if len(cur_clique) != len(corresponding_component):
            continue

        source_annos = [x.sourceUsername for x in cur_clique]
        if len(source_annos) != len(set(source_annos)):
            # Not all synapses are from distinct annotations
            continue

        handled_synapses.update(cur_clique)
        cur_consolidated_synapse = ConsolidatedSynapse(cur_clique)
        consensus_synapses.setdefault(len(cur_clique), []).append(
            cur_consolidated_synapse)

    # We must add all those synapses that were part of an "ambiguous" clique,
    # or clique rejected due to combining same original annotations multiple
    # times, as 1-redundant synapses so they aren't lost.
    #

    unhandled_synapses = set(all_syns) - handled_synapses
    for cur_s in unhandled_synapses:
        cur_consolidated_synapse = ConsolidatedSynapse([cur_s])
        consensus_synapses.setdefault(1, []).append(cur_consolidated_synapse)

    return consensus_synapses, same_matches

def synDiscrepancyHighlighter(listOfPathsToNMLs,
        reqRedundancy=2,
        spotlightRadius=90,
        path='',
        synapseFormat='j0126',
        task='taskA',
        writeToFile=False,
        verbose=False):
    listOfListOfSynObjs = []
    for sourcefile in listOfPathsToNMLs:
        currAnnos = au.loadj0126NML(sourcefile)
        currSyns = synapses_from_j0126(currAnnos)
        listOfListOfSynObjs.append(currSyns)

    # todo insert call to generate_consensus_synapses here and process output
    # accordingly for code after this to work
    #


    # OLD CODE - replace / remove
    #
    #  flatten syn lists
    # allsyns = list(itools.chain.from_iterable(listOfListOfSynObjs))
    #
    # # prepare spatial lookups, just put all synapses into the KDtree
    # synTree = synapsesToKDtree(allsyns)
    #
    # synsWithPartners = {}
    # redNotMet = {}
    #
    # # query all against all, the result is a list of syn lists,
    # # for each syn query
    # foundSyns = synTree.query_ball_point([syn.getAvgPrePostCoord()
    #                                       for syn in allsyns], spotlightRadius)
    # for synIndex, queryResult in enumerate(foundSyns):
    #     #if len(queryResult) > 1: # found other synapses here
    #     synsWithPartners[allsyns[synIndex]] = []
    #     for syn in queryResult:
    #         if syn.sourceUsername != allsyns[synIndex].sourceUsername:
    #             synsWithPartners[allsyns[synIndex]].append(syn)



    for syn in synsWithPartners.keys():
        currAnnotators = {}
        #print 'this syn ' + syn.sourceUsername + ' pre coord: ' +\
        #    str(syn.preNodeCoord) + ' :'   + ' post coord: ' +\
        #    str(syn.postNodeCoord)
        for partnersyn in synsWithPartners[syn]:
            try:
                currAnnotators[partnersyn.sourceUsername].append(partnersyn)
            except:
                currAnnotators[partnersyn.sourceUsername] = [partnersyn]
                #print 'psyn ' + partnersyn.sourceUsername + ' d: ' +\
                #    str(partnersyn.euclDistToOtherSyn(syn)) + ' angl: ' +\
                #    str(partnersyn.angleToOtherSyn(syn)) + ' pre coord: ' +\
                #    str(partnersyn.preNodeCoord) + ' :'   + ' post coord: ' +\
                #    str(partnersyn.postNodeCoord)
                # print 'psyn angle to syn: ' +
                # str(partnersyn.angleToOtherSyn(syn)) + ' ' +
                #partnersyn.sourceUsername
                #'avg coor: ' + str(syn.getAvgPrePostCoord())

        for redAnnoUsername in currAnnotators.keys():

            #print 'len ' + str(len(currAnnotators[redAnnoUsername]))
            if len(currAnnotators[redAnnoUsername]) > 1:
                if verbose:
                    print 'Found a synapse with two partner synapses from ' \
                          'the same other annotator.'

        if len(synsWithPartners[syn]) < reqRedundancy - 1:
            try:
                redNotMet[syn.sourceUsername].append(syn)
            except:
                redNotMet[syn.sourceUsername] = [syn]


    if writeToFile:
        writeConSynsToNML(synsWithPartners, redNotMet)

    return synsWithPartners, redNotMet


def writeConSynsToNML(synsWithPartners, synsRedNotMet):
    skel = au.genj0126SkelObj()

    currNodeID = 1
    currAnnoID = 1

    for srcUsername in synsRedNotMet.keys():
        # create anno
        currAnno = ns.SkeletonAnnotation()
        currAnno.annotation_ID = currAnnoID
        currAnno.comment = srcUsername

        for syn in synsRedNotMet[srcUsername]:
            currNodeID += 1
            # create nodes with comments for each synapse
            newNodePre = ns.SkeletonNode()
            newNodePre.from_scratch(currAnno, syn.preNodeCoord[0],
                syn.preNodeCoord[1],
                syn.preNodeCoord[2], inVp=1,
                inMag=1, time=0,
                ID=currNodeID, radius=1.0)
            comment = srcUsername + '-p4'
            newNodePre.setPureComment(comment)
            currAnno.addNode(newNodePre)

            currNodeID += 1
            newNodePost = ns.SkeletonNode()
            newNodePost.from_scratch(currAnno, syn.postNodeCoord[0],
                syn.postNodeCoord[1],
                syn.postNodeCoord[2], inVp=1,
                inMag=1, time=0, ID=currNodeID,
                radius=1.0)
            comment = srcUsername + '-px'
            newNodePost.setPureComment(comment)

            currAnno.addNode(newNodePost)
            skel.branchNodes.append(currNodeID)

        # add anno to skel
        skel.annotations.add(currAnno)
        currAnnoID += 1

    filename = path + skel.experimentName + \
               ' '.join([username for username in redNotMet.keys()]) \
               + '-mergedSynapes.nml'

    skel.toNml(filename)

    return


def synDiscrepancyAnalysis(listOfPathsToNMLs,
        spotlightRange=(5, 5000, 50),
        synapseFormat='j0126',
        task='taskA'):
    radii = range(spotlightRange[0], spotlightRange[1], spotlightRange[2])
    numWithPByRadius = []
    numLonelyByRadius = []

    for currRadius in radii:
        [synswithp, rednotmet] = synDiscrepancyHighlighter(listOfPathsToNMLs,
            spotlightRadius=currRadius)

        print 'radius: ' + str(currRadius) + ' syns without spatial match: ' + \
              str(len({x for v in rednotmet.itervalues() for x in v}))

        numLonelyByRadius.append(
            len({x for v in rednotmet.itervalues() for x in v}))

    mplot.figure()
    mplot.plot(radii, numLonelyByRadius)
    mplot.title('syns without spatial match for different spotlights')

    listOfListOfSynObjs = []
    for sourcefile in listOfPathsToNMLs:
        currAnnos = au.loadj0126NML(sourcefile)
        currSyns = synapses_from_j0126(currAnnos)
        listOfListOfSynObjs.append(currSyns)


    # flatten syn lists
    allsyns = list(itools.chain.from_iterable(listOfListOfSynObjs))
    pairwisedistancesClosest = []
    for syn in allsyns:
        thispairwise = []
        for syn2 in allsyns:
            thispairwise.append(syn.euclDistToOtherSyn(syn2))
        print 'this dist: ' + str(sorted(thispairwise)[1])
        pairwisedistancesClosest.append(sorted(thispairwise)[1])

    mplot.figure()
    mplot.hist(pairwisedistancesClosest, bins=30)
    #mplot.plot(radii, numLonelyByRadius)
    mplot.title('pairwise distances')
    #mplot.autoscale(tight=True)
    #mplot.ylim((np.median(allfwd)-15., np.median(allfwd)+15.))

    return


def synDiscrepancyHighlighterTaskB(listOfPathsToNMLs, reqRedundancy=2,
        spotlightRadius=90,
        path='', writeToFile=False):
    synsWithPartners, redNotMet = synDiscrepancyHighlighterTaskA(
        listOfPathsToNMLs, reqRedundancy, spotlightRadius, path, writeToFile)

    #if len(redNotMet.values()):
    #    raise Exception('All synapses need at least one redundant synapse
    # annotation inside the spatial spotlight radius')

    synsWithClassMismatch = []

    for syn in synsWithPartners.keys():
        # compare vesicles and density
        agreementCnt = 0
        for partnersyn in synsWithPartners[syn]:
            #if syn.vesicles == partnersyn.vesicles:
            #    agreementCnt += 1
            if syn.density == partnersyn.density:
                agreementCnt += 2
                #print 'this syn agr cnt: ' + str(agreementCnt)
            #print 'len partnersyns: ' + str(len(synsWithPartners[syn]))
        if agreementCnt < reqRedundancy:
            synsWithClassMismatch.append(syn)

    return synsWithClassMismatch
