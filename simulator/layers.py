"""
This module provides implementations of some basis layer types
that together may form a simulated image.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
from functools import partial
import numpy as np

import sys
sys.path.append('../')

# User imports
from colors import COLOR_FCT_REGISTRY
from utils import *

class NotImplementedError(Exception):
    pass


class Layer():
    """
    A Layer is defined by several attributes that control the dimensions,
    its shape and how the layer is rendered. Each layer has an associated
    shape, represented by a numpy array bitmask 'fmask'
    (from {0,1}^(height x width)), e.g. [[0,1,1,0],[0,1,0,0]],
    which defines the support of the feature it represents.
    Every layer has a height and a width and a color function used to
    render it.

    The Layer class is the abstract base class for implementations of
    different layer types.
    """
    def __init__(self, height, width):
        self.dims = (height, width)

    def render(self):
        return self.color_fct(self.fmask)

    def to_point_cloud(self):
        raise NotImplementedError("Feature not implemented: Serialization of layer type %s" % (type(self).__name__,))


class BackgroundLayer(Layer):
    """
    Represents background of an image. The 'fmask' of a BackgroundLayer
    is an all-one matrix of the dimensions of the image.
    """
    def __init__(self, height, width, color_fct):
        Layer.__init__(self, height, width)

        # Initialize all-one bitmask. 
        self.fmask=np.ones((height,width)) 
        self.color_fct = partial(COLOR_FCT_REGISTRY[color_fct['type']], **color_fct['params'])

        # Bind color function to given parameters. Use a partial function. Have a look at e.g. class SkyLayer to see how it is used.


class StraightRoadLayer(Layer):
    """
    Models a straight road consisting of the road itself and at least on lane (meaning
    the lane markings). The road is defined in non-perspective coordinates (bird-view)
    and then projected into perspective view via a homography transformation. This
    transformation is defined by two arrays of coordinates [src_coords],[tgt_coords]. For
    details see for example https://en.wikipedia.org/wiki/Homography.

    All coordinates are scaled to [0;1], such that the size of the image can be easily
    changed without need to adjust the layer definition.

    For an example configuration, see config.py.

    The StraightRoadLayer is a composite layer, consisting of a RoadLayer and several
    LaneLayers. Its bitmask (fmask) is the superposition of all bitmasks of those layers.
    """
    def __init__(self, height, width, color_fcts, road, road_width, lanes, lane_widths, tilt, transform_coordinates):
        """ 
        Initializes straight road 

        Input:
        self        -- it's me
        height      -- layer height
        width       -- layer width
        color_fcts  -- array of functions used to color the road and the lanes
        road        -- [[road_bx, road_by], [road_tx, road_ty]] left boundary of the road; b: bottom, t: top
        road_width  -- width of the road
        lanes       -- [[[lane_1_bx, lane_1_by],[lane_1_tx, lane_1_ty]],...,
                         [lane_n_tx, lane_n_ty]]] left boundaries of the lanes; b: bottom, t: top
        lane_widths -- array of lane widths
        tilt        -- how much the road is tilted
        transform_coordinates
                    -- array of coordinates defining the homography projection
        """
        Layer.__init__(self, height, width)

        # Initialize right border of the road from given left boundary and width
        
        
        
        right_boundary=[[x,y+road_width] for (x,y) in road[::-1]]

        #print("left_bound",road)
        #print("width",road_width)
        #print("right_bound",right_boundary)
        # and complete the set of coordinates to define a polygon representing the road
        coords =road+right_boundary
        # For each pair of coordinates representing the left border of a lane
        # complete the lane polygon
        all_lanes=[]
        for i in range(len(lanes)):
        
            right_boundary=[[x,y+lane_widths[i]] for (x,y) in lanes[i][::-1]]
            all_lanes.append(lanes[i]+right_boundary)
           
        

        # Initialize homography transform (see simulator/utils.py)
        tform=init_transform(**transform_coordinates)
        

        # Intialize self.road_layer. You can find the subclass below.
        
        self.road_layer=RoadLayer(height,width,coords,tform,color_fcts[0])
        self.lane_layers=[]
        
        # Initialize each lane layer and store a reference in self.lane_layers. You can find the subclass below.
       
        for i,lanes in enumerate(all_lanes):
            self.lane_layers.append(LaneLayer(height,width,lanes,tform,color_fcts[i+1]))
            

        # Merge bitmasks (see helper function in this file) and store it in self.fmask
        masks=[]
        road_fmask=[self.road_layer.fmask]
        lane_fmask=[lane_layers.fmask for lane_layers in self.lane_layers]
        
        
        self.fmask=merge_masks(road_fmask+lane_fmask)

    def render(self):
        # Merge road and lane sublayers and render them (see simulator/utils.py)
        layers=[self.road_layer]+self.lane_layers
        img=merge_layers(layers)
        return img

    def to_point_cloud(self):
        """
        Serialize road layer to list of indizes of on-lane pixels
        
        The 5 output dimensions are intended for
        1. x coordinate of pixel
        2. y coordinate of pixel
        3. time index (for extension to 3D applications): always -1
        4. confidence (for compatibility with net output): always 1
        5. lane_id: Corresponds to ordering in self.lane_layers list

        Output:
        point_cloud -- [[vertical_idx],[horizontal_idx],[time],[confidence],[lane_id]]
                       meaning here: [[y],[x],[-1],[1],[lane_id]]
        """

        # Initialize output data structure
        point_cloud = np.ndarray((5,0), dtype=np.int32)

        # For each lane layer:
        #   Transform its bitmask to a point cloud
        #   Append the result to the output
        for (idx, lane_layer) in enumerate(self.lane_layers):
            #Make use of below function fmask_to_point_cloud in order to get variables with the indices of on-lane pixels
            [y,x]=fmask_to_point_cloud(lane_layer.fmask)
            #Conversion to required shape
            point_cloud = np.concatenate((
                    point_cloud,
                    np.vstack((
                        y,
                        x,
                        np.full(y.shape, -1, dtype=np.int32),
                        np.ones(y.shape, dtype=np.int32),
                        np.full(y.shape, idx, dtype=np.int32)
                    ))
                ), axis=1)
                    
        return point_cloud


class RoadLayer(Layer):
    """
    Layer representing the road itself, without lanes.
    """
    def __init__(self, height, width, road_coords, tform, color_fct):
        """
        Initialize RoadLayer

        Input:
        height      -- layer height
        width       -- layer width
        road_coords -- coordinates defining the road polygon
        tform       -- transform used to project the road to perspective view
        color_fct   -- function to color layer
        """

        # Complete road coordinates to form a polygon
        road = road_coords[:]
        if road[0][0] != road[-1][0] or road[0][1] != road[-1][1]:
            road.append(road[0])

        # Project coordinates to perspective view
        proj_coords = project(road_coords, tform)

        # Initialize bitmask via draw_polygon function
        self.fmask = draw_polygon(height, width, proj_coords)

        # Bind color function to input parameters
        self.color_fct = partial(COLOR_FCT_REGISTRY[color_fct['type']], **color_fct['params'])


class LaneLayer(Layer):
    """
    Layer representing the lanes/lane markings on a road.
    """
    def __init__(self, height, width, lane_coords, tform, color_fct):
        """ 
        Initialize LaneLayer

        Input:
        height      -- layer height
        width       -- layer width
        lane_coords -- coordinates defining the lane polygons
        tform       -- transform used to project the lanes to perspective view
        color_fct   -- function to color layer
        """

        # Complete lane coordinates to form a polygon. You may use the function project in simulator/utils.py. Tipp: Have a look at class RoadLayer.
        lane=lane_coords[:]
        if lane[0][0] != lane[-1][0] or lane[0][1] != lane[-1][1]:
            lane.append(lane[0])
        proj_coords = project(lane_coords, tform)
        # Initialize bitmask via draw_polygon function
        self.fmask = draw_polygon(height, width, proj_coords)
        # Bind color function to input parameters
        self.color_fct = partial(COLOR_FCT_REGISTRY[color_fct['type']], **color_fct['params'])

class SkyLayer(Layer):
    """
    This layer type is intended to add a sky-like polygon to the image. However, there is no
    implementation-wise limitation to sky-like shapes, all sorts of polygons can be added to
    the image via this layer.
    """
    def __init__(self, height, width, color_fct, shape):
        Layer.__init__(self, height, width)
        self.fmask = self._make_fmask(height, width, shape)
        self.color_fct = partial(COLOR_FCT_REGISTRY[color_fct['type']], **color_fct['params'])

    def _make_fmask(self, height, width, shape):
        return draw_polygon(height, width, shape)


# Helper functions
"""
Merge several bitmasks to one
"""
def merge_masks(masks):
    masks = list(masks)
    tgt_mask = np.zeros(masks[0].shape, dtype=np.uint8)

    for (idx, mask) in enumerate(masks, start=1):
        tgt_mask[mask>0] = idx

    return tgt_mask


"""
Transform a bitmask to a point cloud, i.e. list of indices of non-zero elements

Input:
fmask -- bitmask, np.array

Output:
point cloud, tuple of arrays containing the indices of non-zero components of 'fmask'
"""
def fmask_to_point_cloud(fmask):
    return np.nonzero(fmask)


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
LAYER_REGISTRY = {
    'BackgroundLayer'         : BackgroundLayer,
    'SkyLayer'                : SkyLayer,
    'StraightRoadLayer'       : StraightRoadLayer
}


