"""
This module contains filters that can be used to randomize the configuration
of single layers. For example, using the TiltRoadFilter, the tilt of the road
(and the lanes) of a StraightRoadLayer can be varied randomly.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import random

class ConfigFilter():
    """
    Base class for config-randomizing filters

    Every ConfigFilter has the ability to filter a given layer configuration,
    modifying parameters in a pseudo-random fashion.
    """
    def filter(self, config):
        return config


class TiltRoadFilter(ConfigFilter):
    """
    Tilts a straight road uniformly within [lb;ub].
    """
    def __init__(self, lb, ub):
        # Draw a tilt uniformly from [lb;ub] and save it as attribute
        self.lb=lb
        self.ub=ub
        self.tilt=random.uniform(self.lb,self.ub)
        pass
        
    def filter(self, config):
        # Use drawn tilt to tilt road and lanes by modifying the coordinates
        # defined in the 'config'
        tilt = self.tilt
        road = config['layer_params']['road']
        lanes = config['layer_params']['lanes']
        
        config['layer_params']['road'] = [[x,(1-x)*tilt+y] for [x,y] in road]
        config['layer_params']['lanes'] = [[[x,(1-x)*tilt+y] for [x,y] in lane] for lane in lanes]    
        
        return config


class ShiftRoadFilter(ConfigFilter):
    """
    Shifts a straight road within [lb;ub].
    """
    def __init__(self, lb, ub):
        # Draw a shift uniformly from [lb;ub] and save it as attribute
        self.lb=lb
        self.ub=ub
        self.shift=random.uniform(self.lb,self.ub)
        
        pass
            
    def filter(self, config):
        # Use drawn shift to shift road and lanes by modifying the coordinates
        # defined in the 'config'
        shift = self.shift
        road = config['layer_params']['road']
        lanes = config['layer_params']['lanes']
        
        config['layer_params']['road'] = [[x,shift+y] for [x,y] in road]
        config['layer_params']['lanes'] = [[[x,shift+y] for [x,y] in lane] for lane in lanes]
       
        return config


class ShiftLanesFilter(ConfigFilter):
    """
    Shifts lanes horizontally within [x-lb;x+ub].
    """
    def __init__(self, lb, ub):
        # Save lower and upper bound as attributes
        self.lb=lb
        self.ub=ub
        pass
        
    def filter(self, config):
        # For each lanes configuration in 'config'
        #   Draw a random shift from [lb;ub]
        #   Shift lane
        shift=random.uniform(self.lb,self.ub)
        lanes = config['layer_params']['lanes']
        config['layer_params']['lanes'] = [[[x,shift+y] for [x,y] in lane] for lane in lanes]
       
        return config


class LaneWidthFilter(ConfigFilter):
    """
    Varies lane width within [width-lb;width+ub].
    """
    def __init__(self, lb, ub):
        # Draw random delta_width from [lb;ub] and save it
        self.lb=lb
        self.ub=ub
        self.delta_width=random.uniform(self.lb,self.ub)
        pass

    def filter(self, config):
        # Use drawn delta_width to modify lane widths
        delta_width=self.delta_width
        lanes = config['layer_params']['lane_widths']
        config['layer_params']['lane_widths'] = [x+delta_width for x in lanes]
       
        return config


class ConstantColorFilter(ConfigFilter):
    """
    Picks random color from ([r-dr;r+dr],[g-dg;g+dg],[b-db;b+db]) to vary color
    of constant color function.
    """
    def __init__(self, dr, dg, db):
        # Draw delta_r/g/b and save them
        self.dist_r = random.randint(-1*dr, dr)
        self.dist_g = random.randint(-1*dg, dg)
        self.dist_b = random.randint(-1*db, db)

    def filter(self, config):
        # Modify color defined in the config
        config['layer_params']['color_fct']['params']['color'][0] += self.dist_r
        config['layer_params']['color_fct']['params']['color'][1] += self.dist_g
        config['layer_params']['color_fct']['params']['color'][2] += self.dist_b

        return config


class RandomColorMeanFilter(ConfigFilter):
    """
    Picks random color from ([r-dr;r+dr],[g-dg;g+dg],[b-db;b+db]) to vary mean
    of random color function.
    """
    def __init__(self, dr, dg, db):
        # Draw delta_r/g/b and save them
        self.dist_r = random.randint(-1*dr, dr)
        self.dist_g = random.randint(-1*dg, dg)
        self.dist_b = random.randint(-1*db, db)

    def filter(self, config):
        # Modify color defined in the config
        config['layer_params']['color_fct']['params']['mean'][0] += self.dist_r
        config['layer_params']['color_fct']['params']['mean'][1] += self.dist_g
        config['layer_params']['color_fct']['params']['mean'][2] += self.dist_b

        return config


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
CONFIG_FILTER_REGISTRY = {
    'ShiftRoadFilter'       : ShiftRoadFilter,
    'ShiftLanesFilter'      : ShiftLanesFilter,
    'TiltRoadFilter'        : TiltRoadFilter,
    'LaneWidthFilter'       : LaneWidthFilter,
    'ConstantColorFilter'   : ConstantColorFilter,
    'RandomColorMeanFilter' : RandomColorMeanFilter
}


