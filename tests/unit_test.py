# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 02:27:36 2018

@author: msabih
"""
import sys
sys.path.append('../')
import numpy as np
import unittest
import h5py 
import json
from simulator.filters import(ShiftRoadFilter, ShiftLanesFilter,
                              TiltRoadFilter, LaneWidthFilter)
from simulator.colors import(color_w_constant_color, color_w_random_color,
                             color_w_constant_color_random_mean)
class TestColors():
    
    def test_color_w_constant_color(self):
        with h5py.File("unit_test_data.h5") as f:
            color = np.array(f['color_w_constant_color_color'])
            fmask = np.array(f['color_w_constant_color_fmask'])
            out = np.array(f['color_w_constant_color_out'])
            user_out = color_w_constant_color(fmask, color)
            #user_out[0][0][0] = 0
            try:
                np.testing.assert_almost_equal(out, user_out)
            except AssertionError:
                print "color_w_constant_color test has not passed" 
            else:
                print 'color_w_constant_color test has passed'
    
    def test_color_w_random_color(self):
        with h5py.File("unit_test_data.h5") as f:
            fmask = np.array(f['color_w_random_color_fmask'])
            mean = np.array(f['color_w_random_color_mean'])
            range_ = np.array(f['color_w_random_color_range'])
            out = np.array(f['color_w_random_color_out'])
            user_out = color_w_random_color(fmask, mean, range_)
            #user_out[0][0][0] = 0
            try:
                np.testing.assert_almost_equal(out, user_out)
            except AssertionError:
                print "test_color_w_random_color test has not passed"
            else:
                print 'test_color_w_random_color test has passed'
    
    def test_color_w_constant_color_random_mean(self):
        with h5py.File("unit_test_data.h5") as f:
            fmask = np.array(f['color_w_constant_color_random_mean_fmask'])
            mean = np.array(f['color_w_constant_color_random_mean_mean'])
            lb = np.array(f['color_w_constant_color_random_mean_lb'])
            ub = np.array(f['color_w_constant_color_random_mean_ub'])
            out = np.array(f['color_w_constant_color_random_mean_out'])
            user_out = color_w_constant_color_random_mean(fmask, mean, lb, ub)
            #user_out[0][0][0] = 0
            try:
                np.testing.assert_almost_equal(out, user_out)
            except AssertionError:
                print "color_w_constant_color_random_mean test has not passed"
            else:
                print 'color_w_constant_color_random_mean test has passed' 

class TestFilters():
    def __init__(self, file='filter_unit_test.json'):
        with open(file) as data_file:
            self.data = json.load(data_file)
    
    def test_shift_road_filter(self):
        filter_obj = ShiftRoadFilter(**self.data['test'][3][0])
        inp = filter_obj.filter(self.data['test'][1]['input'][0])
        out = self.data['test'][2]['output'][0]
        test_mask = []
        for ip, op in zip(inp['layer_params'], out['layer_params']):
            test_mask.append((ip == op))
            
        try:
            assert(sum(test_mask) == 7)
        except AssertionError:
            print 'ShiftRoadFilter test has not passed'
        else:
            print 'ShiftRoadFilter test has passed'
        
    def test_shift_lanes_filter(self):
        filter_obj = ShiftLanesFilter(**self.data['test'][3][1])
        inp = filter_obj.filter(self.data['test'][1]['input'][1])
        out = self.data['test'][2]['output'][1]
        test_mask = []
        for ip, op in zip(inp['layer_params'], out['layer_params']):
            test_mask.append((ip == op))
            
        try:
            assert(sum(test_mask) == 7)
        except AssertionError:
            print 'ShiftLanesFilter test has not passed' 
        else:
            print 'ShiftLanesFilter test has passed'
            
            
    def test_tilt_road_filter(self):
        filter_obj = TiltRoadFilter(**self.data['test'][3][2])
        inp = filter_obj.filter(self.data['test'][1]['input'][2])
        out = self.data['test'][2]['output'][2]
        test_mask = []
        for ip, op in zip(inp['layer_params'], out['layer_params']):
            test_mask.append((ip == op))

        try:
            assert(sum(test_mask) == 7)
        except AssertionError:
            print 'TiltRoadFilter test has not passed'
        else:
            print 'TiltRoadFilter test has  passed'
        
    def test_lane_width_filter(self):
        filter_obj = LaneWidthFilter(**self.data['test'][3][3])
        inp = filter_obj.filter(self.data['test'][1]['input'][3])
        out = self.data['test'][2]['output'][3]
        test_mask = []
        for ip, op in zip(inp['layer_params'], out['layer_params']):
            test_mask.append((ip == op))

        try:
            assert(sum(test_mask) == 7)
        except AssertionError:
            print 'LaneWidthFilter test has not passed'
        else:
            print 'LaneWidthFilter test has passed'
         
if __name__ == "__main__":
    tf = TestFilters()
    tf.test_lane_width_filter()
    tf.test_tilt_road_filter() 
    tf.test_shift_lanes_filter()
    tf.test_shift_road_filter()
    tc = TestColors()
    tc.test_color_w_constant_color()
    tc = TestColors()
    tc.test_color_w_random_color()
    tc = TestColors()
    tc.test_color_w_constant_color_random_mean()
        
