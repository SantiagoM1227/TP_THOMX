#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:23:11 2025

@author: operateur.thomx
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def Profiling( path, plane = 0, region = [0,-1]):
        pixel_size = 13.5e-6 #m
    
        image = Image.open(path)
        image = np.array(image, dtype = 'float64')
        
        profile_on_axis = np.sum(image, axis = plane)
        profile_on_axis = profile_on_axis[region[0]:region[1]]
        profile_on_axis = profile_on_axis/np.max(profile_on_axis)
        
        window_size = 10
        kernel = np.ones(window_size)/window_size
        smoothed = np.convolve(profile_on_axis, kernel, mode = "same")
        derivative = np.gradient(smoothed, pixel_size)
        derivative = np.abs(derivative)
        
        
        fig, axes = plt.subplots(1,3, figsize = (15,5))
        axes[0].imshow(image, cmap = 'gray', aspect = 'auto')
        axes[0].set_title('Camera')
        
        str_plane = ['x','y'][plane]
        
        axes[1].plot(profile_on_axis)
        axes[1].set_title('Profile on axis '+ str_plane)
        
        axes[2].plot(derivative[10:-10])
        axes[2].set_title('Derivative on axis '+ str_plane)
        
        
def path(n):
    
    path = "/data/shared/Commissioning_tools/Mesure/20251121/images/20251121_fente1000{}.tif".format(n)
    return path 


Profiling(path(0), plane = 0, region = [500, 1500])


Profiling(path(0), plane = 0, region = [1800, 2800])



Profiling(path(0), plane = 1, region = [500, 1500])


Profiling(path(0), plane = 1, region = [3000, 3800])