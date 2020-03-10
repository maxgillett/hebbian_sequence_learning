import numpy as np
import matplotlib

def overlap_colors(n, l=0.1, r=0.7):
    cmap = matplotlib.cm.get_cmap("bone")
    colors = [cmap(x) for x in np.linspace(l,r,n)]
    return colors

def single_unit_colors():
    colors = ['indianred','salmon','darkred','firebrick','orangered']
    return colors

def single_unit_colors2():
    colors = ['orange', 'sandybrown', 'mediumpurple', 'lightsalmon', 'goldenrod',
              'gold', 'brown', 'royalblue', 'plum', 'orchid']
    return colors

def single_unit_colors_rainbow(n, l=0.0, r=1.0):
    cmap = matplotlib.cm.get_cmap("nipy_spectral")
    colors = [cmap(x) for x in np.linspace(l,r,n)]
    return colors
