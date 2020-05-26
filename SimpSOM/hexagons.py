"""
Hexagonal tiling library

F. Comitani @2017 
O. Pellicer @2020
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


def coorToHex(x,y):

    """Convert Cartesian coordinates to hexagonal tiling coordinates.

        Args:
            x (float): position along the x-axis of Cartesian coordinates.
            y (float): position along the y-axis of Cartesian coordinates.
            
        Returns:
            array: a 2d array containing the coordinates in the new space.
                
    """


    newy=y*2/np.sqrt(3)*3/4
    newx=x
    if y%2: newx+=0.5
    return [newx,newy]    
    

def plot_hex(fig, centers, weights, edges=[], cmap='viridis', radius=0.95, ax=None):
    
    """Plot an hexagonal grid based on the nodes positions and color the tiles
       according to their weights.

        Args:
            fig (matplotlib figure object): the figure on which the hexagonal grid will be plotted.
            centers (list, float): array containing couples of coordinates for each cell 
                to be plotted in the Hexagonal tiling space.
            weights (list, float): array contaning informations on the weigths of each cell, 
                to be plotted as colors.
            
        Returns:
            ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.
                
    """

    ax = fig.add_subplot(111, aspect='equal') if ax == None else ax

    xpoints = [x[0]  for x in centers]
    ypoints = [x[1]  for x in centers]
    patches = []

    weights_are_colors= any(isinstance(el, list) for el in weights) and len(weights[0])==3
    
    if weights_are_colors:
        for x,y,w in zip(xpoints,ypoints,weights):
            hexagon = RegularPolygon((x,y), numVertices=6, radius=radius/np.sqrt(3), 
                                orientation=np.radians(0), 
                                facecolor=w)
            ax.add_patch(hexagon)
        p = PatchCollection(patches)
        
    else:
        cmap = plt.get_cmap(cmap)
        for x,y,w in zip(xpoints,ypoints,weights):
            hexagon = RegularPolygon((x,y), numVertices=6, radius=radius/np.sqrt(3), 
                                orientation=np.radians(0))
            patches.append(hexagon) 
        p = PatchCollection(patches, cmap=cmap, 
                            edgecolors= [cmap(e) for e in edges] if edges!=[] else None, 
                            linewidths= 5 if edges!=[] else None)
        
    p.set_array(np.array(weights))
    ax.add_collection(p)
        
    ax.axis('off')
    ax.autoscale_view()
    
    return ax