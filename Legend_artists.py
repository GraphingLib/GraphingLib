from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from numpy import array


def histogram_legend_artist(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    xy = array([[0,0,1,1,2,2,3,3,4,4,0], [0,4,4,2.5,2.5,5,5,1.5,1.5,0,0]]).T
    xy[:,0] = width * xy[:,0] / 4 + xdescent
    xy[:,1] = height * xy[:,1] / 5 - ydescent
    patch = Polygon(xy)
    return patch


# def hlines_legend_artist(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
#     segs = array([[[0,1],[3,1]],[[0,2],[3,2]],[[0,3],[3,3]]])
#     segs[:,:,0] = width * segs[:,:,0] / 4 + xdescent
#     segs[:,:,1] = height * segs[:,:,1] / 4 + ydescent
#     collection = LineCollection(segs)
#     return collection
