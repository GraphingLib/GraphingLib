from matplotlib.patches import Polygon, Rectangle
from numpy import array


def histogram_legend_artist(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    xy = array([[0,0,1,1,2,2,3,3,4,4,0], [0,4,4,2.5,2.5,5,5,1.5,1.5,0,0]]).T
    xy[:,0] = width * xy[:,0] / 4 + xdescent
    xy[:,1] = height * xy[:,1] / 5 - ydescent
    patch = Polygon(xy)
    return patch


def hlines_legend_artist(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    xy = (0, 0)
    # xy[:,0] = width * xy[:,0] / 4 + xdescent
    # xy[:,1] = height * xy[:,1] / 5 - ydescent
    patch = Rectangle(xy)
    return patch