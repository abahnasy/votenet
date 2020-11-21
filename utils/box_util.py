# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Jul 2019
"""
from __future__ import print_function

import numpy as np
from typing import Tuple, List, Dict
from scipy.spatial import ConvexHull

from pyquaternion import Quaternion

# ABAHNASY

class Box:
    """Simple data class representing a 3d box including, label, score and velocity."""

    def __init__(
        self,
        center: (List[float], Tuple[float]), # TODO: rename to centroid
        size: (List[float], Tuple[float]),
        heading_angle: float,
        # orientation: Quaternion,
        label: int = np.nan # TODO: rename to classlabel
        # score: float = np.nan,
        # velocity: Tuple = (np.nan, np.nan, np.nan),
        # name: str = None,
        # token: str = None,
    ):
        if np.any(np.isnan(center)):
            raise ValueError("Center coordinates should not have NaN values but we got {}".format(center))

        if np.any(np.isnan(size)):
            raise ValueError("Size values should not have NaN values but we got {}".format(size))

        if len(center) != 3:
            raise ValueError("Center should be defined by 3 numbers but we got {}".format(center))

        if len(size) != 3:
            raise ValueError("Size should be defined by 3 numbers but we got {}".format(size))
        
        if np.isnan(label):
            raise ValueError("Label should be valid but we got {}".format(label))

        self.center = np.array(center) # x,y,z respectively
        self.lwh = np.array(size) # length: x, width: y, height: z
        self.heading_angle = heading_angle
        self.orientation = Quaternion(axis=(0.0, 0.0, 1.0), radians=heading_angle)
        self.label = int(label) if not np.isnan(label) else label
        # rotation around z axis
        
        # self.score = float(score) if not np.isnan(score) else score
        # self.velocity = np.array(velocity)
        # self.name = name
        # self.token = token
        

    
    def __repr__(self):
        repr_str = (
            "label: {}, xyz: [{:.2f}, {:.2f}, {:.2f}], lwh: [{:.2f}, {:.2f}, {:.2f}], "
            "rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}"
        )

        return repr_str.format(
            self.label,
            self.center[0],
            self.center[1],
            self.center[2],
            self.lwh[0],
            self.lwh[1],
            self.lwh[2],
            self.orientation.axis[0],
            self.orientation.axis[1],
            self.orientation.axis[2],
            self.orientation.degrees,
            self.orientation.radians,
        )
    def __eq__(slef):
        pass # TODO

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return a rotation matrix.

        Returns: <np.float: 3, 3>. The box's rotation matrix.

        """
        return self.orientation.rotation_matrix
    
    def corners(self) -> np.ndarray: # TODO
        """Returns the bounding box corners.

        Args:

        Returns: First four corners are the ones facing forward.
                The last four are the ones facing backwards.

        """

        length, width, height = self.lwh

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # # Rotate
        corners = np.dot(self.rotation_matrix, corners)

        # # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners
        
def get_corners_from_labels_array(label, wlh_factor: float = 1.0) -> np.ndarray:
    ''' takes 1x8 array contains label information
    Args:
        np.array 1x8 contains label information [x, y, z, l, w, h, heading, labels]
    Returns:
    '''
    
    
    length = label[3] * wlh_factor
    width = label[4] * wlh_factor
    height = label[5] * wlh_factor
    
    
    
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))
    
    orientation = Quaternion(axis=(0.0, 0.0, 1.0), radians=label[6])
    
    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)
    
    # Translate
    x, y, z = label[0], label[1], label[2]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z
    
    return corners

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
    all zeros) and normalize=False

    Args:
        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
        normalize: Whether to normalize the remaining coordinate (along the third axis).

    Returns: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.

    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

# END ABAHNASY

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def box2d_iou(box1, box2):
    ''' Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1':box1[0], 'y1':box1[1], 'x2':box1[2], 'y2':box1[3]}, \
        {'x1':box2[0], 'y1':box2[1], 'x2':box2[2], 'y2':box2[3]})

# -----------------------------------------------------------
# Convert from box parameters to 
# -----------------------------------------------------------
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
        similar to the membr function in Box class
        return 8x3 array
    '''
    # R = roty(heading_angle)
    # l,w,h = box_size
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    # corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # corners_3d[0,:] = corners_3d[0,:] + center[0];
    # corners_3d[1,:] = corners_3d[1,:] + center[1];
    # corners_3d[2,:] = corners_3d[2,:] + center[2];
    # corners_3d = np.transpose(corners_3d)
    # return corners_3d
    length, width, height = box_size

    orientation = Quaternion(axis=(0.0, 0.0, 1.0), radians=heading_angle)

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)

    # # Translate
    x, y, z = center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return np.transpose(corners)

def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[...,0], -1) # [x1,...,xn,1]
    w = np.expand_dims(box_size[...,1], -1)
    h = np.expand_dims(box_size[...,2], -1)
    corners_3d = np.zeros(tuple(list(input_shape)+[8,3]))
    corners_3d[...,:,0] = np.concatenate((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), -1)
    corners_3d[...,:,1] = np.concatenate((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), -1)
    corners_3d[...,:,2] = np.concatenate((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d

if __name__=='__main__':

    # Function for polygon ploting
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    def plot_polys(plist,scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p)/scale, True)
            patches.append(poly)

    pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)
    colors = 100*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.show()
 
    # Demo on ConvexHull
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    print(('Hull area: ', hull.volume))
    for simplex in hull.simplices:
        print(simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0,0),(300,0),(300,300),(0,300)]
    clip_poly = [(150,150),(300,300),(150,450),(0,300)] 
    inter_poly = polygon_clip(sub_poly, clip_poly)
    print(poly_area(np.array(inter_poly)[:,0], np.array(inter_poly)[:,1]))
    
    # Test convex hull interaction function
    rect1 = [(50,0),(50,300),(300,300),(300,0)]
    rect2 = [(150,150),(300,300),(150,450),(0,300)] 
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
    if inter is not None:
        print(poly_area(np.array(inter)[:,0], np.array(inter)[:,1]))
    
    print('------------------')
    rect1 = [(0.30026005199835404, 8.9408694211408424), \
             (-1.1571105364358421, 9.4686676477075533), \
             (0.1777082043006144, 13.154404877812102), \
             (1.6350787927348105, 12.626606651245391)]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [(0.23908745901608636, 8.8551095691132886), \
             (-1.2771419487733995, 9.4269062966181956), \
             (0.13138836963152717, 13.161896351296868), \
             (1.647617777421013, 12.590099623791961)]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
