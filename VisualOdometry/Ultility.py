import numpy as np
import cv2
class Transform():
    def __init__(self):
        pass
    def homogenousCoordinate(self,Translation,Rotation):
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, 3] = Translation
        homogeneous_matrix[:3, :3] = Rotation
        return homogeneous_matrix
    def EuclideanCoordinate(self,homogeneous_matrix):
        position = homogeneous_matrix[:3, 3]
        rotation = homogeneous_matrix[:3, :3]
        return position,rotation

class NormalizePoint():
    def __init__(self):
        pass
    def UndistorPoint(self,points,index,Camera):
        return [np.squeeze(cv2.undistortPoints(points[i].pt,Camera.camera_Matrix,Camera.distortion_coefficients)) for i in index]


class Camera():
    def __init__(self,yaml=None,init_Pose=None,resolution=None,camera_Matrix=None,distortion_coefficients=None,rectification_matrix=None,projection_matrix=None,mask=None):
        if yaml==None:
            self.init_Pose = np.array(init_Pose)
            self.resolution = np.array(resolution)
            self.camera_Matrix = np.array(camera_Matrix)
            self.distortion_coefficients = np.array(distortion_coefficients)
            self.rectification_matrix = np.array(rectification_matrix)
            self.projection_matrix = np.array(projection_matrix)
        else:
            """ToDo YAML file reading"""
            pass
        self.mask = mask
