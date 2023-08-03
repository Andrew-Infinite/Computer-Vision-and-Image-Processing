import cv2
import Ultility
import numpy as np

#Constant
"""
The preprocess was not wrap in a func is because I wish that we can see and change the setting here directly,
instead of going into function everytime config,
Algorithm: dilate -> bitwise_not
    dilate was done to expand the mask, to remove feature around the edge of the car
    bitwise_not was done, because the mask provided have true for car, and false for other region.
    We want to remove the car, so it should be false for car, true for other region.
"""
Front_Cam = Ultility.Camera(
    init_Pose = np.eye(4),
    resolution = [1440,928],
    camera_Matrix = [[661.949026684, 0.0, 720.264314891], [0.0, 662.758817961, 464.188882538], [0.0, 0.0, 1.0]],
    distortion_coefficients = [-0.0309599861474, 0.0195100168293, -0.0454086703952, 0.0244895806953],
    rectification_matrix = [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
    projection_matrix = [[539.36, 0.0, 721.262, 0.0], [0.0, 540.02, 464.54, 0.0], [0.0, 0.0, 1.0, 0.0]],
    mask = cv2.bitwise_not(
            cv2.dilate(
                cv2.imread('.\camera_info\lucid_cameras_x00\gige_100_f_hdr_mask.png', cv2.IMREAD_GRAYSCALE), 
                np.ones((5,5), dtype=np.uint8)
        )
    )
)
#property is incorrect, just to demostrate how to use this thing
Back_Cam = Ultility.Camera(
    init_Pose = np.eye(4),
    resolution = [1440,928],
    camera_Matrix = [[661.949026684, 0.0, 720.264314891], [0.0, 662.758817961, 464.188882538], [0.0, 0.0, 1.0]],
    distortion_coefficients = [-0.0309599861474, 0.0195100168293, -0.0454086703952, 0.0244895806953],
    rectification_matrix = [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
    projection_matrix = [[539.36, 0.0, 721.262, 0.0], [0.0, 540.02, 464.54, 0.0], [0.0, 0.0, 1.0, 0.0]],
    mask = cv2.erode(
            cv2.bitwise_not(cv2.imread('.\camera_info\lucid_cameras_x00\gige_100_f_hdr_mask.png', cv2.IMREAD_GRAYSCALE)), 
        np.ones((5,5), dtype=np.uint8)
    )
)

"""
Feature detector:
FAST,KLT

Descriptor computer:
BRIEF,LBP,HOG,DAISY

Detect and compute
ORB,SIFT,FREAK,AKAZE,SURF,BRISK

example: 
    feature_finder = cv2.FastFeatureDetector_create().detect
    descriptors_computer = cv2.xfeatures2d.BriefDescriptorExtractor_create().compute
"""
#feature
detectAndCompute = cv2.ORB_create(nfeatures=3000, scaleFactor=1.5, nlevels=8)
feature_finder = detectAndCompute.detect
descriptors_computer = detectAndCompute.compute
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match
findEssentialMat = lambda p1,p2,camera_Matrix: cv2.findEssentialMat(
                p1,p2,cameraMatrix=camera_Matrix, method=cv2.RANSAC, prob=0.9999, threshold=3
            )

#Flag
isDebug = True
VO_Debug = True

#True off all Debug, if isDebug is false here
if not isDebug:
    #all variable in here should set to False
    VO_Debug = False