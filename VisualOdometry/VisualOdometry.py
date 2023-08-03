import cv2
import numpy as np
from Ultility import Transform
from Ultility import NormalizePoint

class VisualOdometry():
    """
    Visual Odometry Class

    General Idea: Read and store first frame,feature,descriptor -> Read N next frame to compute T(1,N), where the base frame 1 is the first frame.
    Algorithm: 
        Turn input frame to gray -> compute keyfeature -> compute descriptor for keyfeature - 
        > match the keyfeature between two frame -> compute pose from essential matrix -
        > compute T(1,N) with T(1,N-1)@T(N-1,N) -> store frame,feature,descriptor for next iteration

    Init_Attributes:
        _camera (Camera): This describe the intrinsic property of camera (init_Pose, resolution, camera_Matrix, distortion_coefficients, rectification_matrix, projection_matrix)
        _feature_detector: Any feature detector algorithm, as long as it satisfy this 
            API: feature_detector(grayImage,Mask) -> keyfeature: tuple[cv2.KeyPoint]
        _descriptors_computer: Any descriptors computer algorithm as long as 
            API: descriptors_computer(grayImage, keyfeature) -> keyfeature: tuple[cv2.KeyPoint], descriptor: numpy.ndarray[numpy.ndarray]
        _matcher: 
            API: matcher(keyfeature_frame_1,keyfeature_frame_2) -> tuple[cv2.DMatch]
        _findEssentialMat:
            API: findEssentialMat(p1,p2,Camera_Matrix) -> essential_mat, mask
        _isDebug: If is set to False, debug image would not be compute nor store (speedup)

    Attributes:
        public:
            Odom: Data structure for storing Odometry, it only have two attributes, pose_tf(homogenous pose) and timestamp
            debugFrame: ONLY USABLE IF isDebug == True, this stores two image for debugging
                debugFrame[0]: Image with Keypoint draw on Image
                debugFrame[1]: Image with Matched keypoint between current frame and previous frame.
        private:
            _tf: transformation class from Ultility.py, use for converting between euclidean and homogenous coordinate
            _frame_prev: store previous frame
            _keypoints_prev: store previous keyfeature
            _keypoints_curr: store current keyfeature
            _descriptors_prev: store previous keyfeature
            _descriptors_curr: store current keyfeature
            _normalize: normalization class from Ultility.py, not used yet, not sure if it is required.


    Methods:
        computeAndUpdateOdom(frame, time): Compute and update odometry, please use this before reading Odom.

    ToDo: Bundle Adjustment
    """
    def __init__(self, camera_object, feature_detector, descriptors_computer, matcher, findEssentialMat, isDebug=False):
            self.Odom = Odometry(camera_object.init_Pose,0)
            self._tf = Transform()
            self._camera = camera_object
            self._feature_detector = feature_detector
            self._descriptors_computer = descriptors_computer
            self._matcher = matcher
            self._findEssentialMat = findEssentialMat 

            self._frame_prev=None

            self._keypoints_prev=[]
            self._keypoints_curr=[]
            self._descriptors_prev=[]
            self._descriptors_curr=[]

            self._normalize = NormalizePoint()
            self._isDebug = isDebug
            if isDebug:
                self.debugFrame=[np.zeros((1,1)),np.zeros((1,1))]

    def computeAndUpdateOdom(self,frame,time):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._keypoints_curr = self._feature_detector(gray,self._camera.mask)
        self._keypoints_curr, self._descriptors_curr = self._descriptors_computer(gray, self._keypoints_curr)

        if len(self._keypoints_prev) > 0:
            # Match keypoints and compute essential matrix
            matches = self._matcher(self._descriptors_prev, self._descriptors_curr)

            # temp = []
            # for m in matches:
            #     if m.distance < 30:
            #         temp.append(m)
            #     #else:
            #         #print(m.distance)
            # matches=temp
            # p1 = self._normalize.UndistorPoint(self._keypoints_prev,[m.queryIdx for m in matches],self._camera)
            # p2 = self._normalize.UndistorPoint(self._keypoints_curr,[m.trainIdx for m in matches],self._camera)
            essential_mat, mask = self._findEssentialMat(
                np.float32([self._keypoints_prev[m.queryIdx].pt for m in matches]),
                np.float32([self._keypoints_curr[m.trainIdx].pt for m in matches]),
                camera_Matrix=self._camera.camera_Matrix
            )
            
            _, R, t, _ = cv2.recoverPose(essential_mat, 
                                        np.float32([self._keypoints_prev[m.queryIdx].pt for m in matches]),
                                        np.float32([self._keypoints_curr[m.trainIdx].pt for m in matches]),
                                        self._camera.camera_Matrix)

            # T(1,k) = T(1,n) @ T(n,k)
            self.Odom.pose_tf = (self.Odom.pose_tf)@(self._tf.homogenousCoordinate(np.squeeze(t),R))

            if self._isDebug:
                #Draw Keypoint only on current image
                self.debugFrame[0] = cv2.drawKeypoints(frame, self._keypoints_prev, None, color=(0,255,0), flags=0)
                #Draw matching point between current and previous image
                self.debugFrame[1] = cv2.drawMatches(self._frame_prev,self._keypoints_prev,frame,self._keypoints_curr,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
        self._frame_prev = frame

        self._keypoints_prev = self._keypoints_curr
        self._descriptors_prev = self._descriptors_curr

class Odometry():
    def __init__ (self,pose_tf,timestamp):#,speed,angular_speed,timestamp):
        self.pose_tf = pose_tf
        self.timestamp = timestamp
    #    self.speed = speed
    #    self.angular_speed = angular_speed
    def getPoseByEuclidean(self):
        return Transform().EuclideanCoordinate(self.pose_tf)
    def getPoseByHomogenous(self):
        return self.pose_tf



