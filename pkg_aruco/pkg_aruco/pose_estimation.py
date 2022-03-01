import builtins
from numpy.core.records import array
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data

import cv2
from cv_bridge import CvBridge

import numpy as np
import math

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose

"-----Dependencies-------"
"Koden krever at markerboard er i FOL for kamera før den kan kjøres"
"minerva/minerva_description/urdf/minerva_sensors.xacro"
#Må inneholde et kamera plassert  <origin xyz="0 0 0" rpy="0 1.57079633 0"/>

"kommando for å kjøre noden"
"ros2 run pkg_aruco node_aruco"



def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])	 

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])

    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def R2quat(R):
    """
    R2quat(H) computes the unit-quaternion given the rotation matrix R
    """
    w = np.sqrt( (1 + R[0][0] + R[1][1] + R[2][2])/2 )
    x = (R[2][1] - R[1][2])/(4*w)
    y = (R[0][2] - R[2][0])/(4*w)
    z = (R[1][0] - R[0][1])/(4*w)
    return np.array([[w, x, y, z]]).T



"homogenious transform from world frame to marker"

H_w2m = np.array(
                 [[1 , 0 , 0, 9.3025],
                  [ 0 , 1 , 0, 1.72325 ],
                  [ 0 , 0 , 1, -93.3 ],
                  [0 , 0, 0, 1]]

)


rot_body2world= np.array(
                          [[0 , -1 , 0],
                          [ 1 , 0 , 0 ],
                          [ 0 , 0 , 1]]  
                        )

class ArucoNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('node_aruco')
        self.bridge = CvBridge()

        

        self.camera_subs = self.create_subscription( 
                            Image, 
                            '/minerva/camera/image_raw',
                            self.image_callback, 
                            1)

        
        self.board_pub = self.create_publisher(
                            PoseArray,
                            '/minerva/camera/pose',
                            1)






    def image_callback(self, msg):
        
        dist_coef = np.asarray([0.0,0.0,0.0,0.0])

        camera_matrix = np.array([ [407.064613,  0.,         384.5],
                                    [0.,        407.064613,  246.5],
                                    [0.,        0.,         1]] 
        )

        aruco_dict  = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_param = cv2.aruco.DetectorParameters_create()

        cv_image = self.bridge.imgmsg_to_cv2(msg,'rgb8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        marker_lenght = 0.195
        marker_seperation = 0.105
        

        rvec=None
        tvec=None
        
        board_aruco =  cv2.aruco.GridBoard_create(5,1,marker_lenght,marker_seperation, aruco_dict,1)
        
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_param)
        if ids is not None:

            marker_detected, rvec1, tvec1 = cv2.aruco.estimatePoseBoard(corners, ids,board_aruco ,camera_matrix,dist_coef,rvec, tvec)
            if marker_detected != 0:
                melding = PoseArray()
                

                "Rodrigues gir ut rotasjonmatrise til marker frame"
                rotM, jacob = cv2.Rodrigues(rvec1) 
                rotM_trans = np.transpose(rotM)
                
                "Transformerer tvec til å tilsvare body i world frame"
                tvec_wf = np.dot(rotM_trans, tvec1)
                tvec_wf = tvec_wf*-1
                tvec_wf = np.concatenate((tvec_wf,[[1]]), 0)
                tvec_wf = np.dot(H_w2m,tvec_wf)

                "Justerer for differensen mellomm body frame og worlld frame--"
                rotM = np.dot(rot_body2world,rotM)

                "skriver pose som melding"
                quat = R2quat(rotM)

                trans_x = tvec_wf[0][0]
                trans_y = tvec_wf[1][0]
                trans_z = tvec_wf[2][0]

                msg_cam= Pose()
                msg_cam.position.x = trans_x
                msg_cam.position.y = trans_y
                msg_cam.position.z = trans_z
                msg_cam.orientation.w = quat[0][0]
                msg_cam.orientation.x = quat[1][0]
                msg_cam.orientation.y = quat[2][0]
                msg_cam.orientation.z = quat[3][0]

                melding.poses.append(msg_cam)
                
                
                "tegner kordinatsystemet på aruco_brettet"
                cv2.aruco.drawAxis(gray, camera_matrix, dist_coef , rvec1,tvec1, 1)

        cv2.imshow("image frame", gray)
        cv2.waitKey(3)

        "Publiserer melding"
        self.board_pub.publish(melding)
        
      
def main():
    rclpy.init()
    node = ArucoNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
