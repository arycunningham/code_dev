#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import numpy as np


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'
    raw_sub_topic = '/depthai_node/image/raw'
    marker_num_topic = '/target_detection/marker_id'

    def __init__(self):
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=2)
        self.aruco_pub_raw = rospy.Publisher('/processed_aruco/image/raw', Image, queue_size=2)
        self.marker_num_pub = rospy.Publisher('/target_detection/marker_id', String, queue_size=2)
        self.br = CvBridge()

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        aruco, marker_ID = self.find_aruco(frame)
        self.publish_to_ros(aruco, marker_ID)

    def find_aruco(self, frame):
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        marker_ID = None
        if len(corners) > 0:
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
                cv2.putText(frame, str(marker_ID), (top_left[0], top_right[1] - 15), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 140, 0), 2)
        
        return frame, marker_ID

    def publish_to_ros(self, frame, marker_ID):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.aruco_pub.publish(msg_out)
        
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        self.aruco_pub_raw.publish(msg_img_raw)
        
        if marker_ID is not None:
            self.marker_num_pub.publish(String(data=str(marker_ID)))


def main():
    rospy.init_node('aruco_detector_node', anonymous=True)
    rospy.loginfo("Processing images for ArUco markers...")

    aruco_detect = ArucoDetector()

    rospy.spin()


if __name__ == '__main__':
    main()
