import os
import asyncio
import cvb
import time

# ROS IMPORTS
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image

# Additional libs
import time
import cv2
import numpy as np

rate_counter = None

async def async_acquire(left_cam_pub, right_cam_pub, left_img_pub, right_img_pub, cam_left_info, cam_right_info, left_img_msg, right_img_msg):
    global rate_counter
    with cvb.DeviceFactory.open(os.path.join(cvb.install_path(), "drivers", "GenICam.vin"), port=0) as device:
        with cvb.DeviceFactory.open(os.path.join(cvb.install_path(), "drivers", "GenICam.vin"), port=1) as device0:

            # Init devices and open stream
            stream = device.stream
            stream0 = device0.stream
            stream.start()
            stream0.start()

            # For counting the achieved framerate
            rate_counter = cvb.RateCounter()

            while not rospy.is_shutdown():

                # Wait for external trigger to send frame.
                result = await  stream.wait_async()
                time1 = time.time()
                result0 = await stream0.wait_async()
                time2 = time.time()

                # Fetch acquisition result (image and status)
                image, status = result.value
                image0, status0 = result0.value

                # Increment counter
                rate_counter.step()
                
                # If frames are ok, send stereo images to ROS Topics
                if status == cvb.WaitStatus.Ok and status0 == cvb.WaitStatus.Ok:
                    stamp = rospy.Time.from_sec(time2)

                    cam_left_info.header.stamp = stamp
                    cam_right_info.header.stamp = stamp

                    left_cam_pub.publish(cam_left_info)
                    right_cam_pub.publish(cam_right_info)

                    left_img_msg.header.stamp = stamp
                    right_img_msg.header.stamp = stamp

                    left_img_msg.data = np.array(cvb.as_array(image, copy=False)).tobytes()
                    right_img_msg.data = np.array(cvb.as_array(image0, copy=False)).tobytes()

                    left_img_pub.publish(left_img_msg)
                    right_img_pub.publish(right_img_msg)

            stream.abort()
            stream0.abort()
    

if __name__ == '__main__':


    left_img_pub = rospy.Publisher("images/left_camera", Image, queue_size=10)
    right_img_pub = rospy.Publisher("images/right_camera", Image, queue_size=10)

    left_cam_pub = rospy.Publisher("images/left_camera_info", CameraInfo)
    right_cam_pub = rospy.Publisher("images/right_camera_info", CameraInfo)

    rospy.init_node("stereo_publisher")

    cam_left_info = CameraInfo()
    cam_left_info.height = 2064
    cam_left_info.width = 1544
    cam_left_info.distortion_model = "rational_polynomial"

    cam_left_info.K = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cam_left_info.D = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cam_left_info.R = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cam_left_info.P = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    cam_right_info = CameraInfo()
    cam_right_info.height = 2064
    cam_right_info.width = 1544
    cam_right_info.distortion_model = "rational_polynomial"

    cam_right_info.K = [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0]
    cam_right_info.D = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cam_right_info.R = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cam_right_info.P = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    
    left_img_msg = Image()
    left_img_msg.height = 2064

    left_img_msg.width = 1544
    left_img_msg.encoding = "rgb8"
    left_img_msg.header.frame_id = "image_rect"

    right_img_msg = Image()
    right_img_msg.height = 2064
    right_img_msg.width = 1544
    right_img_msg.encoding = "rgb8"
    right_img_msg.header.frame_id = "image_rect2"
    
        

    watch = cvb.StopWatch()

    # Asynchronous run the acquisition func
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        async_acquire(left_cam_pub, right_cam_pub, left_img_pub, right_img_pub, cam_left_info, cam_right_info, left_img_msg, right_img_msg))) 
    loop.close()

    duration = watch.time_span

    print("Acquired on port 0 with " + str(rate_counter.rate) + " fps")
    print("Overall measurement time: " +str(duration / 1000) + " seconds")


