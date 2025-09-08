#!/usr/bin/env python3
#needs to run 
	#sudo pigpiod
	#export PIGPIO_ADDR=soft
	#export PIGPIO_PORT=%%%%
		#rostopic pub /actuator_control/actuator_a std_msgs/Bool '{data: True}'
import rospy
import RPi.GPIO as GPIO
from std_msgs.msg import Bool
from gpiozero import AngularServo
from gpiozero import Device
import pigpio
from gpiozero.pins.pigpio import PiGPIOFactory
Device.pin_factory = PiGPIOFactory('127.0.0.1')

from time import sleep


factory = PiGPIOFactory()
sub_a = None
servo_a = AngularServo(12, min_angle=-90, max_angle=90,pin_factory=factory)
servo_b = AngularServo(13, min_angle=-90, max_angle=90,pin_factory=factory)

def callback_a(msg_in):
	# A bool message contains one field called "data" which can be true or false
	# http://docs.ros.org/melodic/api/std_msgs/html/msg/Bool.html
	if msg_in.data:
		rospy.loginfo("Rotating Servo A")
		servo_a.angle=90
		sleep(1)
		servo_a.angle=0
		sleep(1)
	else:
		rospy.loginfo("Center Point")
		servo_a.angle=0
		sleep(1)

sub_b = None

def callback_b(msg_in):
	if msg_in.data:
		rospy.loginfo("Rotating Servo B")
		servo_b.angle=-90
		sleep(1)
		servo_b.angle=0
		sleep(1)
	else:
		rospy.loginfo("Center Point")
		servo_b.angle=0
		sleep(1)


def shutdown():
	# Clean up our ROS subscriber if they were set, avoids error messages in logs
	if sub_a is not None:
		sub_a.unregister()
	if sub_b is not None:
		sub_b.unregister()

	# XXX: Could perform some failsafe actions here!


if __name__ == '__main__':
	# Setup the ROS backend for this node
	rospy.init_node('actuator_controller', anonymous=True)

	# Setup the publisher for a single actuator (use additional subscribers for extra actuators)
	sub_a = rospy.Subscriber('/actuator_control/actuator_a', Bool, callback_a)
	sub_b = rospy.Subscriber('/actuator_control/actuator_b', Bool, callback_b)

	# Make sure we clean up all our code before exiting
	rospy.on_shutdown(shutdown)

	# Loop forever
	rospy.spin()
