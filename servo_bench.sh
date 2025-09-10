rostopic pub /actuator_control/actuator_a std_msgs/Bool '{data: True}'
sleep 2
rostopic pub /actuator_control/actuator_a std_msgs/Bool '{data: False}'
sleep 2
rostopic pub /actuator_control/actuator_b std_msgs/Bool '{data: True}'
sleep 2
rostopic pub /actuator_control/actuator_a std_msgs/Bool '{data: False}'