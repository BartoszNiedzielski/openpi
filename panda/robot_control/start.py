# Panda hostname/IP and Desk login information of your robot
hostname = '172.16.0.2'
username = 'Dentec'
password = 'Frankenstein'

# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)

import panda_py

desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()

from panda_py import libfranka

panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)
print(panda.get_position())

# print(panda.q)
# print(gripper.read_once().is_grasped)
# print(gripper.read_once().width)

# gripper.grasp(width=0.0, speed=0.1, force=40.0)
# print(gripper.read_once().is_grasped)
# print(gripper.read_once().width)


# gripper.move(width=0.08, speed=0.2)
# print(gripper.read_once().is_grasped)
print(gripper.read_once().width)

# input("cos enter")
# gripper.grasp(width=0.0, speed=0.1, force=40.0)
# print(gripper.read_once().is_grasped)
# print(gripper.read_once().width)

# input("cos enter")
# gripper.move(width=0.08, speed=0.2)
# print(gripper.read_once().is_grasped)
# print(gripper.read_once().width)

panda.move_to_start(speed_factor=0.05)
pose = panda.get_pose()
print(pose)
pose[2,3] -= 0.1
q = panda_py.ik(pose)
panda.move_to_joint_position(q, speed_factor=0.05)
print(panda.get_state())
print(panda.q)
print(panda.get_position())
gripper.move(width=0.08, speed=0.2)

# [ 7.81019539e-04 -8.07987119e-01  9.47371393e-03 -2.63888751e+00
#  -1.25035290e-02  1.80613900e+00  7.89604633e-01]
