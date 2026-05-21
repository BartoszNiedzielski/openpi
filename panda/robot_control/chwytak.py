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
# gripper.homing()

# print(panda.get_state())
# print(panda.get_model())
# gripper.grasp(0, 0.2, 5, 0.04, 0.04)
# grip_width = gripper.read_once()
# print(gripper.read_once().is_grasped)
# grip_state = 1.0 if grip_width < 0.01 else 0.0
# print(f"Gripper width: {grip_width:.4f} m, Gripper state: {'Closed' if grip_state == 1.0 else 'Open'}")
gripper.move(0.08, 0.2)
print(gripper.read_once().is_grasped)