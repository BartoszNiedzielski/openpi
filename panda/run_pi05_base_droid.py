import time
import cv2
import numpy as np
import threading
import logging

# --- OpenPI Imports ---
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import panda_py
from panda_py import libfranka
from panda_py import controllers

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'

#place the green cube in the yellow area task
INSTRUCTION = "place the green cube in the yellow area"
CHECKPOINT_DIR = "/home/student/ft/checkpoints/pi05_base_droid/100/"

print("[*] Loading local fine-tuned Pi05 Base droid model...")
pi0_config = _config.get_config("pi05_panda")
policy = policy_config.create_trained_policy(pi0_config, CHECKPOINT_DIR)

# Physics & Control Parameters
CONTROL_HZ = 15
DT = 1.0 / CONTROL_HZ

# --- CAMERA CONFIGURATION ---
EXTERIOR_CAMERA_INDEX = 2
WRIST_CAMERA_INDEX = 0
MIN_STEPS_PER_CHUNK = 6

TEST_VISION_INFLUENCE = True

logging.basicConfig(level=logging.INFO)

# === SHARED VARIABLES ===
state_lock = threading.Lock()
latest_action_chunk = None 
is_running = True
latest_grip = 0.0

# Initialize globals
gripper = None
panda = None
desk = None
pos_ctrl = None
raw_target_joints = None


# ==========================================
# THREAD 1: THE BRAIN (Vision & AI)
# ==========================================
def vision_loop(cap_ext, cap_wrist, policy):
    global latest_action_chunk, is_running, gripper, panda
    
    print("[Brain] AI Online. Listening for visual updates...")
    
    total_vision_delta = 0.0
    total_steps = 0
    
    try:
        while is_running:
            ret_ext, frame_ext = cap_ext.read()
            ret_wrist, frame_wrist = cap_wrist.read()
            
            if not ret_ext or not ret_wrist: 
                continue
            
            image_ext_rgb = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
            image_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

            if TEST_VISION_INFLUENCE:
                blank_ext = np.zeros_like(image_ext_rgb)
                blank_wrist = np.zeros_like(image_wrist_rgb)

            with state_lock:
                width = gripper.read_once().width
                current_grip_state = 1.0 if width<0.04 else 0.0
                current_joints = np.array(panda.q, dtype=np.float32)

            # 1. NORMAL OBSERVATION (With Vision)
            example_with_vision = {
                "exterior_image_1_left": image_ext_rgb,
                "wrist_image_left": image_wrist_rgb,
                "gripper_position": np.array([current_grip_state], dtype=np.float32),
                "joint_position": current_joints,
                "prompt": INSTRUCTION
            }

            if TEST_VISION_INFLUENCE:
            # 2. BLIND OBSERVATION (Without Vision)
                example_without_vision = {
                    "exterior_image_1_left": blank_ext,
                    "wrist_image_left": blank_wrist,
                    "gripper_position": np.array([current_grip_state], dtype=np.float32),
                    "joint_position": current_joints,
                    "prompt": INSTRUCTION
                }

            # --- INFERENCE 1: Normal ---
            result_vision = policy.infer(example_with_vision)
            actions_vision = result_vision["actions"]
            if hasattr(actions_vision, 'cpu'): actions_vision = actions_vision.cpu().numpy()
            elif hasattr(actions_vision, 'device'): actions_vision = np.array(actions_vision)

            if TEST_VISION_INFLUENCE:
                # --- INFERENCE 2: Blind ---
                result_blind = policy.infer(example_without_vision)
                actions_blind = result_blind["actions"]
                if hasattr(actions_blind, 'cpu'): actions_blind = actions_blind.cpu().numpy()
                elif hasattr(actions_blind, 'device'): actions_blind = np.array(actions_blind)
                
                # === CALCULATE VISION IMPACT (L2 NORM) ===
                step_distance = np.linalg.norm(actions_vision - actions_blind)
                total_vision_delta += step_distance
                total_steps += 1
                current_delta_avg = total_vision_delta / total_steps

                print(f"[Analysis] Vision Impact this step: {step_distance:.4f} | Running Avg (\u0394_vision): {current_delta_avg:.4f}")

            h_ext, w_ext, _ = image_ext_rgb.shape
            h_wrist, w_wrist, _ = image_wrist_rgb.shape

            image_wrist_resized = cv2.resize(image_wrist_rgb, (int(w_wrist * h_ext / h_wrist), h_ext))
            combined_view = np.hstack((image_ext_rgb, image_wrist_resized))
            display_frame = cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR)

            try:
                cv2.imshow("Robot Cameras (Left: Ext | Right: Wrist)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            except cv2.error:
                pass 

            with state_lock:
                latest_action_chunk = actions_vision.copy()
                
    finally:
        cap_ext.release()
        cap_wrist.release()
        print("[Brain] Shutting down.")


# ==========================================
# THREAD 2: THE MUSCLE (Robot Control)
# ==========================================
def control_loop():
    global latest_action_chunk, is_running, panda, gripper, pos_ctrl, latest_grip, raw_target_joints

    current_chunk = None
    step_index = 0
    current_vel = np.zeros(7, dtype=np.float64) # Keep track of velocity for acc limiting
    
    if raw_target_joints is None:
        raw_target_joints = np.array(panda.q, dtype=np.float64)

    while is_running:
        step_start = time.time()

        with state_lock:
            if latest_action_chunk is not None:
                if current_chunk is None or step_index >= MIN_STEPS_PER_CHUNK:
                    current_chunk = latest_action_chunk
                    latest_action_chunk = None  
                    step_index = 0

        if current_chunk is not None and step_index < len(current_chunk):
            raw_target = current_chunk[step_index].astype(np.float64)
            grip = raw_target[7]
            raw_target_joints = raw_target[:7]
            try:
                predicted_pose = np.array(panda_py.fk(raw_target_joints))
                safe_pos = predicted_pose[:3, 3]
                if (safe_pos[0] > 0.8 or safe_pos[0] < 0 or 
                    safe_pos[1] > 0.3 or safe_pos[1] < -0.3 or 
                    safe_pos[2] > 0.65 or safe_pos[2] < 0.05):
                    print(f"outside of bounding box: {safe_pos}")
                    continue
            except Exception:
                continue

            step_index += 1
            panda.move_to_joint_position(raw_target_joints, speed_factor=0.1)
            print(f'grip: {grip}')
            
            # Gripper
            if grip >= 0.35 and latest_grip == 0.0:
                gripper.grasp(width=0.0, speed=0.1, force=40)
                latest_grip = 1.0
            elif grip < 0.35 and latest_grip == 1.0:
                gripper.move(width=0.08, speed=0.1)
                latest_grip = 0.0
        else:
            # current_vel = np.zeros(7)
            # pos_ctrl.set_control(last_commanded_q)
            # panda.move_to_joint_position(raw_target_joints, speed_factor=0.1)
            print("no new actions")
            pass

        time.sleep(max(0, DT - (time.time() - step_start)))
    
    print("[Muscle] Shutting down.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[*] Booting Asynchronous Robotics System...")
    
    # --- 1. INITIALIZE AI & CAMERAS FIRST ---
    print(f"[*] Starting Logitech Cameras (Exterior: {EXTERIOR_CAMERA_INDEX}, Wrist: {WRIST_CAMERA_INDEX})...")
    cap_ext = cv2.VideoCapture(EXTERIOR_CAMERA_INDEX)
    cap_wrist = cv2.VideoCapture(WRIST_CAMERA_INDEX)

    if not cap_ext.isOpened() or not cap_wrist.isOpened():
        print("[!] ERROR: Could not open cameras. Check indices.")
        exit(1)

    # --- 2. THE WARM-UP PASS ---
    print("\n[*] Warming up the AI (This will take ~20 seconds)...")
    ret_ext, frame_ext = cap_ext.read()
    ret_wrist, frame_wrist = cap_wrist.read()
    
    if ret_ext and ret_wrist:
        image_ext_rgb = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
        image_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
        
        dummy_joints = np.zeros(7, dtype=np.float32)
        dummy_grip = np.zeros(1, dtype=np.float32)

        warmup_example = {
            "exterior_image_1_left": image_ext_rgb,
            "wrist_image_left": image_wrist_rgb,
            "gripper_position": dummy_grip,
            "joint_position": dummy_joints,
            "prompt": INSTRUCTION
        }
        
        start_time = time.time()
        _ = policy.infer(warmup_example)
        print(f"[+] AI Warmup Complete! (Took {time.time() - start_time:.2f}s)\n")
    else:
        print("[!] ERROR: Could not read cameras for warmup.")
        exit(1)

    # --- 3. CONNECT TO ROBOT ---
    print("[*] Initializing Robot Connection...")
    try:
        desk = panda_py.Desk(ROBOT_IP, ROBOT_USER, ROBOT_PASS)
        desk.unlock()
        desk.activate_fci()

        panda = panda_py.Panda(ROBOT_IP)
        gripper = libfranka.Gripper(ROBOT_IP)
        
        print("[*] Homing robot...")
        panda.move_to_start(speed_factor=0.05)
        pose = panda.get_pose()
        pose[2,3] -= 0.1
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q, speed_factor=0.05)
        gripper.move(width=0.08, speed=0.1)
        time.sleep(1)

    except Exception as e:
        print(f"[*] Fatal Error connecting to Robot: {e}")
        exit(1)
    
    # --- 4. START THREADS ---
    brain_thread = threading.Thread(target=vision_loop, args=(cap_ext, cap_wrist, policy))
    muscle_thread = threading.Thread(target=control_loop)
    
    brain_thread.start()
    muscle_thread.start()
    
    try:
        while brain_thread.is_alive() and muscle_thread.is_alive():
            brain_thread.join(timeout=0.1)
            muscle_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("\n[*] Ctrl+C detected! Signaling threads to shut down safely...")
        is_running = False 
    
    brain_thread.join()
    muscle_thread.join()
    
    try:
        print("[*] Stopping controllers and locking brakes...")
        panda.stop_controller() 
        desk.lock()
        desk.release_control()
    except Exception:
        pass
    
    print("[*] System safely powered down.")