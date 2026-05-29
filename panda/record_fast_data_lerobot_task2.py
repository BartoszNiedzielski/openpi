import logging
import time
import threading
import numpy as np
import cv2
from pathlib import Path
import panda_py
from panda_py import libfranka
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import gc

# --- Configuration ---
hostname = '172.16.0.2'
username = 'Dentec'
password = 'Frankenstein'
FPS = 15

# Camera Indices
EXTERIOR_CAMERA_INDEX = 2
WRIST_CAMERA_INDEX = 0

logging.basicConfig(level=logging.INFO)

# --- Shared Variables for Threading ---
is_recording = False
trajectory_buffer = []

episode_successful = True

def recording_thread(panda, gripper, max_width, cap_ext, cap_wrist):
    global is_recording, trajectory_buffer
    
    # Pre-set buffer size to 1 to ensure we aren't getting old frames
    cap_ext.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_wrist.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while is_recording:
        loop_start = time.perf_counter()
        
        # 1. HARDWARE SYNC: Grab frames as close as possible
        # We read them sequentially, but doing it in a tight block minimizes skew
        ret_ext, frame_ext = cap_ext.read()
        ret_wrist, frame_wrist = cap_wrist.read()
        
        # 2. STATE SYNC: Get robot state immediately after images
        # Use a single call if possible to minimize network roundtrips
        current_q = np.array(panda.q, dtype=np.float32)
        gripper_state = gripper.read_once()
        
        timestamp = time.perf_counter()
        
        if ret_ext and ret_wrist:
            rgb_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
            rgb_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
            
            continuous_grip = np.clip(1.0 - (gripper_state.width / max_width), 0.0, 1.0)
            
            # Store everything + the timestamp for later analysis
            trajectory_buffer.append({
                "ts": timestamp,
                "q": current_q,
                "grip": continuous_grip,
                "ext": rgb_ext,
                "wrist": rgb_wrist
            })

        # 3. PRECISE SLEEP: Calculate remaining time based on start of loop
        elapsed = time.perf_counter() - loop_start
        sleep_time = (1.0 / FPS) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == '__main__':
    # --- 1. Initialization ---
    print("[*] Initializing Cameras...")
    cap_ext = cv2.VideoCapture(EXTERIOR_CAMERA_INDEX)
    cap_wrist = cv2.VideoCapture(WRIST_CAMERA_INDEX)
    
    for cap in (cap_ext, cap_wrist):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap_ext.isOpened() or not cap_wrist.isOpened():
        raise RuntimeError("Failed to open one or both cameras. Check indices.")

    print("[*] Connecting to Robot...")
    desk = panda_py.Desk(hostname, username, password)
    desk.unlock()
    desk.activate_fci()

    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)
    
    # Get maximum physical width of this specific Franka Hand
    MAX_GRIPPER_WIDTH = gripper.read_once().max_width
    print(f"[*] Max Gripper Width identified: {MAX_GRIPPER_WIDTH}m")
    
    gripper.move(width=0.08, speed=0.1)
    gripper.homing()
    
    # --- DROID-LeRobot v2.1 Schema ---
    dataset_dir = Path("outputs/fast_pick_and_place_task2")
    
    if dataset_dir.exists():
        print("[*] Found existing dataset. Loading it to append a new episode...")
        dataset = LeRobotDataset("local/fast_pick_and_place_task2", root=dataset_dir)
    else:
        print("[*] Creating new dataset folder...")
        dataset = LeRobotDataset.create(
            repo_id="local/fast_pick_and_place_task2",
            root=dataset_dir,
            features={
                "exterior_image_1_left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
                "wrist_image_left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
                "joint_position": {"dtype": "float32", "shape": (7,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]},
                "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["grip"]},
                "actions": {"dtype": "float32", "shape": (8,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]},
            },
            fps=FPS,
            image_writer_threads=8,
        )

    print("[*] Homing robot and opening gripper...")
    panda.move_to_start(speed_factor=0.2)
    gripper.move(width=0.08, speed=0.1)
    pose = panda.get_pose()
    pose[2,3] -= 0.1 
    q = panda_py.ik(pose)
    panda.move_to_joint_position(q, speed_factor=0.2)
    time.sleep(1)

    # --- 2. Teaching Phase ---
    print('\n--- Teaching Mode: Poses ---')
    positions = []
    panda.teaching_mode(True)

    for i in range(2):
        input(f'Manually move the arm to Pose {i+1} and press Enter...')
        positions.append(panda.q)
        pose = panda.get_pose()
        print(pose)
        print(panda.get_position())

    panda.teaching_mode(False) 

    # --- Prep for Replay ---
    input('\nPress Enter to move to Start Position (Pose 1)... and replay the trajectory while recording data.')
    panda.move_to_start(speed_factor=0.2)
    gripper.move(width=0.08, speed=0.1)
    pose = panda.get_pose()
    pose[2,3] -= 0.1 
    q = panda_py.ik(pose)
    panda.move_to_joint_position(q, speed_factor=0.2)
    time.sleep(2)

    # --- 4. Replay & Record Phase ---
    print("RECORDING STARTED - Warming up camera exposure...")
    trajectory_buffer = []
    is_recording = True
    gc.disable()
    rec_thread = threading.Thread(target=recording_thread, args=(panda, gripper, MAX_GRIPPER_WIDTH, cap_ext, cap_wrist))
    rec_thread.start()
    
    time.sleep(2) # Give the camera 2 seconds to adjust exposure before movement begins
    trajectory_buffer = [] # Clear the buffer of the "static" frames
    
    # Pass gripper and max_width into the recording thread
    rec_thread = threading.Thread(target=recording_thread, args=(panda, gripper, MAX_GRIPPER_WIDTH, cap_ext, cap_wrist))
    rec_thread.start()

    try:
        print("move to grasp position")
        panda.move_to_joint_position(positions, speed_factor=0.1)

        print("Grasping...")
        gripper.grasp(width=0.0, speed=0.1, force=40.0)

        pose = panda.get_pose()
        pose[2,3] += 0.1
        panda.move_to_pose(pose,speed_factor=0.1)

        # Move above yellow area, then hold there without releasing.
        print("move to prep hold area")
        drop_pose = np.array([
            [ 0.99572986, -0.03349188,  0.0859141,   0.62354445],
            [-0.03706703, -0.99848776,  0.04036099,  0.16269892],
            [ 0.08443242, -0.04337322, -0.99548468,  0.14364749],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ], dtype=np.float64)

        print("moving to hold area above yellow zone")
        hold_pose = np.array([
            [ 0.99768186, -0.04291125,  0.05263387,  0.60838718],
            [-0.04519699, -0.99804111,  0.04303448,  0.15750647],
            [ 0.0506841,  -0.04531361, -0.99768618,  0.15000000],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ], dtype=np.float64)

        # # standing cube
        # print("move to prep drop area")
        # drop_pose = np.array([
        # [ 0.99425161, -0.10452163, 0.0227973, 0.59643517],
        # [-0.10405794, -0.99434695, 0.02066037, 0.16699551],
        # [ 0.02482788, -0.01816937, -0.9995266, 0.19222092],
        # [ 0.0,          0.0,          0.0,          1.0        ]
        # ], dtype=np.float64)

        # print("moving to place area")
        # place_pose = np.array([
        #     [ 0.99903821, -0.0216673,  0.03786763,  0.61696348],
        #     [-0.01932619, -0.99793291,  0.0611328,  0.16463854],
        #     [ 0.03911393, -0.06034217, -0.99741106,  0.1035841],
        #     [ 0.0,         0.0,         0.0,         1.0       ]
        # ], dtype=np.float64)

        panda.move_to_pose([drop_pose, hold_pose], speed_factor=0.1)
        print("holding cube above yellow area")
        time.sleep(2.0)

        pose = panda.get_pose()
        pose[2,3] += 0.1
        panda.move_to_pose(pose, speed_factor=0.1)

    except Exception as e:
        print(f"\n[!] ERROR DETECTED DURING REPLAY: {e}")
        print("[!] Skipping dataset save for this corrupted episode.")
        episode_successful = False
        
        try:
            panda.recover()
        except:
            pass

    finally:
        # D. Stop Recording Safely
        is_recording = False
        rec_thread.join()
        
        cap_ext.release()
        cap_wrist.release()
        print("RECORDING STOPPED.")
        gc.enable()

    if episode_successful:
        print(f"\nProcessing {len(trajectory_buffer)} frames for LeRobot...")
        instruction = "pick up cube one and hold it above the yellow area"

        # Validate timing quality
        ts = np.array([f['ts'] for f in trajectory_buffer])
        diffs = np.diff(ts)
        print(f"Mean FPS: {1.0/np.mean(diffs):.2f}")
        print(f"Max Jitter: {np.max(np.abs(diffs - (1.0/FPS))):.4f}s")

        for i in range(len(trajectory_buffer) - 1):
            # Access dictionary data
            curr = trajectory_buffer[i]
            nxt = trajectory_buffer[i + 1]
            
            # Combine into vectors
            state_vector = np.concatenate([curr['q'], [curr['grip']]]).astype(np.float32)
            
            # Action is typically the state of the robot at the NEXT timestep
            # This is standard for imitation learning policies
            action_vector = np.concatenate([nxt['q'], [nxt['grip']]]).astype(np.float32)

            frame = {
                "exterior_image_1_left": curr['ext'],
                "wrist_image_left": curr['wrist'],
                "joint_position": curr['q'],
                "gripper_position": np.array([curr['grip']], dtype=np.float32),
                "actions": action_vector,
                "task": instruction,
                "state": state_vector,
            }
            
            dataset.add_frame(frame)

        dataset.save_episode()
        print("Dataset episode saved successfully!")

        # After recording:
        ts = np.array([f['ts'] for f in trajectory_buffer])
        diffs = np.diff(ts)
        print(f"Mean FPS: {1.0/np.mean(diffs):.2f}")
        print(f"Max Jitter: {np.max(np.abs(diffs - (1.0/FPS))):.4f}s")