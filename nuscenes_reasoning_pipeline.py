from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import json
import math
from openai import OpenAI

# --- CONFIG ---
NUSCENES_DIR = "/home/aleenatron/Frames_AV_research"  # Change this path
OUTPUT_JSON = "nuscenes_reasoning_output.json"
client = OpenAI(api_key="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")

nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DIR, verbose=True)

# ----------------- Helper Functions -----------------

def compute_speed(ego_pose_1, ego_pose_2, dt):
    p1 = ego_pose_1['translation']
    p2 = ego_pose_2['translation']
    dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
    return dist / dt if dt > 0 else 0.0

def extract_scene_context(scene_idx=0):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    q = Quaternion(ego_pose['rotation'])  # rotation is [w, x, y, z]
    yaw = q.yaw_pitch_roll[0] * 180 / math.pi  # yaw in degrees

    speed = 0.0
    if sample['next'] != '':
        next_sample = nusc.get('sample', sample['next'])
        next_lidar_sd = nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
        next_ego_pose = nusc.get('ego_pose', next_lidar_sd['ego_pose_token'])
        dt = (next_lidar_sd['timestamp'] - lidar_sd['timestamp']) / 1e6
        speed = compute_speed(ego_pose, next_ego_pose, dt)

    annotations = [nusc.get('sample_annotation', ann) for ann in sample['anns']]
    objects = list(set(ann['category_name'] for ann in annotations))

    return {
        "scene_name": scene['name'],
        "ego_speed_m_s": round(speed, 2),
        "ego_turn_angle_deg": round(yaw, 2),
        "objects": objects
    }

# ----------------- ChatGPT Reasoning -----------------

def chatgpt_reasoning(context):
    objects_str = ", ".join(context['objects'])
    prompt = f"""
You are an autonomous driving assistant. Your task is to analyze the driving scene and describe what the ego vehicle is doing and why.

### Few Examples:
Input: speed = 0.5 m/s, angle = 0°, objects = pedestrian crossing.
Answer: The car is stopping because a pedestrian is crossing.

Input: speed = 4.0 m/s, angle = 0°, objects = empty road.
Answer: The car is accelerating because the road ahead is clear.

Input: speed = 2.2 m/s, angle = -30°, objects = car in left lane.
Answer: The car is turning left to follow the road.

### Now Analyze:
speed = {context['ego_speed_m_s']} m/s
turning angle = {context['ego_turn_angle_deg']}°
objects = {objects_str}

Answer in JSON format:
{{"action": "...", "reason": "..."}}.
"""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
    )
    return response.choices[0].message.content
# ----------------- Main -----------------

if __name__ == "__main__":
    context = extract_scene_context(0)
    reasoning = chatgpt_reasoning(context)

    output = {
        context['scene_name']: {
            "context": context,
            "reasoning": json.loads(reasoning)
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=4)

    print("Saved reasoning to", OUTPUT_JSON)
