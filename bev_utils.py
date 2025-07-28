import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from collections import defaultdict
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion


def quaternion_to_yaw(q):
    """
    Convert quaternion to yaw angle in radians.
    """
    q = Quaternion(q)
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y ** 2 + q.z ** 2))


def generate_bev_animation(nusc, scene_index, output_path="bev_animation.gif", fps=5):
    """
    Generate BEV animation for a specific scene and save as GIF.
    
    Args:
        nusc: NuScenes instance.
        scene_index: Index of the scene in nuScenes-mini.
        output_path: Path where the animation GIF will be saved.
        fps: Frames per second for the saved animation.

    Returns:
        Path to the saved GIF file.
    """
    # === Dataset setup ===
    scene = nusc.scene[scene_index]
    first_sample_token = scene['first_sample_token']
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)

    # === Marker style by category root ===
    category_marker_map = {
        'vehicle': 'o',
        'human': 's',
        'movable_object': '^',
        'animal': 'D',
        'static_object': 'x',
        'flat': 'P'
    }

    # === Containers ===
    ego_trajectory, ego_orientations, timestamps = [], [], []
    object_trajectories, object_orientations = defaultdict(list), defaultdict(list)
    object_labels = {}
    object_name_counter = defaultdict(int)
    object_category_root = {}

    # === Loop through frames ===
    sample_token = first_sample_token
    while sample_token:
        sample = nusc.get('sample', sample_token)
        timestamps.append(sample['timestamp'])

        ego_pose = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        ego_trajectory.append(tuple(ego_pose['translation'][:2]))
        ego_orientations.append(quaternion_to_yaw(ego_pose['rotation']))

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            inst = ann['instance_token']
            category = ann['category_name']
            root = category.split('.')[0]
            object_category_root[inst] = root

            if inst not in object_labels:
                object_name_counter[category] += 1
                object_labels[inst] = f"{category}_{object_name_counter[category]}"

            object_trajectories[inst].append(tuple(ann['translation'][:2]))
            object_orientations[inst].append(quaternion_to_yaw(ann['rotation']))
        sample_token = sample['next']

    # === Normalize & Filter ===
    timestamps = np.array(timestamps)
    ego_trajectory = np.array(ego_trajectory)
    ego_orientations = np.array(ego_orientations)

    unique_location_counts = {
        k: len(set(object_trajectories[k])) for k in object_trajectories
    }
    sorted_lens = sorted(set(unique_location_counts.values()), reverse=True)
    threshold = sorted_lens[9] if len(sorted_lens) > 10 else 0
    selected_instances = {k for k, v in unique_location_counts.items() if v >= threshold}

    # === Color map ===
    color_map = cm.get_cmap("tab10", len(selected_instances))
    instance_colors = {inst: color_map(i) for i, inst in enumerate(selected_instances)}

    # === Set up map plot ===
    x_min, y_min = ego_trajectory.min(axis=0) - 100
    x_max, y_max = ego_trajectory.max(axis=0) + 100
    fig, ax = nusc_map.render_map_patch((x_min, y_min, x_max, y_max), ['drivable_area', 'walkway'], alpha=0.05)
    fig.set_size_inches(10, 10)
    ax.set_aspect('equal')

    # Set tight bounding box
    ax.set_position([0.1, 0.1, 0.8, 0.8])

    # === Ego ===
    ego_dot, = ax.plot([], [], 'o', color='black', markersize=6)
    ego_arrow = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='black')

    # === Object dots & legend entries ===
    obj_dots = {}
    grouped_legend = defaultdict(list)

    for i, inst in enumerate(selected_instances):
        category = object_category_root[inst]
        marker = category_marker_map.get(category, 'o')
        color = instance_colors[inst]
        label = object_labels[inst]

        dot, = ax.plot([], [], marker=marker, linestyle='None', color=color, markersize=6)
        obj_dots[inst] = dot

        legend_entry = plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', markersize=8, label=label)
        grouped_legend[category].append(legend_entry)

    # === Animation functions ===
    def init():
        ego_dot.set_data([], [])
        ego_arrow.set_offsets([0, 0])
        ego_arrow.set_UVC(0, 0)
        for dot in obj_dots.values():
            dot.set_data([], [])
        return [ego_dot, ego_arrow] + list(obj_dots.values())

    def update(frame):
        x, y = ego_trajectory[frame]
        yaw = ego_orientations[frame]
        ego_dot.set_data(x, y)
        ego_arrow.set_offsets([x, y])
        ego_arrow.set_UVC(3 * np.cos(yaw), 3 * np.sin(yaw))
        for inst in selected_instances:
            traj = object_trajectories[inst]
            if frame < len(traj):
                px, py = traj[frame]
                obj_dots[inst].set_data(px, py)
        return [ego_dot, ego_arrow] + list(obj_dots.values())

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=len(timestamps), interval=300, blit=False)

    # === Legend ===
    category_order = ['vehicle', 'movable_object', 'human', 'animal', 'static_object', 'flat']
    legend_entries = [plt.Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8, label='Ego Vehicle')]

    for cat in category_order:
        if cat in grouped_legend:
            legend_entries.extend(grouped_legend[cat])

    ax.legend(handles=legend_entries, loc='upper right', fontsize=8, frameon=True)
    plt.title(f"Ego & Object Trajectories on nuScenes Map ({location})")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid()

    # === Save animation ===
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)

    return output_path
