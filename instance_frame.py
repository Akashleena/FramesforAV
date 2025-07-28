import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


def compute_scene_ego_positions(nusc, scene_index):
    scene_ego_positions = {}
    scene = nusc.scene[scene_index]
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    ego_vehicle_raw_start_time = first_sample['timestamp'] / 1e6

    sample_token = first_sample_token
    while sample_token:
        sample = nusc.get('sample', sample_token)
        timestamp = sample['timestamp'] / 1e6
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        scene_ego_positions[timestamp] = tuple(ego_pose['translation'][:2])
        sample_token = sample['next']

    return scene_ego_positions, ego_vehicle_raw_start_time

class InstanceFrame:
    """ Class to store and process instance frame information from nuScenes dataset. """
    
    def __init__(self, nusc, instance_token, scene_index, scene_ego_positions, ego_vehicle_raw_start_time):
        """
        Initializes an InstanceFrame object and extracts relevant data.

        Args:
            nusc: NuScenes database instance.
            instance_token: The token of the instance to extract.
            scene_index: The index of the scene in nuScenes dataset.
            scene_ego_positions: Precomputed ego vehicle positions for this scene.
            ego_vehicle_raw_start_time: The raw start timestamp of the ego vehicle (in seconds).
        """
        self.nusc = nusc
        self.instance_token = instance_token
        self.scene_index = scene_index
        self.scene_ego_positions = scene_ego_positions
        self.ego_vehicle_raw_start_time = ego_vehicle_raw_start_time

        # Core attributes
        self.category_name = None
        self.instance_name = None
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.raw_start_time = None
        self.raw_end_time = None
        self.locations = []
        self.orientations = []
        self.sample_tokens = []
        self.scene_token = None
        self.print_debug = False

        # Extract information
        self._extract_instance_data()
        self._normalize_times()

    def _extract_instance_data(self):
        """ Extract structured Frame representation for the instance in nuScenes. """

        instance = self.nusc.get('instance', self.instance_token)
        first_ann_token = instance['first_annotation_token']
        
        ann = self.nusc.get('sample_annotation', first_ann_token)
        self.category_name = ann['category_name']

        while ann:
            sample = self.nusc.get('sample', ann['sample_token'])
            timestamp = sample['timestamp'] / 1e6  # Convert to seconds

            if self.start_time is None:
                self.start_time = timestamp
                self.raw_start_time = timestamp
                self.scene_token = sample["scene_token"]

            self.end_time = timestamp
            self.raw_end_time = timestamp

            # Transform annotation from global frame to LiDAR frame
            lidar_translation, lidar_rotation = self._get_lidar_pose(sample['data']['LIDAR_TOP'])
            obj_translation = np.array(ann["translation"])
            obj_rotation = Quaternion(ann["rotation"])

            transformed_translation, transformed_yaw = self._transform_global_to_lidar(
                obj_translation, obj_rotation, lidar_translation, lidar_rotation
            )

            self.locations.append(transformed_translation[:2])  # X, Y in LiDAR frame
            self.orientations.append(transformed_yaw)  # Yaw in LiDAR frame

            self.sample_tokens.append(ann["sample_token"])

            if ann["next"]:
                ann = self.nusc.get("sample_annotation", ann["next"])
            else:
                break

        self.duration = self.end_time - self.start_time

    def _normalize_times(self):
        """ Normalize start and end times relative to the ego vehicle's first timestamp in the scene. """
        self.start_time -= self.ego_vehicle_raw_start_time
        self.end_time -= self.ego_vehicle_raw_start_time

    def _get_lidar_pose(self, lidar_token):
        """ Retrieve the LiDAR position and rotation from ego pose. """
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        return np.array(ego_pose['translation']), Quaternion(ego_pose['rotation'])

    def _transform_global_to_lidar(self, obj_translation, obj_rotation, lidar_translation, lidar_rotation):
        """
        Transform object position and orientation from global frame to the LiDAR frame.
        """
        # Step 1: Translate to LiDAR's origin
        relative_translation = obj_translation - lidar_translation

    
        # Step 2: Rotate using inverse of LiDAR rotation
        transformed_translation = lidar_rotation.inverse.rotate(relative_translation)
        
        # **Fix the Axis Swapping Issue**
        transformed_translation = np.array([-transformed_translation[1], transformed_translation[0], transformed_translation[2]])
        

        # Step 3: Convert rotation to LiDAR's frame

        transformed_rotation = lidar_rotation.inverse * obj_rotation
        transformed_yaw = np.arctan2(
            2.0 * (transformed_rotation.w * transformed_rotation.z + transformed_rotation.x * transformed_rotation.y),
            1.0 - 2.0 * (transformed_rotation.y**2 + transformed_rotation.z**2)
        )
    

        # Choose whether debug with print
        if self.print_debug:
            print(f" LiDAR Position: {obj_rotation}")           
            print(f" Global Position: {obj_translation}")
            print(f" LiDAR Position: {lidar_translation}")
            print(f" Relative Position (before rotation): {relative_translation}")
            print(f"✅ Fixed Transformed Position (LiDAR Frame): {transformed_translation}")
            print(f" Original Rotation (Quaternion): {obj_rotation}")
            print(f"✅ Transformed Yaw (LiDAR Frame): {np.degrees(transformed_yaw)} degrees")
    
        return transformed_translation, transformed_yaw


    def _classify_orientation(self, yaw_rad):
        """ Classify the relative orientation in LiDAR frame. """
        yaw_deg = np.degrees(yaw_rad) % 360
        if yaw_deg > 180:
            yaw_deg -= 360  # Normalize to [-180, 180]

        if abs(yaw_deg) < 45:
            return "facing same direction"
        elif abs(yaw_deg - 180) < 45 or abs(yaw_deg + 180) < 45:
            return "facing opposite direction"
        elif 45 <= yaw_deg <= 135:
            return "facing leftward (perpendicular)"
        elif -135 <= yaw_deg <= -45:
            return "facing rightward (perpendicular)"
        else:
            return "heading unclear"



    def describe_movement(self):
        """
        A detailed scenario-based & dominant-direction movement description
        in LiDAR frame (x=right, y=up), considering the **entire trajectory**.
    
        Key features:
        - Describes movement trends over time.
        - Separates Left/Right and Forward/Backward positioning (no center state).
        - Tracks relative positioning with respect to the ego vehicle.
        - Detects crossing events (left↔right, behind↔ahead).
        - Computes movement dominance dynamically.
        - Reports mean and minimum distance to the ego vehicle.
    
        The 'one region' logic: explicitly says "always left/right" or "always behind/ahead"
        if applicable.



        TODO: 
              1. We take the relative distances as the trajectory distance (which might not be the same as the actual path length, according to the ego vehicle).
        """
    
        if len(self.locations) < 2:
            return "Not enough data to describe movement."
    
        distances_to_ego = [np.linalg.norm(loc) for loc in self.locations]
        mean_distance = np.mean(distances_to_ego)
        min_distance = np.min(distances_to_ego)

        movement_segments = []
        heading_segments = []
        segment_changes_lr = []
        segment_changes_fb = []
        relative_lr = []
        relative_fb = []
        total_distance = 0

        prev_x, prev_y = self.locations[0]

        for i in range(1, len(self.locations)):
            cur_x, cur_y = self.locations[i]
            dx, dy = cur_x - prev_x, cur_y - prev_y
            distance = np.hypot(dx, dy)
            total_distance += distance

            # --- Movement direction ---
            if distance < 0.5:
                move_desc = "had minor movement"
            else:
                abs_dx, abs_dy = abs(dx), abs(dy)
                ratio = abs_dx / abs_dy if abs_dy > 1e-6 else float('inf')
                if ratio > 1.5:
                    move_desc = "moved dominantly right" if dx > 0 else "moved dominantly left"
                elif ratio < 0.66:
                    move_desc = "moved dominantly forward" if dy > 0 else "moved dominantly backward"
                else:
                    move_desc = "moved diagonally"
            movement_segments.append(move_desc)

            # --- Orientation status ---
            heading_segments.append(self._classify_orientation(self.orientations[i]))

            # --- Relative lateral / longitudinal position ---
            cur_lr = "left" if cur_x < 0 else "right"
            cur_fb = "behind" if cur_y < 0 else "ahead"

            if i > 1:
                prev_lr, prev_fb = relative_lr[-1], relative_fb[-1]
                if prev_lr != cur_lr:
                    segment_changes_lr.append(f"crossed from {prev_lr} to {cur_lr}")
                if prev_fb != cur_fb:
                    segment_changes_fb.append(f"crossed from {prev_fb} to {cur_fb}")

            relative_lr.append(cur_lr)
            relative_fb.append(cur_fb)

            prev_x, prev_y = cur_x, cur_y

        movement_trend = ", then ".join(dict.fromkeys(movement_segments))
        heading_trend = ", then ".join(dict.fromkeys(heading_segments))

        # Position summaries
        lr_summary = (
            f"always {relative_lr[0]}" if len(set(relative_lr)) == 1 else
            ", ".join(dict.fromkeys(segment_changes_lr)) if segment_changes_lr else
            "exhibited partial left/right shifts"
        )
        fb_summary = (
            f"always {relative_fb[0]}" if len(set(relative_fb)) == 1 else
            ", ".join(dict.fromkeys(segment_changes_fb)) if segment_changes_fb else
            "exhibited partial forward/back shifts"
        )

        return (
            f"A {self.category_name} {movement_trend}. "
            f"Relative to the ego vehicle, it was {lr_summary} laterally "
            f"and {fb_summary} longitudinally. "
            f"It covered a total distance of approximately {total_distance:.2f}m over {self.duration:.2f}s. "
            f"Mean distance: {mean_distance:.2f}m, closest approach: {min_distance:.2f}m. "
            f"Its orientation over time was: {heading_trend}."
        )
