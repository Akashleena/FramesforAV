## Behavior Description → Scene Graph Construction

1. **Behavior Description**  
Li's `describe_movement()` method produces a summary for each agent based on its trajectory in the LiDAR frame:  
`A vehicle.car moved dominantly forward, then diagonally...`  
`Relative to the ego, it was always left and crossed ahead.`

2. **Triplet Extraction**  
From the description, we extract (subject, relation, object) triplets:  
`(car_14, always_ahead_of, ego_vehicle)`  
`(car_14, orientation_change, facing_right → opposite)`

3. **Scene Graph Construction**  
- **Nodes** = dynamic/static agents (car, pedestrian, barrier, etc.)  
- **Edges**:  
  - Spatial: `always_behind_of`, `crossed_left_to_right`  
  - Behavioral: `moved_diagonally`, `had_minor_movement`  
  - Temporal: `orientation_change`  
- The ego vehicle acts as the central node anchoring the scene.

Self-edges denote orientation evolution.  
Ego-relative edges are shown in red.
