## Behavior Description → Scene Graph Construction

# 🚘 Rule-Based Scene Graphs from Frame Descriptions

This project generates scene graphs from symbolic behavior descriptions using Li’s `describe_movement()` function on NuScenes scenes. The goal is to represent driving scenes in an interpretable, structured graph format that highlights agent behaviors relative to the ego vehicle.

---

## 🔧 How the Scene Graph is Generated
Frame theroy output to triples

> A human.pedestrian.adult crossed from left to right and was always ahead of the ego vehicle. It faced rightward.

From these, we extract relation triples in the format:

```python
Triples
(subject, relation, object)
# Example:
(human.pedestrian.adult_7, crossed_left_to_right, ego_vehicle)
(human.pedestrian.adult_7, always_ahead_of, ego_vehicle)
(human.pedestrian.adult_7, facing, rightward)
```

## 🎯 Filtering and Prioritization

Many agent–ego pairs have multiple valid relations, e.g.:

```python
('human.pedestrian.adult_7', 'ego_vehicle') → ['crossed_left_to_right', 'always_ahead_of']
('vehicle.bus.rigid_1', 'ego_vehicle') → ['always_left_of', 'always_ahead_of']
```

This ensures:Graph is minimal. Redundant or overlapping info is dropped at visualization level (but still logged)

**Priority order:**

```text
# Priority order: lower index = higher priority
priority = [
    'crossed_left_to_right', 'crossed_right_to_left',
    'always_left_of', 'always_right_of',
    'always_ahead_of', 'always_behind',
    'facing_rightward', 'facing_leftward'
    'facing_opposite_direction', 'facing_same_direction' 
]
```

## 📊 Current Output

- **Nodes** = scene objects (vehicles, pedestrians, movable objects)  
- **Edges** = symbolic relations (spatial or behavioral)  
- **Edge labels** = rule-based relations extracted from Frame descriptions  
- `ego_vehicle` is pinned at the center of the layout

## 🧠 Next Steps

1. **Chronology-Aware Relations**  
   We plan to encode temporal sequences in edge labels:  
   ```text
   crossed_left_to_right → always_ahead_of
   ```

2. **Learning-Based Comparison**  
   We will compare these rule-based graphs with learned scene graphs which uses sensor data using:  
   - CLIP/ViT-based relation prediction  
   - Graph alignment metrics  
   - Qualitative validation on downstream reasoning

## 📄 Sample Extracted Triples

```text
(human.pedestrian.adult_7, crossed_left_to_right, ego_vehicle)
(vehicle.bus.rigid_1, always_ahead_of, ego_vehicle)
(vehicle.car_18, always_behind, ego_vehicle)
(movable_object.barrier_13, always_left_of, ego_vehicle)
(human.pedestrian.adult_6, crossed_left_to_right, ego_vehicle)
...
```

## 📁 Files

- `scene_graphs/ : Contains plots 
- `scene_02_extracted_triples.txt`: raw extracted triples from NuScenes scene 2
