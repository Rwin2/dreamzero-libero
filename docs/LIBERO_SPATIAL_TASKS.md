# LIBERO-Spatial Tasks (10 tasks)

All tasks involve picking up a black bowl from different spatial locations and placing it on the plate. The challenge is spatial reasoning — understanding relative positions (between, next to, on top of, in drawer, etc.).

| Task ID | Instruction | Spatial Relation |
|---|---|---|
| 0 | pick up the black bowl **between the plate and the ramekin** and place it on the plate | between |
| 1 | pick up the black bowl **next to the ramekin** and place it on the plate | next to (ramekin) |
| 2 | pick up the black bowl **from table center** and place it on the plate | table center |
| 3 | pick up the black bowl **on the cookie box** and place it on the plate | on top of |
| 4 | pick up the black bowl **in the top drawer of the wooden cabinet** and place it on the plate | inside drawer |
| 5 | pick up the black bowl **on the ramekin** and place it on the plate | on top of (ramekin) |
| 6 | pick up the black bowl **next to the cookie box** and place it on the plate | next to (cookie box) |
| 7 | pick up the black bowl **on the stove** and place it on the plate | on stove |
| 8 | pick up the black bowl **next to the plate** and place it on the plate | next to (plate) |
| 9 | pick up the black bowl **on the wooden cabinet** and place it on the plate | on cabinet |

## Zero-Shot Results (DreamZero-DROID, no fine-tuning)

### OSC_POSE controller (delta end-effector actions)
Action space mismatch: model outputs delta joint angles, controller expects delta EE.

| Task | Success Rate | Trials |
|---|---|---|
| 0 | 0% | 3 |
| 1 | 0% | 3 |
| 2 | 0% | 3 |
| 3 | 0% | 3 |
| 4 | 0% | 3 |
| **Avg** | **0%** | 15 (5/10 tasks completed) |

### JOINT_POSITION controller — unscaled (delta joint angles, no rescaling)
Action space match but 20x scaling mismatch: model outputs radians (±0.7), controller scales input by 0.05.

| Task | Success Rate | Trials |
|---|---|---|
| 0-9 | 0% | 1 each |
| **Avg** | **0%** | 10 |

Robot barely moves — actions 20x too small.

### JOINT_POSITION controller — scaled (÷0.05 rescaling)
Correct action type AND correct scaling.

*Results updating as eval runs...*

## Action Space Analysis (from papers)

### DROID dataset (arxiv 2403.12945)
- Stores ALL action types redundantly: cartesian_position, cartesian_velocity, joint_position, joint_velocity, gripper
- Canonical action: 7D absolute EE pose [x,y,z,euler_xyz,gripper]
- Collected via VR teleoperation at 15Hz using Cartesian impedance controller

### DreamZero (arxiv 2602.15922)
- Uses **relative joint positions** (delta joint): `Δθ[t] = joint_pos[t+1] - joint_pos[t]`
- Trained separately per embodiment (not multi-embodiment)
- Actions normalized (q99), denormalized at inference
- DROID model: action_horizon=24 at 15Hz = 1.6s chunks

### LIBERO (arxiv 2306.03310)
- Uses OSC_POSE controller: delta end-effector [Δx,Δy,Δz,Δrx,Δry,Δrz,gripper]
- Input range [-1,1], output scaled by [0.05m, 0.05m, 0.05m, 0.5rad, 0.5rad, 0.5rad]
- 20Hz control frequency
