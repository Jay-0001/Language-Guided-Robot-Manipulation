# Language-Aware-Robot-Manipulation-Using-Vision-Language-Transformers

Author: Jayakaran Saravanan Indira
Course: CS581B -Robot Perception
Semester: Fall 2025

---

1. Project Description

This project implements a complete perception-to-action pipeline for "Language-Guided Robot Manipulation Using Vision Language Transformers."

Given a natural-language command referring to a colored sphere (e.g., “pick the red sphere”), the system:
Uses a Vision-Language Transformer grounding model (GroundingDINO) to detect the specified object in the PyBullet simulation.
Maps the predicted bounding box to a corresponding simulated object ID using camera projection.
Executes a pick-and-place action via inverse kinematics using a Franka Panda arm in PyBullet.

The repository contains:
Synthetic dataset generation for grounding analysis
Grounding experiments and evaluation scripts
Full integration pipeline combining parsing + grounding + projection + manipulation
Scripts to generate the qualitative and quantitative results used in the final report

---

2.Hardware / Software Requirements

Hardware
Ubuntu 20.04 / 22.04
RTX 3070 (Any NVIDIA GPU should be fine)
8 GB RAM minimum
SSD storage(>10GB)

Software and packages
PyTorch ≥ 1.13 (with CUDA)
torchvision
transformers
timm
numpy
Pillow (PIL)
opencv-python
pandas
seaborn (for analysis)
matplotlib
tqdm

---

3.Repository Structure

dataset/ # Dataset generation + metrics
Grounding/ # Grounding experiments and utilities
Images/ # Sample qualitative frames
Initial_Simulation_Attempt_Failed # Archived early attempts
Integration_Pipeline/ # End-to-end perception + manipulation pipeline
Pybullet Exploration/ # Initial PyBullet environment trials
Visual Linguistic reasoning_Attempts/ # VLP experiments
README.md

---

4. Running Description

This repository contains all components required to reproduce the perception-to-action pipeline.
Before running any scripts, ensure that GroundingDINO is correctly installed along with its pretrained weights. The grounding model is used to detect the referenced sphere in each PyBullet scene and forms the core perception module of the system.Information of the download and setup of GroundingDINO can be found at reference[1].

All the project level imports are located within the relevant subdirectories(dataset and Integration_Pipeline). The executable should work as long as the GroundingDINO, along with software requirements are installed and the paths are set within the executable file, in these lines:
"
if **name** == "**main**":
CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"
"

After cloning this repository,

(A) End-to-End Manipulation Pipeline
cd Integration_Pipeline
python end_to_end_pipeline.py

(B) Quantitative metrics
cd dataset
+To generate the sphere count wise metrics
python grounding_eval.py

+And then to summarize global metrics
python dataset_metrics.py

---

5. References

[1].GroundingDINO

IDEA Research. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.
https://github.com/IDEA-Research/GroundingDINO

[2].PyBullet

Coumans, E. PyBullet Physics Simulation for Robotics.
https://pybullet.org
