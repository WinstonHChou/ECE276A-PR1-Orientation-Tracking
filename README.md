# ECE276A-PR1-Orientation-Tracking
WI 25 ECE 276A Project 1: Orientation Tracking

## Course Overview
This is Project 1 for [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, taught by Professor [Nikolay Atanisov](https://natanaso.github.io/).

## Project Description
The project involves processing IMU, VICON, and camera data to:
- Improve quaternions readings using gradient descent
- Determine sensor orientation using Motion Model of a 3D rigid body and Observation Model of a 6-axis IMU (accelerometer, gyroscope)
- Create panoramic images from multiple camera frames

## Prerequisites
- Python 3.12
- Required Packages:
    ```
    pip3 install -r requirements.txt
    ```

## Dataset Setup
1. Download the dataset from [OneDrive](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/w3chou_ucsd_edu/EpRBzH7ljQZFvZ7O9x2R_gQBV4dtu8yBDr3s3wMVzSCLnw?e=Ta5u5S)
2. Organize the folder structure as follows:
    ```text
    .
    └── data
      ├── testset
      │ ├── cam
      │ └── imu
      ├── trainset
      │ ├── cam
      │ ├── imu
      │ └── vicon
      └── README.md
    ```

## Running the Project
### Standard Run:
```python
python3 src/main.py dataset=1
```
### Optional Arguments:
- `dataset=1`: Select dataset number (1-11)
- `iterations=1`: Select desired iterations (Default: 300)
- `stepSize=1`: Select desired step size (Default: 0.025)
- `rough=true`: Choose to roughly plot the panorama
Example:
```python
python3 src/main.py dataset=1 iterations=150 stepSize=0.05 rough=true
```

## Features
- Datasets are labelled 1-11 (10, 11 are from `data/testset/`)
- Generates **spherical projection panorama images** for datasets 1, 2, 8, 9, 10, 11

## Output Graphs
### Euler Angles Comparison
- Predicted quaternions from Motion Model with IMU's angular velocities as input
- Gradient descent processed quaternions
- VICON data as Ground Truth

### Linear Acceleration Comparison
- Linear acceleration from predicted quaternions using Observation Model
- Linear acceleration from Gradient descent processed quaternions using Observation Model
- Raw IMU data

## Sample Results (Dataset 2)
### Euler Angles Comparison
Euler Angles Comparison:
![image](assets/2_AngVel.jpg)

### Linear Acceleration Comparison
Linear Acceleration Comparison:
![image](assets/2_Accel.jpg)

### Panorama
![image](assets/panorama_2.jpg)

## Technical Details
- Quaternions and rotation matrices converted using transforms3d
- Gradient descent used for IMU data refinement

## Acknowledgments
This project is part of the ECE 276A course at UCSD, which is inspired by the teaching and research of various professors in the field of robotics and estimation