# Project Details
---
In this project, an Advantage Actor Critic (A2C) network is trained to control automatic arms that will try to touch and follow the moving balls.

![img](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

## The Environment
The environment for this project involves controlling a double-jointed arm to reach target locations.
### State Space
state is continuous, the state vector has **33** dimensions, corresponding to position, rotation, velocity, and angular velocities of the arm.
### Action Space
Each action is a vector with **4** numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between `-1 and 1`.
### Reward
A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.
### Goal
maintain the agent's hand at the target location for as many time steps as possible.
### Solving the Environment
An average score of `+30` over `100` consecutive episodes, and over all agents.

\* The version with 20 identical copies of the agent sharing the same experience is used in this experiment.

# Getting Started
## Step 1: Clone the Project and Install Dependencies

\*Please prepare a python3 virtual environment if necessary.
```
git clone https://github.com/qiaochen/A2C.git
cd install_requirements
pip install .
```

## Step 2: Download the Unity Environment
For this project, I use the environment form **Udacity**. The links to modules at different system environments are copied here for convenience:
*   Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
*   Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
*   Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
*   Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
I conducted my experiments in Ubuntu 16.04, so I picked the 1st option. Then, extract and place the Reacher_Linux folder within the project root. The project folder structure now looks like this (Program generated .png and model files are excluded):
```
Project Root
     |-install_requirements (Folder)
     |-README.md
     |-Report.md
     |-agent.py
     |-models.py
     |-train.py
     |-test.py
     |-utils.py
     |-Reacher_Linux (Folder)
            |-Reacher.x86_64
            |-Reacher.x86
            |-Reacher_Data (Folder)
```
## Instructions to the Program
---
### Step 1: Training
```
python thain.py
```
After training, the following files will be generated and placed in the project root folder:

- best_model.checkpoint (the trained model)
- training_100avgscore_plot.png (a plot of avg. scores during training)
- training_score_plot.png (a plot of per-episode scores during training)
- unity-environment.log (log file created by Unity)

### Step 2: Test
```
python test.py
```
The testing performance will be summarized in the generated plot within project root:

- test_score_plot.png



