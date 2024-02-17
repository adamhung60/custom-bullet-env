import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources
import cv2
import time

class BulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        self.max_steps = 1000
        self.steps_taken = 0
        self.healthy_z_range = [0.12,1.0]
        self.joints = [1, 2]
        self.forces = [500, 500]
        self.prev_pos = 0
        self.render_mode = render_mode
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.resetSimulation()

        # y displacement (green), z displacement (blue)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=float)
        # left wheel, right wheel
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)

        #self.clock = None

    def get_obs(self):
        y,z = p.getLinkState(self.robot, 0)[0][1:3]
        orientation = p.getLinkState(self.robot, 0)[1]
        joint_pos =  [p.getJointState(self.robot, 1)[0], p.getJointState(self.robot, 2)[0]]
        joint_vel =  [p.getJointState(self.robot, 1)[1], p.getJointState(self.robot, 2)[1]]
        return np.array([y, z, orientation[0], orientation[1], orientation[2], orientation[3], joint_pos[0], joint_pos[1], joint_vel[0], joint_vel[1]])
    
    def is_dead(self):
        z = p.getLinkState(self.robot,0)[0][2]

        # Create a rigid body (a point mass) at the specified coordinates
        #p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.point_visual,basePosition=z)
        #print(z)
        min_z, max_z = self.healthy_z_range
        is_healthy = min_z <= z <= max_z
        return not is_healthy
    
    def reset(self,seed=None, options = None): 
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.steps_taken = 0
        self.prev_pos = 0
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        p.loadURDF("plane.urdf")
        urdf_path = pkg_resources.resource_filename(__name__, 'robot/robot.urdf')
        self.robot = p.loadURDF(urdf_path, [0,0,0.195],useFixedBase = 0)
        #self.point_visual = p.createVisualShape(p.GEOM_SPHERE, radius=.02, rgbaColor=[1, 0, 0, 1], specularColor=[0.4, 0.4, 0])

        self.terminated = False
        self.truncated = False

        observation = self.get_obs()
        info = {"info":"hi"}

        #if self.render_mode == "human":
        #    self.render()

        return observation, info
    
    def step(self, action):
        action = action*20
        #print(action)
        p.setJointMotorControlArray(
            bodyUniqueId = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.VELOCITY_CONTROL, 
            targetVelocities = [action, -action], 
            forces = self.forces)
        p.stepSimulation()
        #print(action)
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            #print("truncated")
            self.truncated = True

        dead = self.is_dead()
        if dead:
            #print("z exceeded")
            self.terminated = True

        observation = self.get_obs()

        forward_reward = 0
        if observation[0] - self.prev_pos > 0:
            forward_reward = observation[0] - self.prev_pos
        reward = forward_reward + 0.002*(not dead)

        #reward = not dead

        self.prev_pos = observation[0]

        if self.render_mode == "human":
            self.render_frame()

        info = {"info":"hi"}

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        
        focus_position,_ = p.getBasePositionAndOrientation(self.robot)
        p.resetDebugVisualizerCamera(
            cameraDistance=1, 
            cameraYaw = 30, 
            cameraPitch = -30, 
            cameraTargetPosition = focus_position
        )
        if self.render_mode == "rgb_array":
            h,w = 4000, 4000
            image = np.array(p.getCameraImage(h, w)[2]).reshape(h,w,4)
            image = image[:, :, :3]
            image = cv2.convertScaleAbs(image)
            return image
            
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
        #self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        return