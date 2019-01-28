"""Makes bipedal walker hardcore easier."""
from gym.envs.box2d import bipedal_walker
import numpy as np
import math

class Walker(bipedal_walker.BipedalWalker):
    def __init__(self, dt: float) -> None:
        bipedal_walker.REF_FPS = 50
        bipedal_walker.FPS = 1. / dt
        super().__init__()

    def step(self, action):
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(bipedal_walker.SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(bipedal_walker.SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(bipedal_walker.SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(bipedal_walker.SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(bipedal_walker.SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(bipedal_walker.MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(bipedal_walker.SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(bipedal_walker.MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(bipedal_walker.SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(bipedal_walker.MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(bipedal_walker.SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(bipedal_walker.MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0/bipedal_walker.FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*bipedal_walker.LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*bipedal_walker.LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/bipedal_walker.FPS,
            0.3*vel.x*(bipedal_walker.VIEWPORT_W/bipedal_walker.SCALE)/bipedal_walker.FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(bipedal_walker.VIEWPORT_H/bipedal_walker.SCALE)/bipedal_walker.FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / bipedal_walker.SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / bipedal_walker.SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / bipedal_walker.SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / bipedal_walker.SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==24

        self.scroll = pos.x - bipedal_walker.VIEWPORT_W/bipedal_walker.SCALE/5

        shaping  = 130*pos[0]/bipedal_walker.SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = (shaping - self.prev_shaping) * bipedal_walker.FPS / bipedal_walker.REF_FPS
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * bipedal_walker.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (bipedal_walker.TERRAIN_LENGTH-bipedal_walker.TERRAIN_GRASS)*bipedal_walker.TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

class WalkerHardcore(bipedal_walker.BipedalWalkerHardcore):
    def __init__(self, dt: float) -> None:
        bipedal_walker.REF_FPS = 50
        bipedal_walker.FPS = 1. / dt
        super().__init__()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if reward == -100:
            reward = 0
        return obs, reward, done, info
