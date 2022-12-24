import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,   # 受控的车辆
            "screen_width": 1500,  # 动画的宽度，单位为像素，一个像素对应一米
            "screen_height": 120,   # 动画的高度，同上
            "centering_position": [0, 0.5],  # 参见road和commom中的graphics.py
            "scaling": 2,   # 世界坐标[m]与像素坐标[px]之间的缩放比例
            "simulation_frequency": 15,  # 每秒仿真步数[Hz]
            "duration": 20,  # 仿真秒
            "policy_frequency": 5,  # 每秒控制步数[Hz]
            "reward_speed_range": [20, 30],
            "COLLISION_REWARD": 200,  # 车辆冲突惩罚
            "HIGH_SPEED_REWARD": 1,  # 车辆速度奖励
            "HEADWAY_COST": 4,   
            "HEADWAY_TIME": 1.2,
            "FLOCKING_REWARD": 4,   # 车辆集群奖励
            "traffic_density": 1,   # 暂时没用
            "lanes_number": 3   # 添加参数，高速路车道数
        })
        return config

    # 所有AV的平均reward，即global reward
    def _reward(self, action: int) -> float:
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    # 计算单个AV的reward，即论文中的reward function
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(vehicle.speed, 
                                  self.config["reward_speed_range"], 
                                  [0, 1])
        flocking_reward = 1/np.linalg.norm(vehicle.position-self.flocking_center)

        headway_distance = self._compute_headway_distance(vehicle)
        headway_reward = np.log(headway_distance/(self.config["HEADWAY_TIME"]*vehicle.speed)) if vehicle.speed>0 else 0
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["FLOCKING_REWARD"] * flocking_reward \
                 + self.config["HEADWAY_COST"] * (headway_reward if headway_reward < 0 else 0)
        return reward

    # 计算regional reward，即local reward
    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = [vehicle]
            neighbor_lanes = self.road.network.side_lanes(vehicle.lane_index) + [vehicle.lane_index]

            for lane in neighbor_lanes:
                v_f, v_r = self.road.neighbour_vehicles(vehicle,lane)
                if type(v_f) is MDPVehicle and v_f is not None:
                    neighbor_vehicle.append(v_f)
                if type(v_r) is MDPVehicle and v_r is not None:
                    neighbor_vehicle.append(v_r)
                
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        # 更新flocking_center，用于计算集群奖励flockin_reward
        self.flocking_center = np.mean([vehicle.position for vehicle in self.controlled_vehicles])
        
        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info  # obs传入算法用于训练 2022.1.25

    def _is_terminal(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, num_CAV=0) -> None:
        self._make_road()
        # self._make_vehicles(num_CAV, num_HDV)
        self._make_vehicles(4, 1)
        # self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])
        
        # 初始化flocking_center, 车群中心
        self.flocking_center = np.mean([vehicle.position for vehicle in self.controlled_vehicles])

    def _make_road(self,) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        n_lanes = self.config["lanes_number"]

        net = RoadNetwork.straight_road_network(lanes=n_lanes,
                                                start=0,   # 前两百米用于生成车辆
                                                length=200+500,
                                                speed_limit=30,
                                                nodes_str=("a","b"))
        road = Road(network=net, 
                    np_random=self.np_random, 
                    record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        road = self.road
        n_lanes = self.config["lanes_number"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        # vehicle initial possition candidates
        spawn_points = [10*i for i in range(10)]

        """Spawn points for CAV"""
        spawn_point_c = np.random.choice(spawn_points, num_CAV, replace=False)
        spawn_point_c = list(spawn_point_c)
        for a in spawn_point_c:
            spawn_points.remove(a)

        """Spawn points for HDV"""
        #spawn_point_h = np.random.choice(spawn_points, num_HDV, replace=False)
        # spawn_point_h = list(spawn_point_h)
        spawn_point_h = [190]  # 超车场景，一辆慢速HDV生成在CAVs前方

        initial_speed = np.random.rand(num_CAV + num_HDV) * 20 + 2    # 生成车辆的初始速度
        # initial_speed = np.zeros(num_CAV + num_HDV)
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5    # 给车辆的生成位置加点噪声
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """spawn the CAV on the straight road"""
        for _ in range(num_CAV):
            ego_vehicle = self.action_type.vehicle_class(road,
                                                         road.network.get_lane(("a","b",np.random.choice(n_lanes))).position(spawn_point_c.pop(0) + loc_noise.pop(0), 0),
                                                         speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the straight road"""
        for _ in range(num_HDV):
            road.vehicles.append(
                other_vehicles_type(road,
                                    road.network.get_lane(("a","b",np.random.choice(n_lanes))).position(spawn_point_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0),
                                    target_speed=10))  # 设置HDV期望速度

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 6
        })
        return config


register(
    id='merge-v1',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='highway_env.envs:MergeEnvMARL',
)
