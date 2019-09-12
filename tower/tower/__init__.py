from gym.envs.registration import register
 
register(id='Tower-v0', 
    entry_point='tower.envs:TowerEnv', 
)