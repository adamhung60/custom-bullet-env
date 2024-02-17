from gymnasium.envs.registration import register

register(
     id="Bullet",
     entry_point="bullet_env.envs:BulletEnv",
)