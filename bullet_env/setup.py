from setuptools import setup, find_packages

setup(
    name="bullet_env",
    packages=find_packages(),
    package_data={'bullet_env': ['robot/robot.urdf']},
    version="0.0.1",
)