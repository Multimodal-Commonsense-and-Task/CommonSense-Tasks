from .mounted_panda import MountedPanda, MountedPanda1
from .on_the_ground_panda import OnTheGroundPanda, OnTheGroundPanda1

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedPanda1": SingleArm,
        "OnTheGroundPanda1": SingleArm,
    }
)
