import collections
import numpy as np
import os
import robosuite
import xml.etree.ElementTree as ET

from copy import copy
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion
from robosuite.utils.placement_samplers import ObjectPositionSampler


class MultiRegionRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.
    Args:
        name (str): Name of this sampler.
        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models
        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects
        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects
        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation
        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation
        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): If True, will check for correct (valid) object placements
        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        assert len(self.x_ranges) == len(self.y_ranges)
        self.num_ranges = len(self.x_ranges)
        self.idx = 0
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.idx = 0

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)

                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")
        # print(placed_objects)
        return placed_objects


def postprocess_model_xml(xml_str, cameras_dict={}, demo_generation=False):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    # also replace paths for libero
    libero_path = os.getcwd() + "/libero"
    libero_path_split = libero_path.split("/")

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" in old_path_split:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            )  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        elif "libero" in old_path_split and demo_generation:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "libero"
            )  # last occurrence index
            new_path_split = libero_path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        else:
            continue

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")

    return ET.tostring(root, encoding="utf8").decode("utf8")


def rectangle2xyrange(rect_ranges):
    x_ranges = []
    y_ranges = []
    for rect_range in rect_ranges:
        x_ranges.append([rect_range[0], rect_range[2]])
        y_ranges.append([rect_range[1], rect_range[3]])
    return x_ranges, y_ranges


def rotate_quaternion(quat, axis, angle_degrees):
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    """Rotate a quaternion around a given axis by a specified angle in degrees."""
    w, x, y, z = quat
    r = R.from_quat([x, y, z, w])
    rotation = R.from_rotvec(np.radians(angle_degrees) * np.array(axis))
    rotated_r = rotation * r
    x, y, z, w = rotated_r.as_quat()
    return [w, x, y, z]
    
def setup_camera_views(mujoco_arena, env):
        """Sets up the cameras for the arena.

        Args:
            mujoco_arena (MujocoArena): The mujoco arena instance to set up cameras for.
            env (str): The environment name.
        """

        # mujoco_arena.set_camera(
        #     camera_name="canonical_agentview",
        #     pos=[0.5386131746834771, 0.0, 1.4903500240372423],
        #     quat=[
        #         0.6380177736282349,
        #         0.3048497438430786,
        #         0.30484986305236816,
        #         0.6380177736282349,
        #     ],
        # )

        # mujoco_arena.set_camera(
        #     camera_name="agentview",
        #     pos=[0.5886131746834771, 0.0, 1.4903500240372423],
        #     quat=[
        #         0.6380177736282349,
        #         0.3048497438430786,
        #         0.30484986305236816,
        #         0.6380177736282349,
        #     ],
        # )

        def angle_between(a, b):
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)

            # 정규화
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)

            # 2D 외적의 z성분
            cross_z = a[0]*b[1] - a[1]*b[0]
            dot = np.dot(a, b)

            angle_rad = np.arctan2(cross_z, dot)
            return np.degrees(angle_rad)
            
        delta_pos_small = [0.0, 0.3, -0.1]
        delta_pos_medium = [-0.2, 0.7, -0.2]
        delta_pos_large = [-1.2, 1.0, -0.2]

        if env == "coffee_table":
            pos=[1.5, 0.0, 0.9], 
            quat=[0.56, 0.43, 0.43, 0.56]

            pos_small = (np.array(pos) + np.array(delta_pos_small)).tolist()
            pos_medium = (np.array(pos) + np.array(delta_pos_medium)).tolist()
            pos_large = (np.array(pos) + np.array(delta_pos_large)).tolist()

            angle_z_small = angle_between(pos[0:2], pos_small[0:2])
            angle_y_small = angle_between(pos[0::2], [float(np.hypot(pos_small[0], pos_small[1])), float(pos_small[2])])
            
            angle_z_medium = angle_between(pos[0:2], pos_medium[0:2])
            angle_y_medium = angle_between(pos[0::2], [float(np.hypot(pos_medium[0], pos_medium[1])), float(pos_medium[2])])

            angle_z_large = angle_between(pos[0:2], pos_large[0:2])
            angle_y_large = angle_between(pos[0::2], [float(np.hypot(pos_large[0], pos_large[1])), float(pos_large[2])])

            mujoco_arena.set_camera(
                camera_name="agentview_small",
                pos=pos_small,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_small),
                    axis=[0,0,1],
                    angle_degrees=angle_z_small)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_medium",
                pos=pos_medium,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_medium),
                    axis=[0,0,1],
                    angle_degrees=angle_z_medium)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_large",
                pos=pos_large,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_large),
                    axis=[0,0,1],
                    angle_degrees=angle_z_large)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_right",
                pos=[0.5886131746834771, 0.0-0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-35)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_left",
                pos=[0.5886131746834771, 0.0+0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=35)
            )


            mujoco_arena.set_camera(
                camera_name="agentview_right_back",
                pos=[0.5886131746834771-1.35, 0.0-0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-140)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_left_back",
                pos=[0.5886131746834771-1.35, 0.0+0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=140)
            )

        elif env == "floor":
            pos=[0.8965773716836134, 5.216182733499864e-07, 0.65]
            quat=[
                0.6182166934013367,
                0.3432307541370392,
                0.3432314395904541,
                0.6182177066802979,
            ]
            pos_small = (np.array(pos) + np.array(delta_pos_small)).tolist()
            pos_medium = (np.array(pos) + np.array(delta_pos_medium)).tolist()
            pos_large = (np.array(pos) + np.array(delta_pos_large)).tolist()

            angle_z_small = angle_between(pos[0:2], pos_small[0:2])
            angle_y_small = angle_between(pos[0::2], [float(np.hypot(pos_small[0], pos_small[1])), float(pos_small[2])])
            
            angle_z_medium = angle_between(pos[0:2], pos_medium[0:2])
            angle_y_medium = angle_between(pos[0::2], [float(np.hypot(pos_medium[0], pos_medium[1])), float(pos_medium[2])])

            angle_z_large = angle_between(pos[0:2], pos_large[0:2])
            angle_y_large = angle_between(pos[0::2], [float(np.hypot(pos_large[0], pos_large[1])), float(pos_large[2])])

            mujoco_arena.set_camera(
                camera_name="agentview_small",
                pos=pos_small,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_small),
                    axis=[0,0,1],
                    angle_degrees=angle_z_small)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_medium",
                pos=pos_medium,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_medium),
                    axis=[0,0,1],
                    angle_degrees=angle_z_medium)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_large",
                pos=pos_large,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_large),
                    axis=[0,0,1],
                    angle_degrees=angle_z_large)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_right",
                pos=[0.5886131746834771, 0.0-0.5, 1.4903500240372423-0.6],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-35)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_left",
                pos=[0.5886131746834771, 0.0+0.5, 1.4903500240372423-0.6],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=35)
            )


            mujoco_arena.set_camera(
                camera_name="agentview_right_back",
                pos=[0.5886131746834771-1.35, 0.0-0.5, 1.4903500240372423-0.6],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-140)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_left_back",
                pos=[0.5886131746834771-1.35, 0.0+0.5, 1.4903500240372423-0.6],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=140)
            )

        elif env == "kitchen":
            pos=[0.6586131746834771, 0.0, 1.6103500240372423]
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ]
            pos_small = (np.array(pos) + np.array(delta_pos_small)).tolist()
            pos_medium = (np.array(pos) + np.array(delta_pos_medium)).tolist()
            pos_large = (np.array(pos) + np.array(delta_pos_large)).tolist()

            angle_z_small = angle_between(pos[0:2], pos_small[0:2])
            angle_y_small = angle_between(pos[0::2], [float(np.hypot(pos_small[0], pos_small[1])), float(pos_small[2])])
            
            angle_z_medium = angle_between(pos[0:2], pos_medium[0:2])
            angle_y_medium = angle_between(pos[0::2], [float(np.hypot(pos_medium[0], pos_medium[1])), float(pos_medium[2])])

            angle_z_large = angle_between(pos[0:2], pos_large[0:2])
            angle_y_large = angle_between(pos[0::2], [float(np.hypot(pos_large[0], pos_large[1])), float(pos_large[2])])

            mujoco_arena.set_camera(
                camera_name="agentview_small",
                pos=pos_small,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_small),
                    axis=[0,0,1],
                    angle_degrees=angle_z_small)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_medium",
                pos=pos_medium,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_medium),
                    axis=[0,0,1],
                    angle_degrees=angle_z_medium)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_large",
                pos=pos_large,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_large),
                    axis=[0,0,1],
                    angle_degrees=angle_z_large)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_right",
                pos=[0.5886131746834771, 0.0-0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-35)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_left",
                pos=[0.5886131746834771, 0.0+0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=35)
            )


            mujoco_arena.set_camera(
                camera_name="agentview_right_back",
                pos=[0.5886131746834771-1.2, 0.0-0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-140)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_left_back",
                pos=[0.5886131746834771-1.2, 0.0+0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=140)
            )

        elif env == "living_room":
            pos=[0.6065773716836134, 0.0, 0.96]
            quat=[
                0.6182166934013367,
                0.3432307541370392,
                0.3432314395904541,
                0.6182177066802979,
            ]
            pos_small = (np.array(pos) + np.array(delta_pos_small)).tolist()
            pos_medium = (np.array(pos) + np.array(delta_pos_medium)).tolist()
            pos_large = (np.array(pos) + np.array(delta_pos_large)).tolist()

            angle_z_small = angle_between(pos[0:2], pos_small[0:2])
            angle_y_small = angle_between(pos[0::2], [float(np.hypot(pos_small[0], pos_small[1])), float(pos_small[2])])
            
            angle_z_medium = angle_between(pos[0:2], pos_medium[0:2])
            angle_y_medium = angle_between(pos[0::2], [float(np.hypot(pos_medium[0], pos_medium[1])), float(pos_medium[2])])

            angle_z_large = angle_between(pos[0:2], pos_large[0:2])
            angle_y_large = angle_between(pos[0::2], [float(np.hypot(pos_large[0], pos_large[1])), float(pos_large[2])])

            mujoco_arena.set_camera(
                camera_name="agentview_small",
                pos=pos_small,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_small),
                    axis=[0,0,1],
                    angle_degrees=angle_z_small)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_medium",
                pos=pos_medium,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_medium),
                    axis=[0,0,1],
                    angle_degrees=angle_z_medium)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_large",
                pos=pos_large,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_large),
                    axis=[0,0,1],
                    angle_degrees=angle_z_large)
            )
        
            mujoco_arena.set_camera(
                    camera_name="agentview_right",
                    pos=[0.5886131746834771, 0.0-0.5, 1.4903500240372423-0.3],
                    quat=rotate_quaternion([
                        0.6380177736282349,
                        0.3048497438430786,
                        0.30484986305236816,
                        0.6380177736282349,
                        ],
                        axis=[0, 0, 1],
                        angle_degrees=-35)
                )

            mujoco_arena.set_camera(
                camera_name="agentview_left",
                pos=[0.5886131746834771, 0.0+0.5, 1.4903500240372423-0.3],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=35)
            )


            mujoco_arena.set_camera(
                camera_name="agentview_right_back",
                pos=[0.5886131746834771-1.2, 0.0-0.5, 1.4903500240372423-0.3],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-140)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_left_back",
                pos=[0.5886131746834771-1.2, 0.0+0.5, 1.4903500240372423-0.3],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=140)
            )
        elif env == "study_tabletop":
            pos=[0.4586131746834771, 0.0, 1.6103500240372423]
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ]
            pos_small = (np.array(pos) + np.array(delta_pos_small)).tolist()
            pos_medium = (np.array(pos) + np.array(delta_pos_medium)).tolist()
            pos_large = (np.array(pos) + np.array(delta_pos_large)).tolist()

            angle_z_small = angle_between(pos[0:2], pos_small[0:2])
            angle_y_small = angle_between(pos[0::2], [float(np.hypot(pos_small[0], pos_small[1])), float(pos_small[2])])
            
            angle_z_medium = angle_between(pos[0:2], pos_medium[0:2])
            angle_y_medium = angle_between(pos[0::2], [float(np.hypot(pos_medium[0], pos_medium[1])), float(pos_medium[2])])

            angle_z_large = angle_between(pos[0:2], pos_large[0:2])
            angle_y_large = angle_between(pos[0::2], [float(np.hypot(pos_large[0], pos_large[1])), float(pos_large[2])])

            mujoco_arena.set_camera(
                camera_name="agentview_small",
                pos=pos_small,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_small),
                    axis=[0,0,1],
                    angle_degrees=angle_z_small)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_medium",
                pos=pos_medium,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_medium),
                    axis=[0,0,1],
                    angle_degrees=angle_z_medium)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_large",
                pos=pos_large,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_large),
                    axis=[0,0,1],
                    angle_degrees=angle_z_large)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_right",
                pos=[0.5886131746834771, 0.0-0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-35)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_left",
                pos=[0.5886131746834771, 0.0+0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=35)
            )


            mujoco_arena.set_camera(
                camera_name="agentview_right_back",
                pos=[0.5886131746834771-1.35, 0.0-0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-140)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_left_back",
                pos=[0.5886131746834771-1.35, 0.0+0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=140)
            )
            
        elif env == "tabletop":
            pos=[0.6586131746834771, 0.0, 1.6103500240372423]
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ]
            pos_small = (np.array(pos) + np.array(delta_pos_small)).tolist()
            pos_medium = (np.array(pos) + np.array(delta_pos_medium)).tolist()
            pos_large = (np.array(pos) + np.array(delta_pos_large)).tolist()

            angle_z_small = angle_between(pos[0:2], pos_small[0:2])
            angle_y_small = angle_between(pos[0::2], [float(np.hypot(pos_small[0], pos_small[1])), float(pos_small[2])])
            
            angle_z_medium = angle_between(pos[0:2], pos_medium[0:2])
            angle_y_medium = angle_between(pos[0::2], [float(np.hypot(pos_medium[0], pos_medium[1])), float(pos_medium[2])])

            angle_z_large = angle_between(pos[0:2], pos_large[0:2])
            angle_y_large = angle_between(pos[0::2], [float(np.hypot(pos_large[0], pos_large[1])), float(pos_large[2])])

            mujoco_arena.set_camera(
                camera_name="agentview_small",
                pos=pos_small,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_small),
                    axis=[0,0,1],
                    angle_degrees=angle_z_small)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_medium",
                pos=pos_medium,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_medium),
                    axis=[0,0,1],
                    angle_degrees=angle_z_medium)
            )
            mujoco_arena.set_camera(
                camera_name="agentview_large",
                pos=pos_large,
                quat=rotate_quaternion(
                        rotate_quaternion(
                            quat,
                            axis=[0,-1,0],
                            angle_degrees=angle_y_large),
                    axis=[0,0,1],
                    angle_degrees=angle_z_large)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_right",
                pos=[0.5886131746834771, 0.0-0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-35)
            )

            mujoco_arena.set_camera(
                camera_name="agentview_left",
                pos=[0.5886131746834771, 0.0+0.5, 1.4903500240372423+0.2],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=35)
            )


            mujoco_arena.set_camera(
                camera_name="agentview_right_back",
                pos=[0.5886131746834771-1.35, 0.0-0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=-140)
            )
            
            mujoco_arena.set_camera(
                camera_name="agentview_left_back",
                pos=[0.5886131746834771-1.35, 0.0+0.5, 1.4903500240372423+0.1],
                quat=rotate_quaternion([
                    0.6380177736282349,
                    0.3048497438430786,
                    0.30484986305236816,
                    0.6380177736282349,
                    ],
                    axis=[0, 0, 1],
                    angle_degrees=140)
            )