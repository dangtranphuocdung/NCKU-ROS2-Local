from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rebotarm_ibvs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'ultralytics', 'opencv-python'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='IBVS grasping pipeline for reBot arm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_vision_node = rebotarm_ibvs.yolo_vision_node:main',
            'ibvs_controller_node = rebotarm_ibvs.ibvs_controller_node:main',
            'grasp_manager_node = rebotarm_ibvs.grasp_manager_node:main',
            'hardware_servo_bridge_node = rebotarm_ibvs.hardware_servo_bridge_node:main',
        ],
    },
)

