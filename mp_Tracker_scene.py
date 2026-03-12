import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import numpy as np
import time
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from scene import TrajGaussianModel
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class Tracker_scene(SLAMParameters):
    def __init__(self, slam):   
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
        self.iter_shared = slam.iter_shared

        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])
        
        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_tracking_scene_keyframe = False
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_tracking_scene_keyframe = 0
        self.scene_extent = 2.5
        if self.trajmanager.which_dataset == "replica":
            self.prune_th = 2.5
        else:
            self.prune_th = 10.0
        
        self.tracking_gaussians = TrajGaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.tracking_scene_cams = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []
        
        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_tracking_scene_keyframe_shared = slam.is_tracking_scene_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.is_tracking_scene_process_started = slam.is_tracking_scene_process_started
    
    def run(self):
        self.tracking_scene()
    
    def tracking_scene(self):
        point_w_color = True
        t = torch.zeros((1,1)).float().cuda()
        
        
        # tracking_scene Process is ready to receive first frame
        self.is_tracking_scene_process_started[0] = 1
        
        # Wait for initial gaussians
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)
            
        self.total_start_time_viewer = time.time()
        
        points, colors, rots, scales, z_values, trackable_filter = self.shared_new_gaussians.get_values()
        self.tracking_gaussians.create_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter)
        self.tracking_gaussians.spatial_lr_scale = self.scene_extent
        self.tracking_gaussians.training_setup(self)
        self.tracking_gaussians.update_learning_rate(1)
        self.tracking_gaussians.active_sh_degree = self.tracking_gaussians.max_sh_degree
        self.is_tracking_keyframe_shared[0] = 0
        
        

        new_keyframe = False
        while True:
            if self.end_of_dataset[0]:
                break   
            
            if self.is_tracking_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, trackable_filter = self.shared_new_gaussians.get_values()
                
                # Add new gaussians to map gaussians
                self.tracking_gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter)

                # Allocate new target points to shared memory
                target_points, target_rots, target_scales  = self.tracking_gaussians.get_trackable_gaussians_tensor(self.trackable_opacity_th)
                self.shared_target_gaussians.input_values(target_points, target_rots, target_scales)

                self.target_gaussians_ready[0] = 1

                self.is_tracking_keyframe_shared[0] = 0


       
