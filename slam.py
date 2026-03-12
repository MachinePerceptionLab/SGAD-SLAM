import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
import cv2
import numpy as np
import open3d as o3d
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.graphics_utils import focal2fov
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from mp_Tracker import Tracker
from mp_Tracker_scene import Tracker_scene
import imageio

with_mapper = True
if with_mapper:
    from mp_Mapper import SLAMMapper
    from src.evaluation.evaluator import Evaluator
    from src.utils.io_utils import load_config
    from src.utils.utils import setup_seed



torch.multiprocessing.set_sharing_strategy('file_system')

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class SGAD_SLAM(SLAMParameters):
    def __init__(self, args):
        super().__init__()

        self.dataset_path = args.dataset_path
        self.config = args.config
        self.output_path = args.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = args.verbose
        self.keyframe_th = float(args.keyframe_th)
        self.knn_max_distance = float(args.knn_maxd)
        self.overlapped_th = float(args.overlapped_th)
        self.max_correspondence_distance = float(args.max_correspondence_distance)
        self.k_choice = int(args.k_choice)
        self.knn_cov = int(args.knn_cov)
        self.init_rate = float(args.init_rate)
        self.keyframe_freq_tracking = int(args.keyframe_freq_tracking)
        self.trackable_opacity_th = float(args.trackable_opacity_th)
        self.overlapped_th2 = float(args.overlapped_th2)
        self.downsample_rate = int(args.downsample_rate)
        self.test = args.test
        self.save_results = args.save_results

        
        camera_parameters_file = open(self.config)
        camera_parameters_ = camera_parameters_file.readlines()
        self.camera_parameters = camera_parameters_[2].split()
        self.W = int(self.camera_parameters[0])
        self.H = int(self.camera_parameters[1])
        self.fx = float(self.camera_parameters[2])
        self.fy = float(self.camera_parameters[3])
        self.cx = float(self.camera_parameters[4])
        self.cy = float(self.camera_parameters[5])
        self.depth_scale = float(self.camera_parameters[6])
        self.depth_trunc = float(args.depth_trunc) 
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
  
        if self.camera_parameters[8] == "scannetpp":
            self.use_pre_delta = False 
            self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path, use_train_split=self.camera_parameters[9])
        else:
            self.use_pre_delta = False
            self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        
        # Make test cam
        # To get memory sizes of shared_cam
        test_rgb_img, test_depth_img = self.get_test_image(f"{self.dataset_path}/images")
        test_points, _, _, _ = self.downsample_and_make_pointcloud(test_depth_img, test_rgb_img)

        # Get size of final poses
        num_final_poses = len(self.trajmanager.gt_poses)
        
        # Shared objects
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                                    image=test_rgb_img, depth_image=test_depth_img,
                                    cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)
        self.shared_new_points = SharedPoints(test_points.shape[0])
        self.shared_new_gaussians = SharedGaussians(test_points.shape[0])
        self.shared_target_gaussians = SharedTargetPoints(1000000000)
        self.end_of_dataset = torch.zeros((1)).int()
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()
        self.is_tracking_scene_keyframe_shared = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.final_pose = torch.zeros((num_final_poses,4,4)).float()
        self.num_poses = torch.zeros((1)).int()
        self.demo = torch.zeros((1)).int()
        self.is_tracking_scene_process_started = torch.zeros((1)).int()
        self.iter_shared = torch.zeros((1)).int()
        self.delta_ls = torch.zeros((num_final_poses,4,4)).float()
        
        self.shared_cam.share_memory()
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_tracking_scene_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_pose.share_memory_()
        self.num_poses.share_memory_()
        self.demo.share_memory_()
        self.is_tracking_scene_process_started.share_memory_()
        self.iter_shared.share_memory_()
        self.delta_ls.share_memory_()
        
        self.demo[0] = args.demo
        self.tracker = Tracker(self)
        self.tracker_scene = Tracker_scene(self)

        if with_mapper:
            # mapper
            config_mapper = load_config(args.config_map_path)
            config_mapper["use_wandb"] = False
            setup_seed(config_mapper["seed"])
            self.mapper = SLAMMapper(self, config_mapper)
        

    def tracking(self, rank):
        self.tracker.run()
    
    def tracking_scene(self, rank):
        self.tracker_scene.run()
    
    if with_mapper:
        def mapping(self, rank):
            self.mapper.run()

    def run(self):
        processes = []
        for rank in range(3):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.tracking_scene, args=(rank, )) 
            elif rank == 2 and with_mapper:
                p = mp.Process(target=self.mapping, args=(rank, ))
            
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def get_test_image(self, images_folder):
        
        if self.camera_parameters[8] == "replica":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
            rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
            depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png")).astype(np.float32)
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        elif self.camera_parameters[8] == "scannet":
            rgb_folder = os.path.join(self.dataset_path, "color")
            depth_folder = os.path.join(self.dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            rgb_image = cv2.resize(rgb_image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            depth_image = cv2.imread(os.path.join(depth_folder, depth_file), cv2.IMREAD_UNCHANGED) 
            depth_image = depth_image.astype(np.float32)
        elif self.trajmanager.which_dataset == 'scannetpp':
            rgb_folder = os.path.join(self.dataset_path, "dslr", "undistorted_images")
            depth_folder = os.path.join(self.dataset_path, "dslr", "undistorted_depths")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = np.asarray(imageio.imread(os.path.join(rgb_folder, rgb_file)), dtype=float)
            rgb_image = cv2.resize(rgb_image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            rgb_image = rgb_image.astype(np.uint8)
            depth_image = np.asarray(imageio.imread(os.path.join(depth_folder, depth_file)), dtype=np.int64)
            depth_image = cv2.resize(depth_image.astype(float), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth_image = depth_image.astype(np.float32)
            
        return rgb_image, depth_image

  

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        filter = torch.where((z_values!=0)&(z_values<=self.depth_trunc))
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values
        x = self.x_pre * z_values
        y = self.y_pre * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    
    def get_image_dirs(self, images_folder):
        if self.camera_parameters[8] == "replica":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = os.listdir(rgb_folder)
            depth_files = os.listdir(depth_folder)
 
        return image_files, depth_files


def get_args(config_path=None):
    parser = ArgumentParser(
        description='Arguments to compute the mesh')
    parser.config_path = config_path
    parser.add_argument('--input_path', default="")
    parser.add_argument('--output_path', default="")
    parser.add_argument('--track_w_color_loss', type=float)
    parser.add_argument('--track_alpha_thre', type=float)
    parser.add_argument('--track_iters', type=int)
    parser.add_argument('--track_filter_alpha', action='store_true')
    parser.add_argument('--track_filter_outlier', action='store_true')
    parser.add_argument('--track_wo_filter_alpha', action='store_true')
    parser.add_argument("--track_wo_filter_outlier", action="store_true")
    parser.add_argument("--track_cam_trans_lr", type=float)
    parser.add_argument('--alpha_seeding_thre', type=float)
    parser.add_argument('--map_every', type=int)
    parser.add_argument("--map_iters", type=int)
    parser.add_argument('--new_submap_every', type=int)
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--group_name', type=str)
    parser.add_argument('--gt_camera', action='store_true')
    parser.add_argument('--help_camera_initialization', action='store_true')
    parser.add_argument('--soft_alpha', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--submap_using_motion_heuristic', action='store_true')
    parser.add_argument('--new_submap_points_num', type=int)
    return parser.parse_args()


def update_config_with_args(config, args):
    if args.input_path:
        config["data"]["input_path"] = args.input_path
    if args.output_path:
        config["data"]["output_path"] = args.output_path
    if args.track_w_color_loss is not None:
        config["tracking"]["w_color_loss"] = args.track_w_color_loss
    if args.track_iters is not None:
        config["tracking"]["iterations"] = args.track_iters
    if args.track_filter_alpha:
        config["tracking"]["filter_alpha"] = True
    if args.track_wo_filter_alpha:
        config["tracking"]["filter_alpha"] = False
    if args.track_filter_outlier:
        config["tracking"]["filter_outlier_depth"] = True
    if args.track_wo_filter_outlier:
        config["tracking"]["filter_outlier_depth"] = False
    if args.track_alpha_thre is not None:
        config["tracking"]["alpha_thre"] = args.track_alpha_thre
    if args.map_every:
        config["mapping"]["map_every"] = args.map_every
    if args.map_iters:
        config["mapping"]["iterations"] = args.map_iters
    if args.new_submap_every:
        config["mapping"]["new_submap_every"] = args.new_submap_every
    if args.project_name:
        config["project_name"] = args.project_name
    if args.alpha_seeding_thre is not None:
        config["mapping"]["alpha_thre"] = args.alpha_seeding_thre
    if args.seed:
        config["seed"] = args.seed
    if args.help_camera_initialization:
        config["tracking"]["help_camera_initialization"] = True
    if args.soft_alpha:
        config["tracking"]["soft_alpha"] = True
    if args.submap_using_motion_heuristic:
        config["mapping"]["submap_using_motion_heuristic"] = True
    if args.new_submap_points_num:
        config["mapping"]["new_submap_points_num"] = args.new_submap_points_num
    if args.track_cam_trans_lr:
        config["tracking"]["cam_trans_lr"] = args.track_cam_trans_lr
    return config


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    parser.add_argument("--config", help="caminfo", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--k_choice", default=7)
    parser.add_argument("--knn_cov", default=10)
    parser.add_argument("--keyframe_freq_tracking", default=5)
    parser.add_argument("--depth_trunc", default=3.0)
    parser.add_argument("--init_rate", default=1.0)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action='store_true', default=None)
    parser.add_argument("--rerun_viewer", action="store_true", default=False)
    parser.add_argument("--config_map_path", type=str)
    args = parser.parse_args()




    sgad_slam = SGAD_SLAM(args)
    sgad_slam.run()

    mapper = sgad_slam.mapper

    evaluator = Evaluator(mapper.output_path, mapper.output_path / "config.yaml")
    evaluator.run()
    print("All done.✨")