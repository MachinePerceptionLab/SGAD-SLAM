import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.visual_odometer import VisualOdometer
import imageio

class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = f'{slam.output_path}/{self.dataset_path.split("/")[-1]}'
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output path: {self.output_path}")
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
        self.test = slam.test
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
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq_tracking = slam.keyframe_freq_tracking
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.k_choice = slam.k_choice
        self.knn_cov = slam.knn_cov
        self.init_rate = slam.init_rate
        self.reg = pygicp.FastGICP()
        
        # Camera poses
        if self.camera_parameters[8] == "scannetpp":
            self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path, use_train_split=self.camera_parameters[9])
        else:
            self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # Share
        self.train_iter = 0
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []


        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_tracking_scene_keyframe_shared = slam.is_tracking_scene_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.num_poses = slam.num_poses
        self.is_tracking_scene_process_started = slam.is_tracking_scene_process_started
        self.delta_ls = slam.delta_ls
        self.use_pre_delta = slam.use_pre_delta
    
        
    def run(self):
        self.tracking()
    
    def tracking(self):
        tt = torch.zeros((1,1)).float().cuda()
        mean_dist_ls = []
        mean_point2plane_dist_ls = []

        
        self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path}/images")
        self.num_images = len(self.rgb_images)
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        self.reg.set_correspondence_randomness(self.knn_cov) #10
        self.reg.set_k_choice_p2p(self.k_choice) # 7




        self.total_start_time = time.time()
        pbar = tqdm(total=self.num_images)

        final_tracking_keyframe_num = 0
        final_tracking_keyframe_ls = []
        ate_ls = []

        self.odometer = VisualOdometer(self.cam_intrinsic, "point_to_plane", tt.device) # "point_to_plane" "hybrid"
    


        for ii in range(self.num_images):
            self.iter_shared[0] = ii
            current_image = self.rgb_images[ii] 
            depth_image = self.depth_images[ii] 
            current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                
        
            points, colors, z_values, trackable_filter, downsample_idx = self.downsample_and_make_pointcloud2(depth_image, current_image)
            

            # GICP
            if self.iteration_images == 0:
                current_pose = self.poses[-1]
                
                    
                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose) # c2w
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()
                
                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T) # world coordinate


                # Original GICP
                self.reg.set_input_target(points)
                
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                
                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots = self.reg.get_target_rotationsq()
                scales = self.reg.get_target_scales()
                rots = np.reshape(rots, (-1,4))
                scales = np.reshape(scales, (-1,3))


                
                # Assign first gaussian to shared memory
                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))
                
                # Add first keyframe
                depth_image = depth_image.astype(np.float32)/self.depth_scale
            
                self.is_tracking_keyframe_shared[0] = 1
                
                


            else:
                


                # Original GICP
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)

                if self.trajmanager.which_dataset == "replica" or self.trajmanager.which_dataset == "tum":
                    ##### previous pose initialization
                    initial_pose = self.poses[-1] # previous pose initialization
                    current_pose = self.reg.align(initial_pose)
                    self.poses.append(current_pose)

                elif self.trajmanager.which_dataset == "scannet":
                    ###### const speed initialization
                    if self.iteration_images == 1:
                        initial_pose = self.poses[-1]
                    else:
                        initial_pose = self.init_rate * self.poses[-1] @ np.linalg.inv(self.poses[-2]) @ self.poses[-1]
                    current_pose = self.reg.align(initial_pose)
                    self.poses.append(current_pose)

                elif self.trajmanager.which_dataset == "scannetpp":
                    if self.use_pre_delta:
                        delta_ls = np.load(F'{self.dataset_path}/delta_ls.npy') # using saved delta_ls.npy from rendering for tracking init
                        initial_pose = self.poses[-1] @ delta_ls[self.iteration_images-1]
                    else:
                        delta_ls = self.delta_ls
                        while torch.all(delta_ls[self.iteration_images-1]==0):
                            time.sleep(1e-15)
                        print(f"Get delta_ls for image {self.iteration_images-1}")
                        initial_pose = self.poses[-1] @ delta_ls[self.iteration_images-1].cpu().numpy()
                    current_pose = self.reg.align(initial_pose)
                    self.poses.append(current_pose)

    




        
                num_poses = len(self.poses)
                self.final_pose[:num_poses,:,:] = torch.tensor(self.poses).float()
                self.num_poses[:] = num_poses

                
                # export ply in GS format
                if self.iteration_images > 60:
                    export_ply = False 
                else:
                    export_ply = False
                if export_ply:
                    current_pose = np.linalg.inv(current_pose)
                    T = current_pose[:3,3]
                    R = current_pose[:3,:3].transpose()

                    # transform current points
                    points_w = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T) # world coordinate
                    rots_c = np.array(self.reg.get_source_rotationsq())
                    rots_c = np.reshape(rots_c, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots_w = self.quaternion_multiply(R_d_q, rots_c)
                    
                    scales_w = np.array(self.reg.get_source_scales())
                    scales_w = np.reshape(scales_w, (-1,3))
                    scales_w = np.log(scales_w)
                    


                    colors_w = colors
                    colors_w = np.ones((colors_w.shape[0], 3)) * 0.5
                    opacities_w = np.ones((colors_w.shape[0],1))

                   
                    

                    from export_ply import save_ply
                    ply_path = f'source_gs_vis_{self.iteration_images}.ply'
                    save_ply(ply_path, points_w, scales_w, rots_w, colors_w, opacities_w)
                    current_pose = self.poses[-1]



                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()

            

                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Use only trackable points when tracking
                target_corres, distances, point2plane_dists = self.reg.get_source_correspondence() # get associated points source points
                mean_dist_ls.append(np.mean(distances))
                mean_point2plane_dist_ls.append(np.mean(point2plane_dists))

               
                
                # Keyframe selection #
                # Tracking keyframe
                len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th
                
                
                if self.iteration_images % self.keyframe_freq_tracking == 0:
                    if_tracking_keyframe = True
                    final_tracking_keyframe_num += 1
                    final_tracking_keyframe_ls.append(self.iteration_images)
                else:
                    if_tracking_keyframe = False
     
                
                if if_tracking_keyframe:
                    
                    while self.is_tracking_keyframe_shared[0] or self.is_tracking_scene_keyframe_shared[0]:
                        time.sleep(1e-15)
                    
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                    
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))
                    
                    # Erase overlapped points from current pointcloud before adding to map gaussian #
                    # Using filter
                    not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                    trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]
                    
                    # Add new gaussians
                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))

                    # Add new keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale

                    
                    self.is_tracking_keyframe_shared[0] = 1
                    
                    # Get new target point
                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)
                    target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
                   
                    self.reg.set_input_target(target_points)
                    self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                    self.target_gaussians_ready[0] = 0

                
            pbar.update(1)
            
            

            traj = np.array([x[:3, 3] for x in self.poses])
            gt_traj_vis = np.array([x[:3, 3] for x in self.trajmanager.gt_poses])
            
 
            
            if ((ii % 10 == 0) or (ii == self.num_images-1)):
                valid = ~np.any(np.isnan(self.trajmanager.gt_poses) | np.isinf(self.trajmanager.gt_poses), axis=(1, 2))
                valid = valid[:self.iteration_images+1]
                gt_poses_curr = self.trajmanager.gt_poses[:self.iteration_images+1]
                poses_curr = self.poses[:self.iteration_images+1]
                gt_poses_curr = np.array(gt_poses_curr)[valid]
                poses_curr = np.array(poses_curr)[valid]
                ate = self.evaluate_ate(gt_poses_curr, poses_curr)*100
                ate_ls.append(round(ate, 2))
 
                plt.clf()
                plt.title(f'Downsample ratio {self.downsample_rate},  ate: {ate:.2f}')
                plt.plot(traj[:, 0], traj[:, 1], label='g-icp trajectory', linewidth=3)
                plt.legend()
                plt.plot(gt_traj_vis[:, 0], gt_traj_vis[:, 1], label='ground truth trajectory')
                plt.legend()
                plt.axis('equal')
                plt.pause(0.01)
                plt.savefig(f'{self.output_path}/trajectory_downsample{self.downsample_rate}.png')

            self.iteration_images += 1
        
        # Tracking end
        pbar.close()
        self.final_pose[:,:,:] = torch.tensor(self.poses).float()
        self.end_of_dataset[0] = 1
        
        from matplotlib import pyplot
        traj = np.array([x[:3, 3] for x in self.poses])
        gt_traj_vis = np.array([x[:3, 3] for x in self.trajmanager.gt_poses])
        valid = ~np.any(np.isnan(self.trajmanager.gt_poses) | np.isinf(self.trajmanager.gt_poses), axis=(1, 2))
        traj = traj[valid]
        gt_traj_vis = gt_traj_vis[valid]
        self.poses = np.array(self.poses)[valid]
        self.trajmanager.gt_poses = np.array(self.trajmanager.gt_poses)[valid]
        pyplot.clf()
        pyplot.plot(traj[:, 0], traj[:, 1], label='g-icp trajectory', linewidth=3)
        pyplot.legend()
        pyplot.plot(gt_traj_vis[:, 0], gt_traj_vis[:, 1], label='ground truth trajectory')
        pyplot.legend()
        pyplot.axis('equal')
        pyplot.pause(0.01)
        pyplot.title(f'Downsample ratio 5\nfps : {1/((time.time()-self.total_start_time)/self.num_images):.2f}   ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}')
        pyplot.savefig(f'{self.output_path}/trajectory_downsample{self.downsample_rate}_ate_kfreq{self.keyframe_freq_tracking:02d}_kforcov{self.knn_cov:02d}_trunc{self.depth_trunc}_kthreh{self.keyframe_th}_kforcorr{self.k_choice}_overt{self.overlapped_th2}_initrate{self.init_rate}.png')
        
        print(f"System FPS: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
        print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}")
        print(f"{final_tracking_keyframe_num} tracking keyframes")
        print(f"tracking keyframes: {final_tracking_keyframe_ls}")



        if self.trajmanager.which_dataset == "replica" or self.trajmanager.which_dataset == "tum" or self.trajmanager.which_dataset == "scannetpp" or self.trajmanager.which_dataset == "scannet":
            print(f"length of camera poses: {len(self.poses)}")
            # save poses in a npy file
            np.save(f'{self.output_path}/poses.npy', self.poses)

        with open(f'{self.output_path}/all_ate.txt', 'a') as f:
            f.write(f'downsample{self.downsample_rate}_ate_kfreq{self.keyframe_freq_tracking:02d}_kforcov{self.knn_cov:02d}_trunc{self.depth_trunc}_kthreh{self.keyframe_th}_kforcorr{self.k_choice}_overt{self.overlapped_th2}_initrate{self.init_rate}_lenkf{final_tracking_keyframe_num}  ')
            f.write(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}\n")
            




    
    def get_images(self, images_folder):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset == "replica":
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files): 
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                
                rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png"))
                
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == "tum":
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == 'scannet':
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = cv2.imread(self.trajmanager.depth_paths[i], cv2.IMREAD_UNCHANGED) #np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_image = cv2.resize(rgb_image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == 'scannetpp':
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = np.asarray(imageio.imread(self.trajmanager.color_paths[i]), dtype=float)
                rgb_image = cv2.resize(rgb_image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                rgb_image = rgb_image.astype(np.uint8)
                depth_image = np.asarray(imageio.imread(self.trajmanager.depth_paths[i]), dtype=np.int64)
                depth_image = cv2.resize(depth_image.astype(float), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                depth_image = depth_image.astype(np.float32)
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images


        

    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        # q1 = [x, y, z, w]
       
        x0, y0, z0, w0 = q1
        
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

    def set_downsample_filter(self, downsample_scale):
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
    
    def set_downsample_filter_beginidx(self, downsample_scale, begin_idx):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val + begin_idx - 1
        h_val[0] = max(0, h_val[0])
        h_val[-1] = min(self.H-1, h_val[-1])
        
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(begin_idx,self.W,sample_interval)) # 0, self.W, sample_interval
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

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)

        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)

        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        pts_idx = self.downsample_idxs[0][zero_filter]
       
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), pts_idx.numpy()
    
    def downsample_and_make_pointcloud2_with_downsamplerate(self, depth_img, rgb_img, downsample_scale):

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

        downsample_idxs = pick_idxs
        
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = x_pre[zero_filter] * z_values
        y = y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    


    def keypoints_and_make_pointcloud2(self, depth_img, rgb_img, keypoints_ids):
        downsample_idxs, x_pre, y_pre = self.set_downsample_filter(1)
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[downsample_idxs]/self.depth_scale
        zero_mask = z_values!=0

        # remove same keypoints
        keypoints_ids = np.unique(keypoints_ids, return_index=False)

        zero_mask_keypoints = zero_mask[keypoints_ids]
        z_values = z_values[keypoints_ids]
        colors = colors[keypoints_ids]
        x = x_pre[keypoints_ids] * z_values
        y = y_pre[keypoints_ids] * z_values
        points = torch.stack([x,y,z_values], dim=-1)


        points = points[zero_mask_keypoints]
        colors = colors[zero_mask_keypoints]
        z_values = z_values[zero_mask_keypoints]
        filter = torch.where(z_values<=self.depth_trunc)

        pts_idx = keypoints_ids[zero_mask_keypoints]
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), pts_idx
    

    def keypoints_and_make_pointcloud2_floatuv(self, depth_img, rgb_img, keypoints_ids, uv):
        downsample_idxs, x_pre, y_pre = self.set_downsample_filter(1)
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[downsample_idxs]/self.depth_scale
        zero_mask = z_values!=0


        # remove same keypoints
        keypoints_ids, uni_idx = np.unique(keypoints_ids, return_index=True)


        zero_mask_keypoints = zero_mask[keypoints_ids]
        # get interpolated z_values based on uv with bilinear interpolation
        uv = uv[uni_idx]
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()/self.depth_scale
        vgrid = torch.from_numpy(uv.reshape(1,1,-1,2)).float()
        vgrid[..., 0] = vgrid[..., 0] / ((self.W-1) * 2 - 1)
        vgrid[..., 1] = vgrid[..., 1] / ((self.H-1) * 2 - 1)
        z_values = torch.nn.functional.grid_sample(z_values.view(1,1,self.H,self.W), vgrid, mode='bilinear', align_corners=True).squeeze()
        z_values = z_values.flatten()


        colors = colors[keypoints_ids]
        x = x_pre[keypoints_ids] * z_values
        y = y_pre[keypoints_ids] * z_values
        points = torch.stack([x,y,z_values], dim=-1)


        points = points[zero_mask_keypoints]
        colors = colors[zero_mask_keypoints]
        z_values = z_values[zero_mask_keypoints]
        filter = torch.where(z_values<=self.depth_trunc)

        pts_idx = keypoints_ids[zero_mask_keypoints]
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), pts_idx
    
    
    def eliminate_overlapped2(self, distances, threshold):
        new_p_indices = np.where(distances>threshold)    # 5e-5
        
        return new_p_indices
        
    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)

        avg_trans_error = trans_error.mean()

        return avg_trans_error
    

    def extrapolate_poses(self, poses: np.ndarray) -> np.ndarray:
        """ Generates an interpolated pose based on the first two poses in the given array.
        Args:
            poses: An array of poses, where each pose is represented by a 4x4 transformation matrix.
        Returns:
            A 4x4 numpy ndarray representing the interpolated transformation matrix.
        """
        return poses[1, :] @ np.linalg.inv(poses[0, :]) @ poses[1, :]
    


    def compute_depthmse_forinitcam(self, init_w2c, pts, intrinsics, overlap_gtdepth, height, width):
        est_w2c = init_w2c
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]


        # Filter out the points that are invisible based on the depth
        curr_gt_depth = overlap_gtdepth.to(projected_pts.device).reshape(1, 1, height, width)
        vgrid = projected_pts.reshape(1, 1, -1, 2)
        # normalize to [-1, 1]
        vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
        vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
        depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
        depth_sample = depth_sample.reshape(-1)
        depth_mse = torch.mean((depth_sample - points_z[:, 0])**2)


        return depth_mse, points_z[:, 0].detach().cpu().numpy()
    