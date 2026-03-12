""" This module includes the Mapper class, which is responsible scene mapping: Paragraph 3.2  """
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from src.entities.arguments import OptimizationParams
from src.entities.datasets import TUM_RGBD, BaseDataset, ScanNet
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.mapper_utils import (calc_psnr, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask, get_init_gs_scales, create_point_cloud_torch,
                                    sample_pixels_based_on_gradient)
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.vis_utils import *  
import glob
from torch import nn
import cv2
import open3d as o3d
import copy
import os
from matplotlib import pyplot as plt



class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_thre = config["alpha_thre"]
        self.pruning_thre = config["pruning_thre"]
        self.current_view_opt_iterations = config["current_view_opt_iterations"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = []
        self.total_opt_time = 0

    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        Args:
            gaussian_model: The current submap
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap:
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask

    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)

        """

        pts = create_point_cloud(gt_color, 1 * gt_depth, intrinsics, estimate_c2w)
        sq_scale_gaussian = get_init_gs_scales(gt_color, 1 * gt_depth, intrinsics, estimate_c2w)
        # sq_scale_gaussian = get_init_gs_scales(gt_color, 1.005 * gt_depth, intrinsics, estimate_c2w)
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        valid_ids = np.flatnonzero(seeding_mask)
        if is_new_submap:
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pts.shape[0])
            else:
                uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            combined_ids = np.concatenate((combined_ids, valid_ids))
            sample_ids = np.unique(combined_ids)
        else:
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        return pts[sample_ids, :].astype(np.float32), sq_scale_gaussian[sample_ids].astype(np.float32)


    
    



    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100, estimate_c2w=None) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        iteration = 0
        losses_dict = {}
        losses_dict["color_loss_ls"] = []
        losses_dict["depth_loss_ls"] = []
        losses_dict["total_loss_ls"] = []
        psnr_list = []  

       

        current_frame_iters = self.current_view_opt_iterations * iterations
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()

        # add delta depth
        fused_point_cloud = gaussian_model.get_xyz().clone().detach()
        factor_opt = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        factor_opt = torch.nn.Parameter(factor_opt.cuda().float().contiguous().requires_grad_(True))
        cam_center = estimate_c2w[:3, 3]
        gaussian_model.optimizer.add_param_group({"params": factor_opt, "lr": 0.01}) 
       
        while iteration < iterations + 1:
           
            gaussian_model._xyz = factor_opt * fused_point_cloud - (factor_opt - 1) * torch.tensor(cam_center).float().cuda()

            gaussian_model.optimizer.zero_grad(set_to_none=True)
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)


            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])


            

            image, depth = render_pkg["color"], render_pkg["depth"]
            if keyframe["exposure_ab"] is not None:
                image = torch.clamp(image * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]

           

            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            if keyframe_id > 0 and len(keyframes) > 1:
                alpha_mask = render_pkg["alpha"] > 0.98
                alpha_mask = alpha_mask.squeeze(0)
                mask = mask & alpha_mask

 
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image, gt_image) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            if keyframe_id > 0 and len(keyframes) > 1:
                color_loss = l1_loss(image, gt_image)

            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            reg_loss = 0.0

            if len(keyframes) > 1:
                if keyframe_id == 0:
                    total_loss = color_loss + reg_loss + depth_loss
                else:
                    total_loss = color_loss + reg_loss + depth_loss 

            else:
                total_loss = color_loss + depth_loss + reg_loss
                

            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}
            losses_dict["color_loss_ls"].append(color_loss.item())
            losses_dict["depth_loss_ls"].append(depth_loss.item())
            losses_dict["total_loss_ls"].append(total_loss.item())

            with torch.no_grad():
                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)

            iteration += 1

           
            
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        losses_dict["factor_opt"] = factor_opt
        return losses_dict
    

    def optimize_submap_5(self, keyframes: list, gaussian_model: GaussianModel, gaussians1, gaussians2, gaussians3, gaussians4, iterations: int = 100, estimate_c2w=None) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        iteration = 0
        losses_dict = {}
        losses_dict["color_loss_ls"] = []
        losses_dict["depth_loss_ls"] = []
        losses_dict["total_loss_ls"] = []

       

        current_frame_iters = self.current_view_opt_iterations * iterations
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()

        # add delta depth
        fused_point_cloud = gaussian_model.get_xyz().clone().detach()
        factor_opt = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        factor_opt = torch.nn.Parameter(factor_opt.cuda().float().contiguous().requires_grad_(True))
        cam_center = estimate_c2w[:3, 3]
        # combine the gaussians with gaussians_model
        curr_gs_num = gaussian_model._xyz.shape[0]
        gaussian_model._rotation = nn.Parameter(torch.cat([gaussian_model._rotation, gaussians1._rotation, gaussians2._rotation, gaussians3._rotation, gaussians4._rotation], dim=0)).requires_grad_(True)
        gaussian_model._features_dc = nn.Parameter(torch.cat([gaussian_model._features_dc, gaussians1._features_dc, gaussians2._features_dc, gaussians3._features_dc, gaussians4._features_dc], dim=0)).requires_grad_(True)
        gaussian_model._features_rest = nn.Parameter(torch.cat([gaussian_model._features_rest, gaussians1._features_rest, gaussians2._features_rest, gaussians3._features_rest, gaussians4._features_rest], dim=0)).requires_grad_(True)
        gaussian_model._opacity = nn.Parameter(torch.cat([gaussian_model._opacity, gaussians1._opacity, gaussians2._opacity, gaussians3._opacity, gaussians4._opacity], dim=0)).requires_grad_(True)
        gaussian_model._scaling = nn.Parameter(torch.cat([gaussian_model._scaling, gaussians1._scaling, gaussians2._scaling, gaussians3._scaling, gaussians4._scaling], dim=0)).requires_grad_(True)

        gaussian_model.optimizer.add_param_group({"params": factor_opt, "lr": 0.01, "name": "factor_opt"}) 
        
        
        while iteration < iterations + 1:

        
            gaussian_model._xyz = factor_opt * fused_point_cloud - (factor_opt - 1) * torch.tensor(cam_center).float().cuda()
            gaussian_model._xyz = torch.cat([gaussian_model._xyz, gaussians1._xyz, gaussians2._xyz, gaussians3._xyz, gaussians4._xyz], dim=0)

            gaussian_model.optimizer.zero_grad(set_to_none=True)
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)

            
            

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])

           
            image, depth = render_pkg["color"], render_pkg["depth"]
            if keyframe["exposure_ab"] is not None:
                image = torch.clamp(image * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]

            
            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            if keyframe_id > 0 and len(keyframes) > 1:
                alpha_mask = render_pkg["alpha"] > 0.98
                alpha_mask = alpha_mask.squeeze(0)
                mask = mask & alpha_mask

            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            if keyframe_id > 0 and len(keyframes) > 1:
                color_loss = l1_loss(image[:, mask], gt_image[:, mask])

            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            reg_loss = isotropic_loss(gaussian_model.get_scaling())
            if len(keyframes) > 1:
                if keyframe_id == 0:
                    total_loss = color_loss + reg_loss + depth_loss 
                else:
                    total_loss = color_loss + reg_loss + depth_loss  
            else:
                total_loss = color_loss + depth_loss + reg_loss
                

            total_loss.backward()


            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}
            losses_dict["color_loss_ls"].append(color_loss.item())
            losses_dict["depth_loss_ls"].append(depth_loss.item())
            losses_dict["total_loss_ls"].append(total_loss.item())

            with torch.no_grad():
                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)
 


            iteration += 1


        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict, curr_gs_num

    def grow_submap(self, gt_depth: np.ndarray, estimate_c2w: np.ndarray, gaussian_model: GaussianModel,
                    pts: np.ndarray, filter_cloud: bool, scales: np.ndarray) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        Args:
            gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
            estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
            gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
            pts: The current set of 3D points in the keyframe of shape (N, 3)
            filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
                outliers or noise before integrating it into the map.
        Returns:
            int: The number of points added to the submap
        """
        gaussian_points = gaussian_model.get_xyz()
        camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics)
        reused_pts_ids = compute_frustum_point_ids(
            gaussian_points, np2torch(camera_frustum_corners), device="cuda")
        new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
                                             radius=self.new_points_radius, device="cuda")
        new_pts_ids = torch2np(new_pts_ids)
        if new_pts_ids.shape[0] > 0:
            cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)
            if filter_cloud:
                cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            gaussian_model.add_points(cloud_to_add, scales[new_pts_ids], estimate_c2w)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        print("Gaussian model size", gaussian_model.get_size())
        return new_pts_ids.shape[0]

    def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool, exposure_ab=None) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """

        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        color_transform = torchvision.transforms.ToTensor()
        keyframe = {
            "color": color_transform(gt_color).cuda(),
            "depth": np2torch(gt_depth, device="cuda"),
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c),
            "exposure_ab": exposure_ab}

        seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)
        #### complete depth==0 pixels in gt_depth with the depth of the nearest pixel
        gt_depth = cv2.inpaint(gt_depth, (gt_depth == 0).astype(np.uint8), 3, cv2.INPAINT_NS)
        pts, scales = self.seed_new_gaussians(
            gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)

        filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap

        new_pts_num = self.grow_submap(gt_depth, estimate_c2w, gaussian_model, pts, filter_cloud, scales)




        mapping_one_frame_gs = True #False
        if frame_id > 4 and not mapping_one_frame_gs:
            self.submap_paths = sorted(glob.glob(str(self.submap_path/"*.ckpt")), key=lambda x: int(x.split('/')[-1][:-5]))
            submap_dict1 = torch.load(self.submap_paths[frame_id-1], map_location="cuda")
            gaussians1 = GaussianModel(sh_degree=0)
            self.fix_opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
            gaussians1.restore_from_params(submap_dict1['gaussian_params'], self.fix_opt)
            gaussians1.fix_setup(exposure_ab)

            submap_dict2 = torch.load(self.submap_paths[frame_id-2], map_location="cuda")
            gaussians2 = GaussianModel(sh_degree=0)
            gaussians2.restore_from_params(submap_dict2['gaussian_params'], self.fix_opt)
            gaussians2.fix_setup(exposure_ab)

            submap_dict3 = torch.load(self.submap_paths[frame_id-3], map_location="cuda")
            gaussians3 = GaussianModel(sh_degree=0)
            gaussians3.restore_from_params(submap_dict3['gaussian_params'], self.fix_opt)
            gaussians3.fix_setup(exposure_ab)

            submap_dict4 = torch.load(self.submap_paths[frame_id-4], map_location="cuda")
            gaussians4 = GaussianModel(sh_degree=0)
            gaussians4.restore_from_params(submap_dict4['gaussian_params'], self.fix_opt)
            gaussians4.fix_setup(exposure_ab)


            



        max_iterations = self.iterations
        if is_new_submap:
            max_iterations = self.new_submap_iterations
        start_time = time.time()
       
        if frame_id > 4 and not mapping_one_frame_gs:
            opt_dict, curr_gs_num = self.optimize_submap_5([(frame_id, keyframe)] + self.keyframes, gaussian_model, gaussians1, gaussians2, gaussians3, gaussians4, max_iterations, estimate_c2w)
        else:
            opt_dict = self.optimize_submap([(frame_id, keyframe)], gaussian_model, max_iterations, estimate_c2w) # only current frame
            # opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations, estimate_c2w) # current frame and previous frame
       
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)
        self.total_opt_time += opt_dict["optimization_time"]

        self.keyframes.append((frame_id, keyframe))

        use_0th_frame = False
        use_5frames = False #True
        if use_0th_frame:
            # use 0th k-1th and kth frame during kth frame mapping
            if frame_id > 0 and len(self.keyframes) > 2:
                self.keyframes.pop(1)
        elif use_5frames:
            # use 5 frames during kth frame mapping
            if frame_id > 4:
                self.keyframes.pop(0)
        else:
            # use k-1th and kth frame during kth frame mapping
            if frame_id > 0:
                self.keyframes.pop(0)




        # Visualise the mapping for the current frame
        with torch.no_grad():
            render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
            if keyframe["exposure_ab"] is not None:
                image_vis = torch.clamp(image_vis * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
            mask = (gt_depth > 0).squeeze()

            psnr_value = calc_psnr(image_vis[:, mask], keyframe["color"][:, mask]).item()
            opt_dict["psnr_render"] = psnr_value
            print(f"PSNR this frame: {psnr_value}")
          

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                          optimization_time/max_iterations, opt_dict)
        if frame_id > 4 and not mapping_one_frame_gs:
            return opt_dict, curr_gs_num
        else:
            return opt_dict

