""" This module includes the Mapper class, which is responsible scene mapping: Paper Section 3.4  """
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial.transform import Rotation as R

from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from utils.visual_odometer import VisualOdometer
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, render_gaussian_model_mask, torch2np)
import glob
from src.utils.mapper_utils import create_point_cloud
import open3d as o3d

class Tracker(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Initializes the Tracker with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the tracking process.
        """
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.filter_alpha = self.config["filter_alpha"]
        self.filter_outlier_depth = self.config["filter_outlier_depth"]
        self.alpha_thre = self.config["alpha_thre"]
        self.soft_alpha = self.config["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["mask_invalid_depth"]
        self.w_color_loss = self.config["w_color_loss"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.frame_depth_loss = []
        self.frame_color_loss = []
        self.odometry_type = self.config["odometry_type"]
        self.help_camera_initialization = self.config["help_camera_initialization"]
        self.init_err_ratio = self.config["init_err_ratio"]
        self.enable_exposure = self.config["enable_exposure"]




    def compute_losses(self, gaussian_model: GaussianModel, render_settings: dict,
                       opt_cam_rot: torch.Tensor, opt_cam_trans: torch.Tensor,
                       gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                       exposure_ab=None, mask_ls=None, uv_tgt=None, iter=0, frame_id=0, total_iters=0) -> tuple:
        """ Computes the tracking losses with respect to ground truth color and depth.
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            opt_cam_rot: Optimizable tensor representing the camera's rotation.
            opt_cam_trans: Optimizable tensor representing the camera's translation.
            gt_color: Ground truth color image tensor.
            gt_depth: Ground truth depth image tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image.
        Returns:
            A tuple containing losses and renders
        """
        rel_transform = torch.eye(4).cuda().float()
        rel_transform[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None]))[0]
        rel_transform[:3, 3] = opt_cam_trans

        pts = gaussian_model.get_xyz()
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_transform @ pts4.T).T[:, :3]

        quat = F.normalize(opt_cam_rot[None])
        _rotations = multiply_quaternions(gaussian_model.get_rotation(), quat.unsqueeze(0)).squeeze(0)
        
     
        if mask_ls is not None:
            loss_keypoint = 0
            final_rendered_color = torch.zeros_like(gt_color)
            final_rendered_depth = torch.zeros_like(gt_depth).unsqueeze(0)
            for i in range(len(mask_ls)):
                color_tgt_gt = gt_color[:, uv_tgt[i, 1], uv_tgt[i, 0]]
      
                depth_tgt_gt = gt_depth[uv_tgt[i, 1], uv_tgt[i, 0]]
                mask_keypoint = mask_ls[i]
                render_dict_keypoint = render_gaussian_model_mask(gaussian_model, render_settings,
                                                override_means_3d=transformed_pts, override_rotations=_rotations, mask=mask_keypoint)
                rendered_color_keypoint, rendered_depth_keypoint = render_dict_keypoint["color"], render_dict_keypoint["depth"]

                rendered_color_tgt = rendered_color_keypoint[:, uv_tgt[i, 1], uv_tgt[i, 0]]
                rendered_depth_tgt = rendered_depth_keypoint[:, uv_tgt[i, 1], uv_tgt[i, 0]]

                loss_keypoint += l1_loss(rendered_depth_tgt, depth_tgt_gt, agg="none")
                final_rendered_color += rendered_color_keypoint
                final_rendered_depth += rendered_depth_keypoint
            loss_keypoint = loss_keypoint.sum()





        render_dict = render_gaussian_model(gaussian_model, render_settings,
                                            override_means_3d=transformed_pts, override_rotations=_rotations)
        rendered_color, rendered_depth = render_dict["color"], render_dict["depth"]
        if self.enable_exposure:
            rendered_color = torch.clamp(torch.exp(exposure_ab[0]) * rendered_color + exposure_ab[1], 0, 1.)
        alpha_mask = render_dict["alpha"] > self.alpha_thre

        tracking_mask = torch.ones_like(alpha_mask).bool()
        tracking_mask &= depth_mask
        depth_err = torch.abs(rendered_depth - gt_depth) * depth_mask

        if self.filter_alpha:
            tracking_mask &= alpha_mask
        if self.filter_outlier_depth and torch.median(depth_err) > 0:
            tracking_mask &= depth_err < 50 * torch.median(depth_err)

        color_loss = l1_loss(rendered_color, gt_color, agg="none")
        depth_loss = l1_loss(rendered_depth, gt_depth, agg="none") * tracking_mask

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            depth_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask

        mean_color_loss = color_loss.mean()
        mean_depth_loss = depth_loss.mean()

        color_loss = color_loss.sum()
        depth_loss = depth_loss.sum()

        if mask_ls is not None:
            depth_loss += 1000 * loss_keypoint

        

        # Visualize the tracking process
        tracking_mask = tracking_mask & alpha_mask
        if mask_ls is not None:
            return color_loss, depth_loss, rendered_color, rendered_depth, alpha_mask, mean_color_loss, mean_depth_loss, loss_keypoint
        else:
            return color_loss, depth_loss, rendered_color, rendered_depth, alpha_mask, mean_color_loss, mean_depth_loss

    
 



    def track(self, frame_id: int, gaussian_model: GaussianModel, prev_c2ws: np.ndarray) -> np.ndarray:
        """
        Updates the camera pose estimation for the current frame based on the provided image and depth, using either ground truth poses,
        constant speed assumption, or visual odometry.
        Args:
            frame_id: Index of the current frame being processed.
            gaussian_model: The current Gaussian model of the scene.
            prev_c2ws: Array containing the camera-to-world transformation matrices for the frames (0, i - 2, i - 1)
        Returns:
            The updated camera-to-world transformation matrix for the current frame.
        """

        # load 1st gaussian model
        use_1st_gaussian = False #True
        w_1st = 1
        if use_1st_gaussian:
            self.submap_paths = sorted(glob.glob(str(self.submap_path/"*.ckpt")), key=lambda x: int(x.split('/')[-1][:-5]))
            submap_dict = torch.load(self.submap_paths[0], map_location="cuda")
            gaussians = GaussianModel(sh_degree=0)
            self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
            gaussians.restore_from_params(submap_dict['gaussian_params'], self.opt)


        self.odometer = VisualOdometer(self.dataset.intrinsics, self.config["odometer_method"])


        _, image, depth, gt_c2w = self.dataset[frame_id]

        if (self.help_camera_initialization or self.odometry_type == "odometer") and self.odometer.last_rgbd is None:
            _, last_image, last_depth, _ = self.dataset[frame_id - 1]
            self.odometer.update_last_rgbd(last_image, last_depth)

        if self.odometry_type == "gt":
            return gt_c2w
        elif self.odometry_type == "const_speed":
            init_c2w = extrapolate_poses(prev_c2ws[1:])
        elif self.odometry_type == "odometer":
            odometer_rel = self.odometer.estimate_rel_pose(image, depth)
            init_c2w = prev_c2ws[-1] @ odometer_rel
        elif self.odometry_type == "previous":
            init_c2w = prev_c2ws[-1]

        
        last_c2w = prev_c2ws[-1]
        last_w2c = np.linalg.inv(last_c2w)
        init_rel = init_c2w @ np.linalg.inv(last_c2w)
        init_rel_w2c = np.linalg.inv(init_rel)
        reference_w2c = last_w2c
        render_settings = get_render_settings(
            self.dataset.width, self.dataset.height, self.dataset.intrinsics, reference_w2c)
        opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
        if self.enable_exposure:
            exposure_ab = torch.nn.Parameter(torch.tensor(
                0.0, device="cuda")), torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
        else:
            exposure_ab = None
        gaussian_model._xyz = gaussian_model.get_xyz().detach()
        gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)
        if use_1st_gaussian:
            gaussians.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)


        gt_color = self.transform(image).cuda()
        gt_depth = np2torch(depth, "cuda")
        depth_mask = gt_depth > 0.0
        gt_trans = np2torch(gt_c2w[:3, 3])
        gt_quat = np2torch(R.from_matrix(gt_c2w[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]])
        num_iters = self.config["iterations"]
        current_min_loss = float("inf")

        print(f"\nTracking frame {frame_id}")
        # Initial loss check
        color_loss, depth_loss, _, _, _, _, _ = self.compute_losses(gaussian_model, render_settings, opt_cam_rot,
                                                              opt_cam_trans, gt_color, gt_depth, depth_mask, 
                                                              exposure_ab)
        if use_1st_gaussian:
            color_loss_1st, depth_loss_1st, _, _, _, _, _ = self.compute_losses(gaussians, render_settings, opt_cam_rot,
                                                                                opt_cam_trans, gt_color, gt_depth, depth_mask,
                                                                                exposure_ab)
            color_loss = 1 * color_loss + w_1st * color_loss_1st
            depth_loss = 1 * depth_loss + w_1st * depth_loss_1st

        if len(self.frame_color_loss) > 0 and (
            color_loss.item() > self.init_err_ratio * np.median(self.frame_color_loss)
            or depth_loss.item() > self.init_err_ratio * np.median(self.frame_depth_loss)
        ):
            num_iters *= 2
            print(f"Higher initial loss, increasing num_iters to {num_iters}")
            if self.help_camera_initialization and self.odometry_type != "odometer":
                _, last_image, last_depth, _ = self.dataset[frame_id - 1]
                self.odometer.update_last_rgbd(last_image, last_depth)
                odometer_rel = self.odometer.estimate_rel_pose(image, depth)
                init_c2w = last_c2w @ odometer_rel
                init_rel = init_c2w @ np.linalg.inv(last_c2w)
                init_rel_w2c = np.linalg.inv(init_rel)
                opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
                gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)
                if use_1st_gaussian:
                    gaussians.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)

                render_settings = get_render_settings(
                    self.dataset.width, self.dataset.height, self.dataset.intrinsics, last_w2c)
                print(f"re-init with odometer for frame {frame_id}")

        color_loss_ls = []
        depth_loss_ls = []
        total_loss_ls = []

        for iter in range(num_iters):
            # for keypoint loss
            color_loss, depth_loss, _, _, _, mean_color_loss, mean_depth_loss = self.compute_losses(
                gaussian_model, render_settings, opt_cam_rot, opt_cam_trans, gt_color, gt_depth, depth_mask, exposure_ab, iter=iter, frame_id=frame_id, total_iters=num_iters)
            if use_1st_gaussian:
                color_loss_1st, depth_loss_1st, _, _, _, mean_color_loss_1st, mean_depth_loss_1st = self.compute_losses(
                    gaussians, render_settings, opt_cam_rot, opt_cam_trans, gt_color, gt_depth, depth_mask, exposure_ab, iter=iter, frame_id=frame_id, total_iters=num_iters)
                color_loss = 1 * color_loss + w_1st * color_loss_1st
                depth_loss = 1 * depth_loss + w_1st * depth_loss_1st
                mean_color_loss = 1 * mean_color_loss + w_1st * mean_color_loss_1st
                mean_depth_loss = 1 * mean_depth_loss + w_1st * mean_depth_loss_1st

            
            total_loss = (self.w_color_loss * color_loss + (1 - self.w_color_loss) * depth_loss)
            mean_total_loss = (self.w_color_loss * mean_color_loss + (1 - self.w_color_loss) * mean_depth_loss)


            color_loss_ls.append(mean_color_loss.item())
            depth_loss_ls.append(mean_depth_loss.item())
            total_loss_ls.append(mean_total_loss.item())

            
            total_loss.backward()
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if total_loss.item() < current_min_loss:
                    current_min_loss = total_loss.item()
                    best_w2c = torch.eye(4)
                    best_w2c[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None].clone().detach().cpu()))[0]
                    best_w2c[:3, 3] = opt_cam_trans.clone().detach().cpu()

                cur_quat, cur_trans = F.normalize(opt_cam_rot[None].clone().detach()), opt_cam_trans.clone().detach()
                cur_rel_w2c = torch.eye(4)
                cur_rel_w2c[:3, :3] = build_rotation(cur_quat)[0]
                cur_rel_w2c[:3, 3] = cur_trans
                if iter == num_iters - 1:
                    cur_w2c = torch.from_numpy(reference_w2c) @ best_w2c
                else:
                    cur_w2c = torch.from_numpy(reference_w2c) @ cur_rel_w2c
                cur_c2w = torch.inverse(cur_w2c)
                cur_cam = transformation_to_quaternion(cur_c2w)
                if (gt_quat * cur_cam[:4]).sum() < 0:  # for logging purpose
                    gt_quat *= -1

                if iter == num_iters - 1:
                    self.frame_color_loss.append(color_loss.item())
                    self.frame_depth_loss.append(depth_loss.item())



        final_c2w = torch.inverse(torch.from_numpy(reference_w2c) @ best_w2c)
        final_c2w[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=final_c2w.dtype, device=final_c2w.device)
        return torch2np(final_c2w), exposure_ab
