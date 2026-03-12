""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import roma

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds 
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import * 

import glob
import cv2
from matplotlib import pyplot as plt
import time


class SLAMMapper(object):

    def __init__(self, slam, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])
        self.exposures_ab = torch.zeros(len(self.dataset), 2)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        if self.dataset_name == "replica":
            self.opt.scaling_lr = config["mapping"]["scaling_lr"]
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)
        self.enable_exposure = self.tracker.enable_exposure


        self.tracker.submap_path = self.output_path / "submaps"
        self.mapper.submap_path = self.output_path / "submaps"
        self.submap_path = self.output_path / "submaps"
        self.factor_opt_graph_path = self.output_path / "factor_opt_graph"
        os.makedirs(self.factor_opt_graph_path, exist_ok=True)

        self.mapper.output_path = self.output_path
        self.pts_vis_path = self.output_path / "pts_vis"
        os.makedirs(self.pts_vis_path, exist_ok=True)

        self.gt_depth_path = self.output_path / "gt_depth"
        os.makedirs(self.gt_depth_path, exist_ok=True)
        self.gt_rgb_path = self.output_path / "gt_rgb"
        os.makedirs(self.gt_rgb_path, exist_ok=True)

        self.gt_camera = config["tracking"]["gt_camera"]
        self.gicp_camera = config["tracking"]["gicp_camera"]
        
        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])

        self.final_pose = slam.final_pose
        self.num_poses = slam.num_poses
        self.delta_ls = slam.delta_ls
        self.use_pre_delta = slam.use_pre_delta

        

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                print(f"\nNew submap at {frame_id}")
                return True
        elif frame_id in self.new_submap_frame_ids:
            return True
        return False

    def save_current_submap(self, gaussian_model: GaussianModel):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
        

    
    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        # self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        return gaussian_model
    
    
    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0


        for frame_id in range(len(self.dataset)):
            

            if self.gt_camera:
                print(f"Using GT camera: {self.gt_camera}")
                estimated_c2w = self.dataset[frame_id][-1]
                
            elif self.gicp_camera and (self.dataset_name != "scannetpp" or self.use_pre_delta == True):
                print(f"Using GICP camera: {self.gicp_camera}")
                while frame_id >= self.num_poses.item():
                    time.sleep(1e-15)
                estimated_c2w = self.final_pose[frame_id].cpu().numpy()

            else:
                if frame_id in [0, 1]:
                    estimated_c2w = self.dataset[frame_id][-1]
                   
                else:
                    estimated_c2w, exposure_ab = self.tracker.track(
                        frame_id, gaussian_model,
                        torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
            exposure_ab = exposure_ab if self.enable_exposure else None
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            if frame_id > 0 and self.dataset_name == "scannetpp" and not self.use_pre_delta:
                delta = np.linalg.inv(torch2np(self.estimated_c2ws[frame_id-1])).dot(torch2np(self.estimated_c2ws[frame_id]))
                self.delta_ls[frame_id-1] = torch.tensor(delta)

            # Reinitialize gaussian model for new segment
            if self.should_start_new_submap(frame_id) or frame_id == len(self.dataset) - 1:
                # first save current submap and its keyframe info
                self.save_current_submap(gaussian_model)
                

                gaussian_model = self.start_new_submap(frame_id, gaussian_model)




            if frame_id in self.mapping_frame_ids:
                print("\nMapping frame", frame_id)
                gaussian_model.training_setup(self.opt, exposure_ab) 
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                new_submap = not bool(self.keyframes_info)
                print(f"Frame {frame_id} new submap: {new_submap}")

                

                mapping_one_frame_gs = True #False
                if frame_id > 4 and not mapping_one_frame_gs:
                    opt_dict, curr_gs_num = self.mapper.map(
                        frame_id, estimate_c2w, gaussian_model, new_submap, exposure_ab)
             
                    # prune_mask = torch.tensor([False] * curr_gs_num + [True] * (gaussian_model._xyz.shape[0] - curr_gs_num))
                    # prunr_opt_mask = torch.tensor([False] * curr_gs_num)
                    gaussian_model.prune_points_5()
                 
                else:
                    opt_dict = self.mapper.map(
                        frame_id, estimate_c2w, gaussian_model, new_submap, exposure_ab)
     
     
               
                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": frame_id, 
                    "opt_dict": opt_dict,
                }
                if self.enable_exposure:
                    self.keyframes_info[frame_id]["exposure_a"] = exposure_ab[0].item()
                    self.keyframes_info[frame_id]["exposure_b"] = exposure_ab[1].item()
                
            if frame_id == len(self.dataset) - 1:
                self.save_current_submap(gaussian_model)
            
           
            if self.enable_exposure:
                self.exposures_ab[frame_id] = torch.tensor([exposure_ab[0].item(), exposure_ab[1].item()])

        # save delta ls
        np.save(self.output_path / "delta_ls.npy", np.array(self.delta_ls))
        print(len(self.delta_ls))


        if self.dataset_name == "scannetpp":
            save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w_mapper.ckpt", directory=self.output_path)
            self.estimated_c2ws[:frame_id + 1] = self.final_pose[:frame_id + 1].cpu()

        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
        if self.enable_exposure:
            save_dict_to_ckpt(self.exposures_ab, "exposures_ab.ckpt", directory=self.output_path)

