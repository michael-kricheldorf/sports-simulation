import position_estimator as rs
import json
import pandas as pd
import numpy as np
from mmengine.logging import print_log
from mmpose.apis import inference_topdown
from rtmpose3d import *

class Player:
    def __init__(self, tracker_id: int, team_value: int, image_crop: np.ndarray, 
                xy_pos: np.ndarray, player_model, pose_estimator, 
                position_estimator: rs.RoboflowSports):
        self.tracker_id = tracker_id.astype(int)
        self.team_value = team_value
        
        self.image_crops = []
        self.image_crops.append(image_crop)
        self.xy_positions = []
        self.xy_positions.append(xy_pos)
        self.keypoints = []

        self.player_model = player_model

        self.pose_estimator = pose_estimator
        self.position_estimator = position_estimator

    def append_frame_data(self, frame_no, image_crop, xy_pos):
        # check if any frames have been missed and fill in the XY values with -1, 1
        while (frame_no - 1 > len(self.xy_positions)):
            self.xy_positions.append([-1, -1])
        self.xy_positions.append(xy_pos)

        # check if any frames have been missed and fill in missing frames with None values
        while (frame_no - 1 > len(self.image_crops)):
            self.image_crops.append(None)
        self.image_crops.append(image_crop)

    def generate_keypoints(self):
        # use image crops as inputs to rtmpose3d single image call and generate array for each frame containing keypoints
        for image_crop in self.image_crops:
            if (image_crop is not None):
                pose_est_result = inference_topdown(self.pose_estimator, image_crop)[0]

                pred_instances = pose_est_result.pred_instances
                keypoints = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores
                if keypoint_scores.ndim == 3:
                    keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                    pose_est_result.pred_instances.keypoint_scores = keypoint_scores
                if keypoints.ndim == 4:
                    keypoints = np.squeeze(keypoints, axis=1)

                keypoint = -keypoints[..., [0, 2, 1]]
            else:
                # TODO: if keypoint is missing, interpolate between previous frame and next frame
                keypoint = None
            self.keypoints.append(keypoint)
        return
    
    def export(self, source_video_path):
        kp_list = []

        for np_arr in self.keypoints:
            if (np_arr is not None):
                kp_list.append(np_arr.tolist())
            else:
                kp_list.append(None)

        with open(f'{source_video_path}_out/{self.tracker_id}.json', 'w') as json_out:
            json.dump(kp_list, json_out)

        # save position as csv
        print(f'xy_positions[0]: {self.xy_positions[0]}')
        print(f'xy_positions[1]: {self.xy_positions[1]}')
        print(f'xy_positions[2]: {self.xy_positions[2]}')
        print(f'type(xy_positions): {type(self.xy_positions)}')
        df = pd.DataFrame(self.xy_positions)
        df.to_csv(f"{source_video_path}_out/{self.tracker_id}.csv", index=True)
        return