from __future__ import annotations
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

from rtmpose3d import *  # noqa: F401, F403

class RTMPose3D:
    def __init__(self,
                pose_estimate_config: str, 
                pose_estimate_checkpoint: str,
                device: str):
        self.pose_estimator = init_model(pose_estimate_config, pose_estimate_checkpoint, device=device)
        
    def process_bbox(self, bbox: np.ndarray):
        # TODO: handle multiple bounding boxes that may be caused by multiple players being detected within a single box, unlikely
        # for soccer, but a tackle in American football may need something like this
        pose_est_result = inference_topdown(self.pose_estimator, bbox)[0]

        #print(type(pose_est_result))
        #pose_est_result.track_id = pose_est_result[0].get('track_id', 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_result.pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = -keypoints[..., [0, 2, 1]]

        return keypoints

# from __future__ import annotations
# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import List

# import numpy as np
# from mmengine.logging import print_log

# from mmpose.apis import inference_topdown, init_model
# from mmpose.structures import (PoseDataSample, merge_data_samples,
#                                split_instances)

# from rtmpose3d import *  # noqa: F401, F403

# class RTMPose3D:
#     def __init__(self,
#                 pose_estimate_config: str, 
#                 pose_estimate_checkpoint: str,
#                 device: str):
#         self.pose_estimator = init_model(pose_estimate_config, pose_estimate_checkpoint, device=device)
        
#     def process_bbox(self, bbox: np.ndarray):
#         # TODO: handle multiple bounding boxes that may be caused by multiple players being detected within a single box, unlikely
#         # for soccer, but a tackle in American football may need something like this
#         pose_est_result = inference_topdown(self.pose_estimator, bbox)[0]

#         #print(type(pose_est_result))
#         #pose_est_result.track_id = pose_est_result[0].get('track_id', 1e4)

#         pred_instances = pose_est_result.pred_instances
#         keypoints = pred_instances.keypoints
#         keypoint_scores = pred_instances.keypoint_scores
#         if keypoint_scores.ndim == 3:
#             keypoint_scores = np.squeeze(keypoint_scores, axis=1)
#             pose_est_result.pred_instances.keypoint_scores = keypoint_scores
#         if keypoints.ndim == 4:
#             keypoints = np.squeeze(keypoints, axis=1)

#         keypoints = -keypoints[..., [0, 2, 1]]

#         pred_instances.keypoints = keypoints

#         pred_instances_list = split_instances(pred_instances_list)

#         return dict(meta_info=self.pose_estimator.dataset_meta, instance_info=pred_instances_list)