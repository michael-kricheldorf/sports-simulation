import position_estimator as rs
import numpy as np
import pose_estimator as rtm
import json
import pandas as pd

class Player:
    def __init__(self, tracker_id: int, team_value: int, image_crop: np.ndarray, xy_pos: np.ndarray, player_model, pose_estimator: rtm.RTMPose3D, position_estimator: rs.RoboflowSports):
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
        # check if any frames have been missed
        while (frame_no - 1 > len(self.xy_positions)):
            print(f"{frame_no - 1} > {len(self.xy_positions)}")
            self.xy_positions.append([-1, -1])
        self.xy_positions.append(xy_pos)
        # check if any frames have been missed
        while (frame_no - 1 > len(self.image_crops)):
            print(f"{frame_no - 1} > {len(self.image_crops)}")
            self.image_crops.append(None)
        print(f'{frame_no}: {xy_pos}')
        self.image_crops.append(image_crop)

    def generate_keypoints(self):
        # use image crops as inputs to rtmpose3d single image call and generate array for each frame containing keypoints
        for image_crop in self.image_crops:
            if (image_crop is not None):
                keypoint = self.pose_estimator.process_bbox(image_crop)
            else:
                # TODO: if keypoint is missing, interpolate between previous frame and next frame
                keypoint = None
            self.keypoints.append(keypoint)
        return
    
    def export(self, source_video_path):
        print(f'{self.tracker_id}')
        print(f'len(keypoints): {len(self.keypoints)}')
        print(f'len(keypoints[0]): {len(self.keypoints[0])}')
        
        kp_list = []

        for np_arr in self.keypoints:
            if (np_arr is not None):
                kp_list.append(np_arr.tolist())
            else:
                kp_list.append(None)

        with open(f'{source_video_path}_out/{self.tracker_id}.json', 'w') as json_out:
            json.dump(kp_list, json_out)

        #print(f'keypoints[0] info: {np.info(self.keypoints[0])}')
        #print(f'keypoints[0] head: {self.keypoints[0]}')
        # # Convert NumPy arrays to lists 
        # keypoints_as_lists = [keypoint_arr.tolist() for keypoint_arr in self.keypoints] 
        # # # Create a dictionary to hold the frame data 
        # data = { 
        #     "frames": keypoints_as_lists 
        #     } # Save the dictionary to a JSON file
        
        # with open('video_frames.json', 'w') as json_file: 
        #     json.dump(data, json_file, indent=4)


        # export fbx player animation as `[source_video_name]/[tracker_id].fbx`

        # save position as csv
        print(f'xy_positions[0]: {self.xy_positions[0]}')
        print(f'xy_positions[1]: {self.xy_positions[1]}')
        print(f'xy_positions[2]: {self.xy_positions[2]}')
        print(f'type(xy_positions): {type(self.xy_positions)}')
        df = pd.DataFrame(self.xy_positions)
        df.to_csv(f"{source_video_path}_out/{self.tracker_id}.csv", index=True)
        return