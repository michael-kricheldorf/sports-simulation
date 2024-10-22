# imports
import argparse
import json
import sys
import position_estimator as rs
import numpy as np
import pose_estimator as rtm

class Video:
    def __init__(self, source_video_path: str, rtmpose3d_detection_config_path: str, rtmpose3d_detection_checkpoint_path: str,
        rtmpose3d_estimator_config_path: str, rtmpose3d_estimator_checkpoint_path: str, roboflow_sports_ball_detection_path: str,
        roboflow_sports_player_detection_path: str, roboflow_sports_pitch_detection_path: str, device: str, expected_charcters: int = 24):

        self.source_video_path = source_video_path
        self.expected_characters = expected_charcters
        self.characters = [None] * self.expected_charcters

        self.position_estimator = rs.RoboflowSports(roboflow_sports_ball_detection_path, roboflow_sports_player_detection_path, 
                                                    roboflow_sports_pitch_detection_path, device)
        self.pose_estimator = rtm.RTMPose3D(rtmpose3d_detection_config_path, rtmpose3d_detection_checkpoint_path, 
                                            rtmpose3d_estimator_config_path, rtmpose3d_estimator_checkpoint_path)

    def generate_characters(self):
        # take source video and then update the characters attribute with a list of player objects
        frame_generator = self.position_estimator.run_radar(self.source_video_path)

        # run frame generator
        for frame_index, (frame_data, frame_crops) in enumerate(frame_generator):
            if (len(frame_data) == len(frame_crops)):               # make sure that the data and crop lengths match, if not then ignore the data
                for character_index, character_data in enumerate(frame_data):                   # loop through each character's data within the frame 
                    tracker_id = character_data[0]      
                    if (tracker_id > self.expected_characters):     # ignore any tracker_id greater than the expected character count
                        if (self.characters[tracker_id] == None):   # check if the object exists at the tracker_id
                            self.characters[tracker_id] = Player(tracker_id=tracker_id, team_value=character_data[1], 
                                                                 image_crop=frame_crops[character_index], 
                                                                 xy_pos=character_data[2], player_model=None)
                        else:
                            self.characters[tracker_id].append_frame_data(frame_index, frame_crops[character_index], character_data[2])


class Player:
    def __init__(self, tracker_id: int, team_value: int, image_crop: np.ndarray, xy_pos: np.ndarray, player_model):
        self.tracker_id = tracker_id
        self.team_value = team_value
        
        self.image_crops = []
        self.image_crops.append(image_crop)
        self.xy_positions = []
        self.xy_positions.append(xy_pos)
        self.keypoints = []

        self.player_model = player_model

    def append_frame_data(self, frame_no, image_crop, xy_pos):
        # check if any frames have been missed
        while (frame_no > len(self.xy_positions) - 1):
            self.xy_positions.append(None)

        # check if any frames have been missed
        while (frame_no > len(self.image_crops) - 1):
            self.image_crops.append(None)

        self.xy_positions.append(xy_pos)
        self.xy_positions.append(image_crop)

    def generate_keypoints(self):
        # use image crops as inputs to rtmpose3d single image call and generate array for each frame containing keypoints
        return
    
    def save_fbx(self): 
        # export fbx player animation as `[source_video_name]/[tracker_id].fbx`
        return

    def save_xy_as_csv(self):
        # export xy data as a csv file `[source_video_name]/[tracker_id].csv`
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_json_path', type=str, required=True)

    args_dict = {}

    try:
        args = parser.parse_args()
    except Exception:
        print("Failed to parse command line argument.")
        sys.exit(1)
    
    try:
        with open(args.config_json_path, 'r') as file:
            args_dict = json.load(file)
    except:
        print("Error opening JSON file: ", args.config_json_path)
        sys.exit(1)

    main(
        args_dict["source_video_path"], 
        args_dict["rtmpose3d_detection_config_path"],
        args_dict["rtmpose3d_detection_checkpoint_path"],
        args_dict["rtmpose3d_estimator_config_path"],
        args_dict["rtmpose3d_estimator_checkpoint_path"],
        args_dict["roboflow_sports_ball_detection_path"],
        args_dict["roboflow_sports_player_detection_path"],
        args_dict["roboflow_sports_pitch_detection_path"],
        args_dict["device"]
    )