import position_estimator as rs
import numpy as np
from mmengine.logging import print_log
from mmpose.apis import init_model
import player
from rtmpose3d import *

# The Video class is responsible for taking an input video and converting it into a list of Player objects
# which store frame-by-frame data of the Player's bounding boxes and xy positions. These bounding boxes are then 
# used when estimating the 3D pose. 

class Video:
    def __init__(self, source_video: str, pose_estimate_config: str, pose_estimate_checkpoint: str, 
                 ball_detection: str, player_detection: str, pitch_detection: str, device: str, player_model: str):

        self.source_video = source_video    # the path to the video
        self.players = []                   # the array of Player objects
        self.player_model = player_model    # the path to the player model

        # this is the position estimator we're using for now
        self.position_estimator = rs.RoboflowSports(ball_detection, player_detection, pitch_detection, device)
        # initialize the mmpose model 
        self.pose_estimator = init_model(pose_estimate_config, pose_estimate_checkpoint, device)
        
    def generate_players(self):
        # "run_radar" is a generator function that is repeatedly called on each frame
        frame_generator = self.position_estimator.run_radar(self.source_video)

        # call the generator function on each frame
        for frame_index, (frame_data, frame_crops) in enumerate(frame_generator):
            # make sure that the data and crop lengths match, if not then ignore the data
            if (len(frame_data) == len(frame_crops)):
                # for a given frame, loop through each player's data
                for character_index, character_data in enumerate(frame_data): 
                    tracker_id = character_data[0].astype(int) - 1
                    # need to check if the index at tracker_id exists and if it does not
                    # then extend the list up to that and then append the player
                    missing = len(self.players) - (tracker_id + 1)
                    if (missing <= 0):
                        self.players.extend([None] * (-1*missing))
                    if (self.players[tracker_id] is None):   # check if the object exists at the tracker_id
                        self.players[tracker_id] = player.Player(tracker_id=tracker_id, team_value=character_data[1], 
                                                                image_crop=frame_crops[character_index], 
                                                                xy_pos=[character_data[2], character_data[3]], 
                                                                player_model=self.player_model, 
                                                                pose_estimator=self.pose_estimator, 
                                                                position_estimator=self.position_estimator)
                    else:                                       # if the player already exists then just add to the player's existing data
                        self.players[tracker_id].append_frame_data(frame_index, frame_crops[character_index], [character_data[2], character_data[3]])


        # generate keypoints for each character
        for character in self.players:
            if character != None:
                character.generate_keypoints()
                character.export(self.source_video)
            
        return