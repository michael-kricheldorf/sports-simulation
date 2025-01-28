import position_estimator as rs
import numpy as np
import pose_estimator as rtm
import player

class Video:
    def __init__(self, source_video: str, pose_estimate_config: str, pose_estimate_checkpoint: str, 
                 ball_detection: str, player_detection: str, pitch_detection: str, device: str, player_model: str):

        self.source_video = source_video
        self.characters = []
        self.player_model = player_model

        self.position_estimator = rs.RoboflowSports(ball_detection, player_detection, pitch_detection, device)
        self.pose_estimator = rtm.RTMPose3D(pose_estimate_config, pose_estimate_checkpoint, device)

    def generate_characters(self):
        # take source video and then update the characters attribute with a list of player objects
        frame_generator = self.position_estimator.run_radar(self.source_video)

        # run frame generator
        for frame_index, (frame_data, frame_crops) in enumerate(frame_generator):
            if (len(frame_data) == len(frame_crops)):               # make sure that the data and crop lengths match, if not then ignore the data
                for character_index, character_data in enumerate(frame_data):                   # loop through each character's data within the frame 
                    tracker_id = character_data[0].astype(int) - 1
                    # print(f'{tracker_id}', end=", ")
                    # need to check if the index at tracker_id exists and if it does not
                    # then extend the list up to that and then append the player
                    missing = len(self.characters) - (tracker_id + 1)
                    if (missing <= 0):
                        self.characters.extend([None] * (-1*missing))
                    #print([character.tracker_id for character in self.characters if character is not None])
                    if (self.characters[tracker_id] is None):   # check if the object exists at the tracker_id
                        self.characters[tracker_id] = player.Player(tracker_id=tracker_id, team_value=character_data[1], 
                                                                image_crop=frame_crops[character_index], 
                                                                xy_pos=[character_data[2], character_data[3]], 
                                                                player_model=self.player_model, 
                                                                pose_estimator=self.pose_estimator, 
                                                                position_estimator=self.position_estimator)
                    else:                                       # if the player already exists then just add to the player's existing data
                        self.characters[tracker_id].append_frame_data(frame_index, frame_crops[character_index], [character_data[2], character_data[3]])
                # print("")
            # else:
            #     print(f'{len(frame_data)} / {len(frame_crops)}')

        # generate keypoints for each character
        print(len(self.characters))
        for character in self.characters:
            # print(character, end=', ')
            if character != None:
                character.generate_keypoints()
                character.export(self.source_video)
            
        return