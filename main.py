# imports
import argparse
import json
import sys
import video
import os
# TODO: add a reprocess and overwrite command line option, i.e. if there already exists the output dir with all of the needed files, then don't bother reprocessing
#       unless the command line option is specified

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

    try:
        os.mkdir(f'{args_dict["source_video_path"]}_out/')
    except OSError:
        print("Error creating output directory for video. Either already exists or error.")
                # self, source_video: str, pose_estimate_config: str, pose_estimate_checkpoint: str, 
                #  ball_detection: str, player_detection: str, pitch_detection: str, device: str, 
                #  expected_charcters: int, player_model: str):
    video = video.Video(args_dict["source_video_path"], args_dict["pose_estimate_config"], 
                        args_dict["pose_estimate_checkpoint"], args_dict["ball_detection"], 
                        args_dict["player_detection"], args_dict["pitch_detection"],
                        args_dict["device"], args_dict["player_model"])
    


    video.generate_characters()    