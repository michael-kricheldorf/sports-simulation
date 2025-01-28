from typing import Iterator, List, Tuple

#import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

class RoboflowSports:
    def __init__(self, ball_detection_path: str, player_detection_path: str, pitch_detection_path: str, device: str):
        # replace the paths below with YOLO model inits
        self.ball_detection_path = ball_detection_path
        self.player_detection_path = player_detection_path
        self.pitch_detection_path = pitch_detection_path
        self.source_video_path = None
        self.device = device

        self.player_detection_model = YOLO(self.player_detection_path).to(device=self.device)
        self.pitch_detection_model = YOLO(self.pitch_detection_path).to(device=self.device)

    def run_radar(self, source_video_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self.source_video_path = source_video_path

        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path, stride=STRIDE)

        crops = []
        for frame in tqdm(frame_generator, desc='collecting crops'):
            result = self.player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            crops += self.get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

        team_classifier = TeamClassifier(device=self.device)
        team_classifier.fit(crops)

        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        for frame in frame_generator:
            result = self.pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            result = self.player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = self.get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)

            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            goalkeepers_team_id = self.resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers)
            
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            detections = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            
            trans_xy = self.radar_transform(detections, keypoints)
            frame_data = np.column_stack((detections.tracker_id, color_lookup, trans_xy))
            frame_crops = self.get_crops(frame, detections)

            yield (frame_data, frame_crops)

    def get_crops(self, frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
        return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

    def radar_transform(
        self,
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
    ) -> np.ndarray:
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32)
        )
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        return transformer.transform_points(points=xy)
    
    def resolve_goalkeepers_team_id(
        self,
        players: sv.Detections,
        players_team_id: np.array,
        goalkeepers: sv.Detections
    ) -> np.ndarray:
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        return np.array(goalkeepers_team_id)
