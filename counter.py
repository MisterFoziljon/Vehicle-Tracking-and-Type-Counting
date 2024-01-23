import sys
import os

HOME = os.getcwd()
sys.path.append(f"{HOME}/ByteTrack")

from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.tools.detections import Detections, BoxAnnotator
from yolox.tracker.byte_tracker import BYTETracker, STrack
from supervision.geometry.dataclasses import Point
from onemetric.cv.utils.iou import box_iou_batch
from supervision.draw.color import ColorPalette
from dataclasses import dataclass
from ultralytics import YOLO
from typing import List
import numpy as np
import cv2

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

class TACV:
    def __init__(self, rtsp):
        self.rtsp = rtsp
        self.model = YOLO("yolov8x.pt")
        self.model.fuse()
        self.CLASS_NAMES_DICT = self.model.model.names

        self.CLASS_ID = [1, 2, 3, 5, 7]

        self.LINE_START = Point(50, 1500)
        self.LINE_END = Point(3790, 1500)
        
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.line_counter = LineCounter(start=self.LINE_START, end=self.LINE_END)
        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    def detections2boxes(self, detections: Detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]))
    
    def tracks2boxes(self, tracks: List[STrack]) -> np.ndarray:
        return np.array([
            track.tlbr
            for track
            in tracks
        ], dtype=float)
    
    def match_detections_with_tracks(self, detections: Detections, tracks: List[STrack]) -> Detections:
        if not np.any(detections.xyxy) or len(tracks) == 0:
            return np.empty((0,))
    
        tracks_boxes = self.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detections.xyxy)
        track2detection = np.argmax(iou, axis=1)
    
        tracker_ids = [None] * len(detections)
    
        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id
    
        return tracker_ids

    def run(self):
        
        video = cv2.VideoCapture(self.rtsp)
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (960,540))
        
        while True:
            ret, frame = video.read() 
            if ret == True:
                results = self.model(
                    source = frame,
                    conf = 0.5,
                    classes = self.CLASS_ID)
                
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int))
            
                mask = np.array([class_id in self.CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                
                tracks = self.byte_tracker.update(
                        output_results=self.detections2boxes(detections=detections),
                        img_info=frame.shape,
                        img_size=frame.shape)
                
                tracker_id = self.match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                
                labels = [
                    f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections]
                
                counts = self.line_counter.update(detections=detections)
    
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                self.line_annotator.annotate(frame=frame, line_counter=self.line_counter, counts=counts)
                
                frame = cv2.resize(frame,(960, 540))
                cv2.imshow('frame',frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
        
        video.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SOURCE_VIDEO_PATH = f"{HOME}/vehicle-counting.mp4"
    track = TACV(SOURCE_VIDEO_PATH)
    track.run()