from typing import Dict

import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections


class LineCounter:
    def __init__(self, start: Point, end: Point):
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        
        self.bicycle: int = 0
        self.car: int = 0
        self.motorcycle: int = 0
        self.bus: int = 0
        self.truck: int = 0
    
    def update(self, detections: Detections):
        for xyxy, confidence, class_id, tracker_id in detections:
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
         
            if not tracker_state:
                if class_id == 1:
                    self.bicycle += 1
                    
                if class_id == 2:
                    self.car += 1
                    
                if class_id == 3:
                    self.motorcycle += 1
                    
                if class_id == 5:
                    self.bus += 1
                    
                if class_id == 7:
                    self.truck += 1
                    
                self.in_count += 1
        return [self.bicycle, self.car, self.motorcycle, self.bus, self.truck]

class LineCounterAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.white(),
        text_scale: float = 0.5,
    ):
        
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale

    def annotate(self, frame: np.ndarray, line_counter: LineCounter, counts) -> np.ndarray:
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
  
        cv2.putText(
            frame,
            f"bicycle:{counts[0]}",
            (70, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"car:{counts[1]}",
            (70, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"motorcycle:{counts[2]}",
            (70, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"bus:{counts[3]}",
            (70, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"truck:{counts[4]}",
            (70, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )