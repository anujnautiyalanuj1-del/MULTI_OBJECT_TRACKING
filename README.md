
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

from ultralytics import 
from ultralytics.utils.plotting import colors


class ObjectTracking:

    def __init__(
        self,
        model="mpt26n.pt",
        source="videos/xyz.mp4"
    ):
        # Load  model
        self.model = (model)
        self.names = self.model.names

        # Check video file
        if not Path(source).is_file():
            raise FileNotFoundError(
                f"Video not found: {source}"
            )

        # Open video
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

        # Get video details
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30.0

        # Create output video
        self.writer = cv2.VideoWriter(
            "object-tracking(2).avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            self.cap.release()
            raise RuntimeError(
                "Could not create output video."
            )

        # Save recent positions of tracked objects
        self.track_history = defaultdict(list)

        # Drawing settings
        self.rect_width = 2
        self.font = 0.7
        self.text_width = 2
        self.trail_length = 50
        self.polyline_thickness = 2

        self.window_name = "YOLO Object Tracking"
        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

    def draw_bbox(self, frame, box, track_id, cls):
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box)
        color = colors(int(cls), True)

        # Draw bounding box
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            color, self.rect_width
        )

        # Create object label
        label = f"{self.names[int(cls)]}: {int(track_id)}"

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font,
            self.text_width
        )

        # Position label above the box
        bg_x1 = max(0, x1)
        bg_x2 = min(
            frame.shape[1] - 1,
            bg_x1 + tw + 16
        )
        bg_y2 = max(0, y1)
        bg_y1 = max(0, bg_y2 - th - baseline - 12)

        # Draw label background
        cv2.rectangle(
            frame,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1
        )

        # Center the label text
        text_x = bg_x1 + max(
            0, (bg_x2 - bg_x1 - tw) // 2
        )
        text_y = bg_y1 + max(
            th, (bg_y2 - bg_y1 + th) // 2
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font,
            (255, 255, 255),
            self.text_width,
            cv2.LINE_AA
        )

    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()

                if not success:
                    print("End of video.")
                    break

                # Detect and track objects
                                # Detect and track objects
                results = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    imgsz=416,
                    half=False
                )
                if results and len(results) > 0:
                    boxes = results[0].boxes

                    if boxes is not None and boxes.id is not None:
                        coordinates = boxes.xyxy.cpu().numpy()
                        ids = boxes.id.int().cpu().tolist()
                        classes = boxes.cls.int().cpu().tolist()

                        for box, track_id, cls in zip(
                            coordinates, ids, classes
                        ):
                            self.draw_bbox(
                                frame, box, track_id, cls
                            )

                            x1, y1, x2, y2 = box

                            # Find object center
                            center = (
                                int((x1 + x2) / 2),
                                int((y1 + y2) / 2)
                            )

                            # Store movement history
                            track = self.track_history[track_id]
                            track.append(center)

                            if len(track) > self.trail_length:
                                track.pop(0)

                            # Draw center point
                            cv2.circle(
                                frame,
                                center,
                                5,
                                colors(int(cls), True),
                                -1
                            )

                            # Draw movement trail
                            if len(track) > 1:
                                points = np.array(
                                    track, dtype=np.int32
                                ).reshape((-1, 1, 2))

                                cv2.polylines(
                                    frame,
                                    [points],
                                    False,
                                    colors(int(cls), True),
                                    self.polyline_thickness
                                )

                # Save and display the frame
                self.writer.write(frame)
                cv2.imshow(self.window_name, frame)

                # Press Q to stop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Tracking stopped.")
                    break

        finally:
            self.cap.release()
            self.writer.release()
            cv2.destroyAllWindows()
            print("Output saved as object-tracking.avi")


if __name__ == "__main__":
    tracker = ObjectTracking(
        model="mpt26n.pt",
        source="videos/xyz.mp4"
    )
    tracker.run()
