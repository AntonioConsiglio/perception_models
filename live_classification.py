import os
import cv2
#import matplotlib.pyplot as plt
#from video_classifier import VideoClasifier
from video_classifier_onnx import VideoClasifierONNX
USE_MATPLOTLIB = False
import numpy as np
#backends = ['Qt5Agg', 'TkAgg', 'Agg']

if USE_MATPLOTLIB:
    import matplotlib
    import numpy as np
    os.environ["QT_QPA_PLATFORM"]="offscreen"
    matplotlib.use('TkAgg')
    from matplotlib.backend_bases import MouseButton,KeyEvent,MouseEvent

class VideoGenerator:
    def __init__(self, cap, output_path, alpha=0.5, show_display=False):
        self.cap = cap
        self.output_path = output_path
        self.alpha = alpha
        self.out = None
        self.frame_idx = 0
        self.show_display = show_display
    
    def _init_display(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dpi = 100
        figsize = (width / dpi, height / dpi)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.img_display = self.ax.imshow(np.zeros((height, width, 3)))
        self.ax.axis('off')
        plt.show(block=False)

    def __enter__(self):
        # Initialize video writer
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if self.show_display and USE_MATPLOTLIB:
            self._init_display()
        return self

    def add_frame(self, frame, heatmap_colored, bounding_boxes, predicted_class, confidence, frame_name):
        
        if heatmap_colored is not None:
            overlay = cv2.addWeighted(frame, 1 - self.alpha, heatmap_colored, self.alpha, 0)
        else:
            overlay = frame
        # Draw bounding box
        if bounding_boxes is not None:
            x1, y1, x2, y2 = bounding_boxes
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw class text with background rectangle
        text = f"{predicted_class} : {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Measure text size
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        margin = 5
        rect_x1, rect_y1 = 10, 10
        rect_x2, rect_y2 = rect_x1 + text_w + 2 * margin, rect_y1 + text_h + 2 * margin

        # Draw filled rectangle background
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

        # Put text
        text_x = rect_x1 + margin
        text_y = rect_y1 + text_h + margin - 2
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Write frame
        self.out.write(overlay)

        if self.show_display:
            if USE_MATPLOTLIB:
                img_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                self.img_display.set_data(img_rgb)
                self.fig.canvas.draw_idle()
                plt.pause(0.001)  # Small pause to update display
            else:
                overlay = self._check_frame_shape_and_resize(overlay)
                cv2.imshow(frame_name, overlay)
                cv2.waitKey(1)

        self.frame_idx += 1

    def _check_frame_shape_and_resize(self,frame):
        ratio = 1
        if max(frame.shape) > 1280:
            ratio = 1280 / max(frame.shape)
        resized = cv2.resize(frame,(0,0),fx=ratio,fy=ratio) 
        return resized
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.out:
            self.out.release()
        if self.show_display:
            if USE_MATPLOTLIB:
                plt.close(self.fig)
            else:
                cv2.destroyAllWindows()
# Example usage with your existing model
if __name__ == "__main__":

    # Your existing setup
    model_name = 'PE-Core-B16-224' 
    # captions = [
    #     "a violent fight between people",
    #     "a violent fight in the street",
    #     "people fighting",
    #     "people arguing and fighting",
    #     "people walking without fighting",
    #     "people talking calmly",
    #     "a peaceful conversation",
    #     "people standing without fighting",
    #     "people interacting peacefully",
    #     "people drinking water",
    #     "people drinking using a cup",
    #     "no people making actions"
    # ]
    captions = [
        "people drinking water",
        "no people making actions",
        "people drinking using a cup",
        "people watching in front of the camera",
        "people clapping the hands",
        "people using the smarthphone",
        "people reading a book"
    ]


    # video_classifier = VideoClasifier(model_name)
    # video_classifier.encode_labes(captions)
    # ONNX
    print("LOADING MODELS IN VIDEO CLASSIFIER...")
    video_classifier_onnx = VideoClasifierONNX(model_name)
    print("VIDEO CLASSIFIER LOADED SUCCESSFULLY ...")
    video_classifier_onnx.encode_labes(captions)
    print("Labels encoded successfully start the video capture!!")
    video_path="./apps/pe/docs/assets/fi001.mp4"
    video_path="./test_fighting.mp4"
    video_path="./Street fighting.mp4"
    video_path = 0
    videocap = cv2.VideoCapture(video_path)
    status = True

    with VideoGenerator(videocap,"result_video_street.mp4",alpha=0.2,show_display=True) as vg:
        with VideoGenerator(videocap,"result_video_street_onnx.mp4",alpha=0.3,show_display=True) as vg_onnx:
            while status:
                status,frame = videocap.read()
                if not status:
                    break
                H,W,_ = frame.shape
                # results = video_classifier.classify_video(frame)
                results_onnx = video_classifier_onnx.classify_video(frame)
        
                # Print results
                # print("=" * 50)
                # print("PRODUCTION VIDEO ANALYSIS RESULTS")
                # print("=" * 50)
                # print(f"Predicted Class: {results['class_name']}")
                # print(f"Confidence: {results['confidence']:.3f}")
                # print(f"Bounding Box: {results['bounding_box']}")
                # # print(f"Spatial Coverage: {results['spatial_coverage']:.2%}")
                # # print(f"Inference Time: {results['inference_time']:.3f}s")
                # print(f"All Probabilities: {[f'{p:.3f}' for p in results['all_probabilities']]}")


                # vg.add_frame(
                #     frame,
                #     results['heatmap'],
                #     results['bounding_box'],
                #     results['class_name'],
                #     results["confidence"],
                #     "Pytorch_results"
                # )
                vg_onnx.add_frame(
                    frame,
                    results_onnx['heatmap'],
                    results_onnx['bounding_box'],
                    results_onnx['class_name'],
                    results_onnx["confidence"],
                    "ONNX_results"
                )

