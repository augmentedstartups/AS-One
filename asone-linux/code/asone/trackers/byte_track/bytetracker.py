from .tracker.byte_tracker import BYTETracker
import cv2

class ByteTrack(object):
    def __init__(self, detector, min_box_area=10):
        self.min_box_area = min_box_area

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.detector = detector
        self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        self.tracker = BYTETracker(frame_rate=30)

    def inference(self, image):
 
        dets, image_info = self.detector.detect(image, input_shape=self.input_shape)

        bboxes, ids, scores = self._tracker_update(
            dets,
            image_info,
        )

        image = self.draw_tracking_info(
            image,
            bboxes,
            ids,
            scores,
        )

        return image
        
    def get_id_color(self, index):
        temp_index = abs(int(index)) * 3
        color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
                (29 * temp_index) % 255)
        return color

    def draw_tracking_info(
        self,
        image,
        tlwhs,
        ids,
        scores,
        frame_id=0,
        elapsed_time=0.,
    ):
        text_scale = 1.5
        text_thickness = 2
        line_thickness = 2

        text = 'frame: %d ' % (frame_id)
        text += 'elapsed time: %.0fms ' % (elapsed_time * 1000)
        text += 'num: %d' % (len(tlwhs))
        cv2.putText(
            image,
            text,
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 255, 0),
            thickness=text_thickness,
        )
        
        for index, tlwh in enumerate(tlwhs):
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = x1 + int(tlwh[2]), y1 + int(tlwh[3])
            color = self.get_id_color(ids[index])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            text = str(ids[index])
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, (0, 0, 0), text_thickness + 3)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, (255, 255, 255), text_thickness)
        return image

    def _tracker_update(self, dets, image_info):
        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_tlwhs, online_ids, online_scores