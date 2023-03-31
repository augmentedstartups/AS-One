from asone.trackers import ByteTrack
from asone.trackers import NorFair
from asone.trackers import DeepSort
from asone.trackers import Motpy
from asone.trackers import OcSort
from asone.trackers import StrongSort


class Tracker:
    def __init__(self, tracker: int, detector: object, use_cuda=True) -> None:
        
        self.trackers = {
            '0': ByteTrack,
            '1': DeepSort,
            '2': NorFair,
            '3': Motpy,
            '4': OcSort,
            '5': StrongSort
        }

        self.tracker = self._select_tracker(tracker, detector, use_cuda=use_cuda)

    def _select_tracker(self, tracker, detector, use_cuda):
        _tracker = self.trackers.get(str(tracker), None)

        if _tracker is not None:
            if _tracker is DeepSort:
                return _tracker(detector, use_cuda=use_cuda)
            else:
                return _tracker(detector)
        else:
            raise ValueError(f'Invalid tracker: {tracker}')

    def detect_and_track(self, image, config: dict):
        
        return self.tracker.detect_and_track(image, config)

    def get_tracker(self):
        return self.tracker