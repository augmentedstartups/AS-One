import contextlib
from pathlib import Path
import sys

@contextlib.contextmanager
def yolov7_in_syspath():
    """
    Temporarily add yolov5 folder to `sys.path`.
    
    torch.hub handles it in the same way: https://github.com/pytorch/pytorch/blob/75024e228ca441290b6a1c2e564300ad507d7af6/torch/hub.py#L387
    
    Proper fix for: #22, #134, #353, #1155, #1389, #1680, #2531, #3071   
    No need for such workarounds: #869, #1052, #2949
    """
    yolov7_folder_dir = str(Path(__file__).parents[1].absolute())
    try:
        sys.path.insert(0, yolov7_folder_dir)
        yield
    finally:
        sys.path.remove(yolov7_folder_dir)