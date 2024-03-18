import os
import sys
import traceback


class PathResolver:
    def __init__(self, path: str = None) -> None:
        if path is None:
            stack = traceback.extract_stack()
            file_path = stack[-2].filename
            dir_path = os.path.dirname(file_path)
            path = os.path.join(dir_path, os.path.basename(dir_path))
                    
        self.path = path        
        
    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.pop(0)
 