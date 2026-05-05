import os
import sys
import logging
from PySide6.QtCore import QThread, Signal

# Add SuperNormal to path so it can find its modules
import sys
supernormal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "SuperNormal"))
if supernormal_path not in sys.path:
    sys.path.insert(0, supernormal_path)

from exp_runner import Runner

class SuperNormalWorker(QThread):
    progress = Signal(int, int) # current_iter, max_iter
    log_msg = Signal(str)
    finished = Signal(bool, str) # success, msg
    mesh_extracted = Signal(str) # Path to the extracted mesh ply file

    def __init__(self, conf_text, mode='train', is_continue=False, parent=None):
        super().__init__(parent)
        self.conf_text = conf_text
        self.mode = mode
        self.is_continue = is_continue
        self.runner = None

    def run(self):
        try:
            self.log_msg.emit("Initializing SuperNormal Runner...")
            self.runner = Runner(self.conf_text, self.mode, self.is_continue)
            
            # Wire progress signal: the callback is called every iteration by exp_runner
            self.runner.progress_callback = lambda cur, total: self.progress.emit(cur, total)
            
            self.log_msg.emit("Starting training loop...")
            self.runner.train()
            
            if getattr(self.runner, 'stop_flag', False):
                self.finished.emit(False, "Training stopped by user.")
            else:
                self.finished.emit(True, self.runner.base_exp_dir)
                
        except Exception as e:
            self.log_msg.emit(f"Error in SuperNormalWorker: {str(e)}")
            self.finished.emit(False, str(e))

    def stop(self):
        if self.runner:
            self.runner.stop_flag = True
