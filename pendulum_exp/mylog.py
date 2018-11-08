"""Log facilities"""
from os import makedirs
from os.path import join, exists
from typing import Dict
import pickle
import numpy as np

class Logger:
    CURRENT = None # current logger

    def __init__(self):
        self._logs: Dict[str, Dict[int, float]] = dict()
        self._buffering = 300
        self._count = 0
        self._dir = None

    def log(self, key: str, value: float, timestamp: int):
        if key not in self._logs:
            self._logs[key] = {timestamp: value}
        else:
            self._logs[key][timestamp] = value
        self._count += 1
        if self._count == self._buffering:
            self._count = 0
            self.dump()

    def log_video(self, tag: str, timestamp: int, frames):
        video_dir = join(self._dir, "videos")
        if not exists(video_dir):
            makedirs(video_dir)
        np.savez(join(video_dir, f"{tag}_{timestamp}.npz"), frames)

    def set_dir(self, logdir):
        self._dir = logdir

    def dump(self):
        assert self._dir is not None
        pickle.dump(self._logs, open(join(self._dir, 'logs.pkl'), 'wb'))

def logto(logdir: str):
    assert Logger.CURRENT is not None
    Logger.CURRENT.set_dir(logdir)

def log(key: str, value: float, timestamp: int):
    assert Logger.CURRENT is not None
    Logger.CURRENT.log(key, value, timestamp)

def log_video(tag: str, timestamp: int, frames):
    assert Logger.CURRENT is not None
    Logger.CURRENT.log_video(tag, timestamp, frames)

def _configure():
    Logger.CURRENT = Logger()


_configure() # configure logger on import
