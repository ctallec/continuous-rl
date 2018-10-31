"""Log facilities"""
from typing import Dict
import pickle

class Logger:
    CURRENT = None # current logger

    def __init__(self):
        self._logs: Dict[str, Dict[int, float]] = dict()
        self._buffering = 10
        self._count = 0
        self._file = None

    def log(self, key: str, value: float, timestamp: int):
        if key not in self._logs:
            self._logs[key] = {timestamp: value}
        else:
            self._logs[key][timestamp] = value
        self._count += 1
        if self._count == self._buffering:
            self._count = 0
            self.dump()

    def set_file(self, filename):
        self._file = filename

    def dump(self):
        assert self._file is not None
        pickle.dump(self._logs, open(self._file, 'wb'))

def logto(filename: str):
    assert Logger.CURRENT is not None
    Logger.CURRENT.set_file(filename)

def log(key: str, value: float, timestamp: int):
    assert Logger.CURRENT is not None
    Logger.CURRENT.log(key, value, timestamp)

def _configure():
    Logger.CURRENT = Logger()


_configure() # configure logger on import
