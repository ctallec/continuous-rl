"""Log facilities"""
from abc import ABC, abstractmethod
from os import makedirs
from os import remove
from shutil import rmtree
from os.path import join, exists, isfile, dirname
from typing import Dict
import pickle
import numpy as np
from scipy.misc import toimage

from logging import info

class KVTWritter(ABC):
    @abstractmethod
    def writekvts(self, key: str, value: float, timestamp: int):
        pass

    @abstractmethod
    def set_dir(self, logdir: str, reload: bool = False):
        pass

    @abstractmethod
    def load(self):
        pass

class PickleKVTWriter(KVTWritter):
    def __init__(self):
        self._logs: Dict[str, Dict[int, float]] = dict()
        self._buffering = 500
        self._count = 0
        self._dir = None

    def writekvts(self, key: str, value: float, timestamp: int):
        if key not in self._logs:
            self._logs[key] = {timestamp: value}
        else:
            self._logs[key][timestamp] = value
        self._count += 1
        if self._count == self._buffering:
            self._count = 0
            self.dump()

    def set_dir(self, logdir: str, reload: bool = True):
        self._dir = logdir
        log_file = join(self._dir, 'logs.pkl')
        if isfile(log_file):
            if reload:
                self.load()
            else:
                remove(log_file)
                rmtree(join(self._dir, "videos"), ignore_errors=True)

        info("logfile: {}".format(join(self._dir, 'logs.pkl')))

    def load(self):
        assert self._dir is not None
        with open(join(self._dir, 'logs.pkl'), 'rb') as f:
            self._logs = pickle.load(f)

    def dump(self):
        assert self._dir is not None
        pickle.dump(self._logs, open(join(self._dir, 'logs.pkl'), 'wb'))

class TensorboardKVTWriter(KVTWritter):
    def __init__(self):
        self._writer = None
        self._dir = None

    def set_dir(self, logdir: str, reload: bool = True):
        self._dir = join(logdir, 'train')
        try:
            makedirs(self._dir)
        except OSError:
            if not reload:
                rmtree(self._dir, ignore_errors=True)
                makedirs(self._dir)
        self.load()

    def load(self):
        from tensorboardX import SummaryWriter
        self._writer = SummaryWriter(self._dir)
        args_file = join(dirname(self._dir), 'args')
        args = vars(pickle.load(open(args_file, 'rb')))
        args_str = "|Arg\t|Value\t|\n"
        args_str += "|---\t|---\t|\n"
        args_str += "\n".join([f"|**{k}**\t|{str(v)}\t|" for (k, v) in args.items()])
        print(args_str)
        self._writer.add_text('args', args_str)

    def writekvts(self, key: str, value: float, timestamp: int):
        self._writer.add_scalar(key, value, timestamp)

class Logger:
    CURRENT = None # current logger

    def __init__(self):
        assert Logger.CURRENT is None
        self._writers = [TensorboardKVTWriter(), PickleKVTWriter()]
        self._dir = None

    def log(self, key: str, value: float, timestamp: int):
        for writer in self._writers:
            writer.writekvts(key, value, timestamp)

    def log_video(self, tag: str, timestamp: int, frames):
        video_dir = join(self._dir, "videos")
        if not exists(video_dir):
            makedirs(video_dir)
        np.savez(join(video_dir, f"{tag}_{timestamp}.npz"), frames)

    def log_image(self, tag: str, timestamp: int, image):
        img_dir = join(self._dir, "imgs")
        if not exists(img_dir):
            makedirs(img_dir)
        toimage(image).save(join(img_dir, f"{tag}_{timestamp}.png"))

    def set_dir(self, logdir: str, reload: bool = True):
        self._dir = logdir
        for writer in self._writers:
            writer.set_dir(logdir, reload)

def logto(logdir: str, reload: bool = True):
    assert Logger.CURRENT is not None
    Logger.CURRENT.set_dir(logdir, reload)

def log(key: str, value: float, timestamp: int):
    assert Logger.CURRENT is not None
    Logger.CURRENT.log(key, value, timestamp)

def log_video(tag: str, timestamp: int, frames):
    assert Logger.CURRENT is not None
    Logger.CURRENT.log_video(tag, timestamp, frames)

def log_image(tag: str, timestamp: int, image):
    assert Logger.CURRENT is not None
    Logger.CURRENT.log_image(tag, timestamp, image)


if Logger.CURRENT is None:
    Logger.CURRENT = Logger() # type: ignore
