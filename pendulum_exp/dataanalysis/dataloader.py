import os
import pickle as pkl
from datetime import datetime
from typing import List, Callable, Any
import numpy as np

class ExperimentData:
    def __init__(self, runs: List['ExperimentRun']) -> None:
        # experimentrunlist = [ExperimentRun(args_file, logs_file) for args_file, logs_file in argslogs_filelist]

        settings: List[ExperimentSetting] = []
        for run in runs:
            args = run.args
            for setting in settings:
                if setting.args == args:
                    setting.append(run)
                    break
            else:
                settings.append(ExperimentSetting([run]))

        self._settings = settings


    @property
    def deltakeys(self):
        if not hasattr(self, "_deltaargs"):
            args_list = [setting.args for setting in self._settings]

            expargsdict = {}
            for args in args_list:
                for k, v in args.items():
                    if k not in expargsdict:
                        expargsdict[k] = set()
                    expargsdict[k].add(v)

            self._deltaargs = [k for k, vset in expargsdict.items() if len(vset) > 1]
            self._sharedargs = dict((k, vset.pop()) for k, vset in expargsdict.items() if len(vset) == 1)

        return self._deltaargs

    @property
    def sharedargs(self):
        if not hasattr(self, "_sharedargs"):
            self.deltakeys
        return self._sharedargs


    def deltaitems(self):
        if not hasattr(self, "_deltaitems"):
            self._deltaitems = [dict((k, setting.args[k]) for k in self.deltakeys) for setting in self._settings]
        return self._deltaitems

    def filter_settings(self, predicate: Callable[[Any], bool]):
        runs: List[ExperimentRun] = []
        for setting in self._settings:
            if predicate(setting.args):
                runs.extend(setting._runs)
        return ExperimentData(runs)

    def repr_rawlogs(self, key: str, nlastvalues: int):
        deltakeys = self.deltakeys
        print(f'{nlastvalues} last values of key {key} for all settings')
        for setting in self._settings:
            print(' ; '.join(f'{key}: {setting.args[key]}' for key in deltakeys))
            stats = setting.timeseq(key)
            timeseq = stats['mean']
            std = stats['std']
            if stats is None:
                print('No logs')
                print('----')
                continue
            idxs = sorted(list(timeseq.keys()))[-nlastvalues:]
            for i in idxs:
                print(f"{i}: {timeseq[i]}+-{std[i]}")
            print('----')


class ExperimentSetting:
    def __init__(self, runs: List['ExperimentRun']) -> None:
        self.args = runs[0].args
        assert all(run.args == self.args for run in runs)

        self._runs = runs

    def append(self, run: 'ExperimentRun'):
        assert run.args == self.args
        self._runs.append(run)

    def timeseq(self, key: str):
        # TODO
        if key not in self._runs[0].logs:
            return None
        timeseqs = {k: [r.logs[key][k] for r in self._runs if k in r.logs[key]]
                    for k in self._runs[0].logs[key]}
        stats_seq = {k: {
            'mean': np.mean(t),
            'std': np.std(t),
            'median': np.median(t),
            'max': np.max(t),
            'min': np.min(t)} for k, t in timeseqs.items()}
        timeseq = {}
        for k1 in stats_seq:
            for k2 in stats_seq[k1]:
                if k2 not in timeseq:
                    timeseq[k2] = {}
                timeseq[k2][k1] = stats_seq[k1][k2]
        return timeseq

class ExperimentRun:
    def __init__(self, args_file: str, logs_file: str) -> None:
        self._args_file = args_file
        self._logs_file = logs_file


    @property
    def args(self):
        if not hasattr(self, '_args'):
            assert os.path.isfile(self._args_file)

            with open(os.path.join(self._args_file), 'rb') as f:
                args = pkl.load(f)
                if not isinstance(args, dict):
                    args = vars(args)
                self._args = args

        return self._args

    @property
    def logs(self):
        if not hasattr(self, '_logs'):
            assert os.path.isfile(self._logs_file)

            with open(self._logs_file, 'rb') as f:
                self._logs = pkl.load(f)

        return self._logs


def loader_leonard(workdir: str, exp_name: str, start_date=None, stop_date=None):
    list_exp_names = os.listdir(workdir)
    if exp_name not in list_exp_names:
        raise ValueError('The experience {} is not in the work directory {}'.format(exp_name, workdir))

    exp_dir = os.path.join(workdir, exp_name)


    list_exp_dates = [datetime.strptime(exp_dir, "%Y_%m_%d_%H_%M_%S") for exp_dir in os.listdir(exp_dir)]

    if start_date == 'last':
        list_exp_dates = [max(list_exp_dates)]
    else:
        if start_date is not None:
            list_exp_dates = [d for d in list_exp_dates if start_date <= d]
        if stop_date is not None:
            list_exp_dates = [d for d in list_exp_dates if d < stop_date]
    list_exp_dates = [d.strftime("%Y_%m_%d_%H_%M_%S/") for d in list_exp_dates] # type: ignore

    argslogs_filelist = []


    for exp_date in list_exp_dates:
        expdate_dir = os.path.join(exp_dir, str(exp_date))
        job_expdate_list = [d for d in os.listdir(expdate_dir) if os.path.isdir(os.path.join(expdate_dir, d)) and d.isnumeric()]
        for jobid in job_expdate_list:
            job_dir = os.path.join(expdate_dir, jobid)
            args_file = os.path.join(job_dir, 'args')
            logs_file = os.path.join(job_dir, 'logs.pkl')
            argslogs_filelist.append((args_file, logs_file))

    return [ExperimentRun(args_file, logs_file) for args_file, logs_file in argslogs_filelist]
