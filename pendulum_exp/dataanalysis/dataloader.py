import os 
import pickle as pkl
from datetime import datetime

class ExperimentData:
    def __init__(self, experimentrunlist):
        # experimentrunlist = [ExperimentRun(args_file, logs_file) for args_file, logs_file in argslogs_filelist]

        setting_list = []
        for run in experimentrunlist:
            args = run.args
            for setting in setting_list:
                if setting.args == args:
                    # setting.push(run)
                    break
            setting_list.append(ExperimentSetting([run]))

        self._setting_list = setting_list


    @property
    def deltakeys(self):
        if not hasattr(self, "_deltaargs"):
            args_list = [setting.args for setting in self._setting_list]

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
            self._deltaitems = [dict((k, setting.args[k]) for k in self.deltakeys) for setting in self._setting_list]
        return self._deltaitems
    
    def filter_settings(self, bool_function):
        run_list = []
        for setting in self._setting_list:
            if bool_function(setting.args):
                run_list.extend(setting._experimentrunlist)
        return ExperimentData(run_list)

    def repr_rawlogs(self, key, nlastvalues):
        deltakeys = self.deltakeys
        print(f'{nlastvalues} last values of key {key} for all settings')
        for setting in self._setting_list:
            print(' ; '.join(f'{key}: {setting.args[key]}' for key in deltakeys))
            timeseq = setting.timeseq(key)
            if timeseq is None:
                print('No logs')
                print('----')
                continue
            idxs = sorted(list(timeseq.keys()))[-nlastvalues:]
            for i in idxs:
                print(f"{i}: {timeseq[i]}")
            print('----')

    

class ExperimentSetting:
    def __init__(self, experimentrunlist):
        self.args = experimentrunlist[0].args
        assert all(run.args == self.args for run in experimentrunlist)

        self._experimentrunlist = experimentrunlist

    def append(self, experimentrun):
        assert experimentrun == self.args
        self._experimentrunlist.push(experimentrun)

    def timeseq(self, key):
        ## TODO
        if key not in self._experimentrunlist[0].logs:
            return None
        return self._experimentrunlist[0].logs[key]


class ExperimentRun:
    def __init__(self, args_file, logs_file):
        self._args_file = args_file
        self._logs_file = logs_file


    @property
    def args(self):
        print(self._args_file)
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


def loader_leonard(workdir, exp_name, start_date=None, stop_date=None):
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
    list_exp_dates = [d.strftime("%Y_%m_%d_%H_%M_%S/") for d in list_exp_dates]

    argslogs_filelist = []


    for exp_date in list_exp_dates:
        expdate_dir = os.path.join(exp_dir, exp_date)
        job_expdate_list = [d for d in os.listdir(expdate_dir) if os.path.isdir(os.path.join(expdate_dir, d)) and d.isnumeric()]
        for jobid in job_expdate_list:
            job_dir = os.path.join(expdate_dir, jobid)
            args_file = os.path.join(job_dir, 'args')
            logs_file = os.path.join(job_dir, 'logs.pkl')
            argslogs_filelist.append((args_file, logs_file))

    return [ExperimentRun(args_file, logs_file) for args_file, logs_file in argslogs_filelist]



