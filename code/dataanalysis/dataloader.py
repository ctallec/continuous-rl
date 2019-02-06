import os
from analog.load import load, LoadPredicate
from datetime import datetime

def loader(workdir: str, exp_name: str, start_date=None, stop_date=None):
    list_exp_names = os.listdir(workdir)
    if exp_name not in list_exp_names:
        raise ValueError(f'The experience {exp_name} '
                         'is not in the work directory {workdir}')

    exp_dir = os.path.join(workdir, exp_name)

    # build predicate
    if start_date == 'last':
        predicate = LoadPredicate(nb_lasts=1)
    else:
        if start_date is None:
            start_date = datetime(1900, 1, 1, 1, 1, 1)
        if stop_date is None:
            stop_date = datetime.now()
        predicate = LoadPredicate(time_range=(start_date, stop_date))

    return load(exp_dir, predicate)
