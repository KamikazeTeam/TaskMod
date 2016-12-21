from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict

def default_config():
    c = OrderedDict()
    c['nb_runs'] = 1
    c['nb_epoch']   = 16
    c['patience']   = 4
    c['batch_size'] = 32#160
    c['epoch_fract']= 1
    return c

class AbstractTask(object):
    def __init__(self):
        self.c = default_config()
        self.task_config()
    def get_conf(self):
        return self.c
    def set_conf(self, c):
        self.c = c
    def initial(self):
        self.task_initial()
    def load_resources(self):
        return self.task_load_resources()
    def load_data(self, pfx, trainf, validf, testf):
        self.trainf= trainf
        self.validf= validf
        self.testf = testf
        self.gr  = self.task_load_data(pfx+trainf)
        self.grv = self.task_load_data(pfx+validf)
        self.grt = self.task_load_data(pfx+testf)

    def build_model(self, mod_prep_model):
        return self.task_build_model(mod_prep_model)

    def eval_model(self, model, predout=''):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.validf), (self.grt, self.testf)]:
            res.append(self.task_eval_model(model, gr, fname, predout=predout))
        return tuple(res)
    def fit_model(self, model, wfname, iwfname=None):
        return self.task_fit_model(model, self.gr, self.grv, self.grt, wfname, iwfname)

    def debug_model(self, mod_debug_model):
        return self.task_debug_model(mod_debug_model)
