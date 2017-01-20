#!/usr/bin/python3
#set fileencoding=utf8

from __future__ import print_function
from __future__ import division

import numpy as np

import tasks
import mods

fname_plot="./simplot"
fname_log ="./simlog"
path_ress ='../ress/'
path_data ='../data/'
path_wf   ='../weights/bestvalid-'
path_pred ='./'

if __name__ == "__main__":
    import time
    start_time = time.time()
    import sys
    modname, taskname, trainf, validf, testf = sys.argv[1:6]
    params = sys.argv[6:]

    import importlib
    task= importlib.import_module('.'+taskname,'tasks').task()
    mod = importlib.import_module( '.'+modname, 'mods').mod()

    conf = task.get_conf()
    mod.add_conf(conf)
    for param in params:
        name, value= param.split('=')
        try:
            conf[name] = eval(value)###eval
        except:
            conf[name] = value###eval
    fplot = open(fname_plot,'a')
    conf['fplot']=fplot###
    task.set_conf(conf)
    import json
    h = hash(json.dumps(dict([(n, str(v)) for n, v in conf.items()]), sort_keys=True))
    print('', file=fplot)###
    print('H: %x' % h, file=fplot)
    print(conf,end='',file=fplot)
    flog  = open(fname_log,'a')
    print('H: %x' % h, file=flog)
    print(conf,file=flog)

    task.initial()
    task.load_resources(path_ress)
    task.load_data(path_data, trainf, validf, testf)
    end_time = time.time()
    print('Time:%f' %(end_time-start_time))
    print('Time:%f' %(end_time-start_time), file=flog)

    res = {trainf: [], validf: [], testf: []}
    bestresult = 0.0
    bestwfname = None
    import keras
    from keras import backend as K
    #csgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    for i_run in range(conf['nb_runs']):###
        print()###
        print("",file=conf['fplot'])###
        runid = '%s-%s-%x-%02d' % (taskname, modname, h, i_run)
        print('RunID: %s\n%s' % (runid, conf))

        print('Creating Model')
        task.create_model(mod)
        if 'fweight' in conf: model.load_weights(conf['fweight'], by_name=True)

        print('Training')
        task.compile_model()
        wfname = path_wf+runid
        task.fit_model(wfname)
        if conf['debug']: task.debug_model(mod)###
        ############################
        if 0:
            conf = task.get_conf()
            conf['opt'] = 'sgd'#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
            task.set_conf(conf)
            task.compile_model()
            model=task.get_model()
            model.optimizer.lr = K.variable(0.1)
            model.load_weights(wfname)
            task.set_model(model)
            task.fit_model(wfname)
        ############################
        print('Predict&Eval (best val epoch)')
        model=task.get_model()
        model.load_weights(wfname)
        task.set_model(model)
        resT, resv, rest = task.eval_model()
        res[trainf].append(resT)
        res[validf].append(resv)
        res[testf].append(rest)
        if resv.Pearson>bestresult:###use pearson
            bestresult = resv.Pearson
            bestwfname = wfname
    print()
    print('Output Best Run Prediction')
    model=task.get_model()
    model.load_weights(bestwfname)
    task.set_model(model)
    task.pred_model(predout=path_pred+'%x'%h)

    #Report statistics and compute 95% bounds
    print()
    print('Output Static Results')
    fres = {trainf: {}, validf: {}, testf: {}}  # res ransposed from split-run-field to split-field-run
    mres = {trainf: {}, validf: {}, testf: {}}  # mean
    bres = {trainf: {}, validf: {}, testf: {}}  # 95% bound
    import scipy.stats as ss
    def stat(file, niter, fname, qty, r, alpha=0.95, bonferroni=1.):
        if len(r)==0: bar = np.nan
        else: bar = ss.t.isf( (1-alpha)/bonferroni/2, len(r)-1 ) * np.std(r) / np.sqrt(len(r))
        print('%s: Final %d×%s %f ±%f (%s)' % (fname, niter, qty, np.mean(r), bar, r))
        print('%s: Final %d×%s %f ±%f (%s)' % (fname, niter, qty, np.mean(r), bar, r), file=file)
        return bar
    for split in [trainf, validf, testf]:
        if split is None or not res[split]: continue
        for field in res[validf][0]._fields:
            values = [getattr(r, field) for r in res[split]]
            fres[split][field] = values
            mres[split][field] = np.mean(values)
            bres[split][field] = stat(flog, conf['nb_runs'], split, field, values)
    for field in res[validf][0]._fields:
        if fres[trainf]:
            print('train-valid %s Pearsonr: %f' 
                % (field, ss.pearsonr(fres[trainf][field], fres[validf][field])[0],))
            print('train-valid %s Pearsonr: %f' 
                % (field, ss.pearsonr(fres[trainf][field], fres[validf][field])[0],), file=flog)
        if fres[testf]:
            print('valid-test %s Pearsonr: %f' 
                % (field, ss.pearsonr(fres[validf][field], fres[testf][field])[0],))
            print('valid-test %s Pearsonr: %f' 
                % (field, ss.pearsonr(fres[validf][field], fres[testf][field])[0],), file=flog)
    def res_columns(mres, pfx=' '):
        return('%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[trainf]['Pearson'],
                  pfx, mres[validf]['Pearson'],
                  pfx, mres[testf].get('Pearson', np.nan)))###use pearson
    print('| % -24s |%s | %s' % (modname, res_columns(mres),
          '(defaults)' if not params
          else ' '.join(['``%s``' % (p,) for p in params if not p.startswith('vocabf=')])))
    print('| % -24s |%s |' % ('', res_columns(bres, '±')))
    print('| % -24s |%s | %s' % (modname, res_columns(mres),
          '(defaults)' if not params
          else ' '.join(['``%s``' % (p,) for p in params if not p.startswith('vocabf=')])), file=flog)
    print('| % -24s |%s |' % ('', res_columns(bres, '±')), file=flog)
    end_time = time.time()
    print('Time:%f' %(end_time-start_time))
    print('Time:%f' %(end_time-start_time), file=flog)
