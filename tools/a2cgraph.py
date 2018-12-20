import tensorflow as tf
import numpy as np
import sys, gym, tqdm, argparse, json, easydict, os, time, random, pprint
import matplotlib.pyplot as plt
from itertools import cycle

def main():
    env_num = int(sys.argv[1])
    config_args = easydict.EasyDict()
    config_args.curvesname  = "curves"
    config_args.doneepsname = "doneeps"
    config_args.max_episodes= int(sys.argv[2])

    doneeps = []
    for i in range(env_num):
        doneeps += [int(doneep) for doneep in open(config_args.doneepsname+str(i),'r').read().splitlines()[-1].split(",")[:-1]]
    plt.figure(111)
    bins = np.linspace(0,config_args.max_episodes,50)
    plt.hist(doneeps, bins=bins, normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
    plt.savefig(config_args.doneepsname.replace('/','T')+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    COLORS = cycle(['black', 'red', 'orange', 'green', 'cyan', 'blue', 'purple'])
    lines = []
    for i in range(env_num):
        lines.append(open(config_args.curvesname+str(i),"r").read().splitlines())
    for j in range(len(lines[0])):
        color=next(COLORS)
        recordmeans = []
        for i in range(env_num):
            #color=next(COLORS)
            record, avgnum = [float(strs) for strs in lines[i][j].split("|")[:-1][:config_args.max_episodes]], 5 ###
            recordmean = [np.mean(record[k:k+avgnum]) for k in range(len(record)-avgnum+1)]
            recordmeans.append(recordmean)
            #plt.figure(112)
            #plt.plot(record,color=color,alpha=0.2)
            #plt.plot(recordmean,color=color,alpha=1.0)
        recordmeansmean = np.array(recordmeans).mean(axis=0)
        recordmeansvar = np.array(recordmeans).std(axis=0)
        plt.figure(113)
        plt.plot(recordmeansmean,color=color,alpha=0.6)
        plt.fill_between(range(len(recordmeansmean)), recordmeansmean-recordmeansvar, recordmeansmean+recordmeansvar,facecolor=color,alpha=0.2)
    plt.savefig(config_args.curvesname.replace('/','T')+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
if __name__ == '__main__':
    main()
