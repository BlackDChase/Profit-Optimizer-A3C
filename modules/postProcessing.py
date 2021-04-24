"""
Post processing to produce graphs from logs
- [X] Reward
- [ ] Policy Loss
- [ ] Critic Loss
- [ ] Advantage

Currently is a RIPOFF of postProcessing made for MIDAS
#"""
__author__ = 'BlackDChase'
__version__ = '0.3.0'

# Imports

import os
import matplotlib.pyplot as plt
import sys
import numpy as np

def getAvg(array):
    arr = []
    for i in array:
        arr.append(float(i))
    return (sum(arr)/len(arr),len(arr))


def uniqueColor():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())

def rewardAvg(fileN):
    arr = []
    with open(fileN) as reward:
        for line in reward:
            avgReward = line.strip().split(",")
            while  len(avgReward)>0 and avgReward[-1]=='':
                avgReward.pop()
            if len(avgReward)<=0:
                continue
            arr.append(getAvg(avgReward))
    return arr

def modelLoss(fileN):
    arr = []
    with open(fileN) as loss:
        for line in loss:
            l = line.strip()
            arr.append(float(l))
    return arr


def rewardAvgLen(data):
    avgReward = []
    rewardLen = []
    for avg,length in data:
        avgReward.append(avg)
        rewardLen.append(length)
    return avgReward,rewardLen

def getMostRecent(folder):
    path=os.path.dirname(os.path.realpath(""))
    pwd = path.split("/")[0]
    if pwd=='Saved_model':
        all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
    raise NotImplemented

if __name__ == '__main__':
    #print(sys.argv)
    #print(os.path.dirname(os.path.realpath("")))
    #folderName = getMostRecent(sys.argv[1])
    folderName = ""
    avgAdvantage,episodeLength = rewardAvgLen(rewardAvg(folderName+"advantageLog.tsv"))
    avgReward, episodeLength = rewardAvgLen(rewardAvg(folderName+"rewardLog.tsv"))
    criticLoss = modelLoss(folderName+"policyLossLog.tsv")
    policyLoss = modelLoss(folderName+"criticLossLog.tsv")

    # Ploting average reward
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(avgReward)
    baseLine = [0.142776477819396]*len(avgReward)
    plt.plot(baseLine,color='r',label='Average Reward of dataset')
    plt.savefig(folderName+"avgReward.svg")
    plt.close()

    # Ploting average advantage
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Average advantage")
    plt.plot(avgAdvantage)
    plt.savefig(folderName+"avgAdvantage.svg")
    plt.close()

    # Ploting episodic Policy Loss
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    plt.plot(policyLoss)
    plt.savefig(folderName+"policyLoss.svg")
    plt.close()

    # Ploting episodic Critic Loss
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("criticLoss")
    plt.plot(criticLoss)
    plt.savefig(folderName+"criticLoss.svg")
    plt.close()
