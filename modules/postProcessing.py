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

def stateExtract(fileN,order):
    with open(fileN) as state:
        old=[]
        new=[]
        demand=[]
        temp=order
        x,y,z=0,0,0
        for i in state:
            sp = i.strip().split(",")
            x+=float(sp[0])
            y+=float(sp[1])
            z+=float(sp[2])
            if temp==0:
                temp=order
            if temp==order:
                x/=order
                y/=order
                z/=order
                old.append(x)
                new.append(y)
                demand.append(z)
                z,y,z=0,0,0
            temp-=1
    return old,new,demand



if __name__ == '__main__':
    #print(sys.argv)
    #print(os.path.dirname(os.path.realpath("")))
    #folderName = getMostRecent(sys.argv[1])
    folderName = ""
    avgAdvantage,episodeLength = rewardAvgLen(rewardAvg(folderName+"advantageLog.tsv"))
    avgReward, episodeLength = rewardAvgLen(rewardAvg(folderName+"rewardLog.tsv"))
    policyLoss = modelLoss(folderName+"policyLossLog.tsv")
    criticLoss = modelLoss(folderName+"criticLossLog.tsv")
    old,new,demand = stateExtract(folderName+"stateLog.tsv",len(episodeLength)/4)

    # Ploting LSTM Price vs Demand 
    plt.figure(dpi=400)
    plt.xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    plt.plot(old,label="Actual Price")
    plt.plot(demand,label="Demand")
    plt.legend()
    plt.savefig(folderName+"Actual Price vs Demand.svg")
    plt.close()

    # Ploting A3C Price vs Demand
    plt.figure(dpi=400)
    plt.xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    plt.plot(new,label="Model Price")
    plt.plot(demand,label="Demand")
    plt.legend()
    plt.savefig(folderName+"Model Price vs Demand.svg")
    plt.close()


    # Ploting LSTM Price vs Actual Price
    plt.figure(dpi=400)
    plt.xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    plt.plot(new,label="Model Price")
    plt.plot(old,label="Actual Price")
    plt.legend()
    plt.savefig(folderName+"Model Price vs Actual Price.svg")
    plt.close()

    # Ploting average reward
    plt.figure(dpi=400)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(avgReward)
    baseLine = [0.142776477819396]*len(avgReward)
    plt.plot(baseLine,color='r',label='Average Reward of dataset')
    plt.savefig(folderName+"avgReward.svg")
    plt.close()

    # Ploting average advantage
    plt.figure(dpi=400)
    plt.xlabel("Episode")
    plt.ylabel("Average advantage")
    plt.plot(avgAdvantage)
    plt.savefig(folderName+"avgAdvantage.svg")
    plt.close()

    # Ploting episodic Policy Loss
    plt.figure(dpi=400)
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    plt.plot(policyLoss)
    plt.savefig(folderName+"policyLoss.svg")
    plt.close()

    # Ploting episodic Critic Loss
    plt.figure(dpi=400)
    plt.xlabel("Episode")
    plt.ylabel("criticLoss")
    plt.plot(criticLoss)
    plt.savefig(folderName+"criticLoss.svg")
    plt.close()
