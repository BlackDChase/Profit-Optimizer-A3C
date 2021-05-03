"""
Post processing to produce graphs from logs
Combinations:
    - [X] Reward
    - [X] Policy Loss
    - [X] Critic Loss
    - [X] Advantage
    - [X] Demand
    - [X] Supply
    - [X] Price
#"""

__author__ = 'BlackDChase'
__version__ = '0.4.0'

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

def stateExtract(fileN,order=None):
    with open(fileN) as state:
        price=[]
        corre=[]
        demand=[]
        supply=[]
        temp=order
        x,y,z,w=0,0,0,0
        for i in state:
            sp = i.strip().split(",")
            x+=float(sp[0])
            y+=float(sp[1])
            z+=float(sp[2])
            w+=float(sp[3])
            if temp==0:
                temp=order
            if temp==order:
                if order!=None:
                    x/=order
                    y/=order
                    z/=order
                    w/=order
                price.append(x)
                corre.append(y)
                demand.append(z)
                supply.append(w)
                x,y,z,w=0,0,0,0
            if order!=None:
                temp-=1
    return price,corre,demand,supply



if __name__ == '__main__':
    #print(sys.argv)
    #print(os.path.dirname(os.path.realpath("")))
    #folderName = getMostRecent(sys.argv[1])
    folderName = ""
    avgAdvantage,episodeLength = rewardAvgLen(rewardAvg(folderName+"advantageLog.tsv"))
    avgReward, episodeLength = rewardAvgLen(rewardAvg(folderName+"rewardLog.tsv"))
    policyLoss = modelLoss(folderName+"policyLossLog.tsv")
    criticLoss = modelLoss(folderName+"criticLossLog.tsv")
    priceAvg,correAvg,demandAvg,supplyAvg = stateExtract(folderName+"stateLog.tsv",len(episodeLength)/4)
    price,corre,demand,supply = stateExtract(folderName+"stateLog.tsv")
    demSupAvg = [-supplyAvg[i]+demandAvg[i] for i in range(len(demandAvg))]
    demSup = [-supply[i]+demand[i] for i in range(len(demand))]
    correAvg2 = []
    for i in episodeLength:
        correAvg2.append(sum(corre[:i])/i)
        corre=corre[i:]


    # Ploting Demand, Supply 
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(demandAvg,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Demand',color=color)
    ax1.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(demSupAvg,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Demand-Supply',color=color)
    ax3 = ax1.twinx()
    color='g'
    ax3.plot(supplyAvg,color=color)
    ax3.tick_params(axis='y',labelcolor=color)
    ax3.set_ylabel('Supply',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"Supply vs Demand.svg")
    plt.close()



    # Ploting AVG A3C Price vs Exchange
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(priceAvg,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Model Price',color=color)
    ax1.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(demSupAvg,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Demand-Supply',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"AVG Model Price vs Exchange.svg")
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

    # Ploting average reward
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(avgReward,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Average Reward',color=color)
    ax1.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(correAvg2,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Average Correction',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"avgRewardnCorrection.svg")
    plt.close()

    #"""
    # Ploting A3C Price vs Exchange
    plt.figure(dpi=400)
    plt.xlabel(f"Episode")
    plt.plot(price,label="Model Price")
    plt.plot(demSup,label="Demand-Supply")
    plt.legend()
    plt.savefig(folderName+"Price VS Exchange.svg")
    plt.close()
    #"""
