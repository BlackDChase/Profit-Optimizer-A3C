"""
Post processing to produce graphs from logs
- [X] A3C Profit
- [X] Normal Profit
- [X] A3C State
- [X] Normal State
- [X] Diff
#"""
__author__ = 'BlackDChase'
__version__ = '0.4.0'

# Imports

import os
import matplotlib.pyplot as plt
import sys
import numpy as np

def uniqueColor():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())


def getMostRecent(folder):
    path=os.path.dirname(os.path.realpath(""))
    pwd = path.split("/")[0]
    if pwd=='Saved_model':
        all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
    raise NotImplemented


def readState(fileN):
    states = []
    with open(fileN) as state:
        for i in state:
            x = i.strip().split(",")
            while "" in x:
                x.remove('')
            states.append(list(map(float,x)))
    states = np.array(states,dtype=np.float32).transpose()
    return states

def readProfit(fileN):
    profit = []
    with open(fileN) as state:
        for i in state:
            profit.append(float(i.strip()))
    return profit


if __name__ == '__main__':
    #print(sys.argv)
    #print(os.path.dirname(os.path.realpath("")))
    #folderName = getMostRecent(sys.argv[1])
    folderName = ""
    #a3cState = readState(folderName+"A3CState.tsv")
    a3cProfit = readProfit(folderName+"A3CProfit.tsv")
    #normalState = readState(folderName+"NormalState.tsv")
    normalProfit = readProfit(folderName+"NormalProfit.tsv")
    diff = readProfit(folderName+"ProfitDiff.tsv")

    # Ploting Profit
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(a3cProfit,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('A3C Profit',color=color)
    ax1.set_xlabel(f"Time step")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(normalProfit,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Profit w/o A3C',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"Profit.svg")
    plt.close()

    # Plotting Differnce in Profit
    fig,ax = plt.subplots(dpi=100)
    ax.set_xlabel(f"Time step")
    ax.plot(diff)
    ax.set_ylabel(f"A3C Profit - Normal Profit")
    fig.tight_layout()
    plt.savefig(folderName+"Differnce in profit.svg")
    plt.close()


    """
    # Ploting States
    # Plotting A3C State
    states=len(normalState)
    square=int(np.ceil(states**(1/2)))
    fig,ax = plt.subplot(square,square,dpi=800)
    for i in range(states):
        lstm=2*i
        a3c=lstm+1

        x,y=lstm//square,lstm%square
        ax1=ax[x,y]
        ax1.set_title(f"State {i}")
        ax1.set_xlabel(f"Time step")

        color='r'
        ax1.plot(normalState[i],color=color)
        ax1.tick_params(axis='y',labelcolor=color)
        ax1.set_ylabel('Normal',color=color)

        ax2 = ax1.twinx()
        color='b'
        ax2.plot(normalProfit,color=color)
        ax2.tick_params(axis='y',labelcolor=color)
        ax2.set_ylabel('A3C',color=color)

    fig.tight_layout()
    plt.savefig(folderName+"States.svg")
    plt.close()
    """
