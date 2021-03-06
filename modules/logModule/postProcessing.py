import matplotlib.pyplot as plt
import sys
import numpy as np

def getAvg(array):
    arr = []
    for i in array:
        arr.append(float(i))
    return (sum(arr)/len(arr),len(arr))

def rewardEnd(fileN):
    arr = []
    with open(fileN) as reward:
        for line in reward:
            lastReward = float(line.strip().split(",")[-1])
            arr.append(lastReward)
    return arr

def uniqueColor():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())

def rewardAvg(fileN):
    arr = []
    with open(fileN) as reward:
        for line in reward:
            avgReward = line.strip().split(",")
            arr.append(getAvg(avgReward))
    return arr

def modelLoss(fileN):
    arr = []
    with open(fileN) as loss:
        for line in loss:
            l = line.strip()
            arr.append(float(l))
    return arr

def modelBalance(fileN,episodeLength):
    arr = []
    with open(fileN) as totalBalance:
        for line in totalBalance:
            bal = line.strip()
            arr.append(float(bal))
    episodicStop = np.cumsum(episodeLength)
    coords = []
    start = 0
    endBal = []
    for end in episodicStop:
        x = []
        y = []
        for i in range(start,end):
            x.append(i)
            y.append(arr[i])
        endBal.append(y[-1])
        start = end
        coords.append((x,y))
    return coords,endBal


def rewardAvgLen(data):
    avgReward = []
    rewardLen = []
    for avg,length in data:
        avgReward.append(avg)
        rewardLen.append(length)
    return avgReward,rewardLen

if __name__ == '__main__':
    #print(sys.argv)
    folderName = sys.argv[1]
    folderName = "Saved_model/"+folderName+"/"
    lastReward = rewardEnd(folderName+"rewardLog.tsv")
    avgReward, episodeLength = rewardAvgLen(rewardAvg(folderName+"rewardLog.tsv"))
    loss = modelLoss(folderName+"lossLog.tsv")

    """
    balanceSet,endBal = modelBalance(folderName+"balanceLog.tsv",episodeLength)

    # Ploting balance
    plt.figure(figsize=(1980,1080),dpi=40000)
    fig, ax = plt.subplots()
    fig
    for x,y in balanceSet:
        ax.plot(x, y, color=uniqueColor())
    plt.xlabel("Iteration (Every episode in different color)")
    plt.ylabel("Balance")
    plt.yticks(np.arange(0, 5000, 100))
    plt.draw()
    fig.savefig(folderName+"balance.svg",dpi=400)
    plt.close()
    #"""

    # Ploting final rewards
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Final reward")
    plt.plot(lastReward)
    plt.savefig(folderName+"finalReward.svg")
    plt.close()


    # Ploting average reward
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(avgReward)
    plt.savefig(folderName+"avgReward.svg")
    plt.close()

    # Ploting final rewards
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.savefig(folderName+"loss.svg")
    plt.close()

    # Ploting end Balance
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Final Balance")
    plt.plot(endBal)
    plt.savefig(folderName+"final Balance.svg")
    plt.close()

