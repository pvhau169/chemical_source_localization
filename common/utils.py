import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth


class MeanVal:
    def __init__(self):
        self.k = 0
        self.val = 0
        self.mean = 0

    def append(self, x):
        self.k += 1
        self.val += x
        self.mean = self.val / self.k

    def get(self):
        return self.mean


class Logger:
    def __init__(self):
        self.log = dict()
        self.name = ''
    def setName(self, name):
        self.name = name
    def loadLog(self, path, show_demo=False):
        self.log = json.load(open(path))
        if show_demo:
            self.showGraph('avg_loss')
            self.showGraph('real_return')

    def addLog(self, name):
        self.log[name] = []

    def addItem(self, name, x):
        try:
            self.log[name].append(x)
        except:
            self.addLog(name)
            self.log[name].append(x)

    def addListItem(self, dic_item):
        for key, value in dic_item.items():
            self.addItem(key, value)

    def getLog(self, name):
        return self.log[name]

    def getLength(self):
        if len(self.getKeys()) > 0:
            sample_key = self.getKeys()[0]
            return len(self.log[sample_key])

    def getKeys(self):
        return np.array(list(self.log.keys()))

    def getCurrent(self, name):
        return self.log[name][-1]

    def writeToFile(self, file_path):
        json.dump(self.log, open(file_path, 'w'))

    def analysis(self, reward_threshold=500 - 120, n_part=10):
        interval = self.getLength() // n_part
        real_return = np.array(self.log['real_return'])
        start = np.array(self.log['start'])
        end = np.array(self.log['end'])
        time = np.array(self.log['time'])
        success = np.array(self.log['success'])

        success_start = start[success]
        success_end = end[success]
        success_time = time[success]

        # total success rate
        success_rate_total = np.sum(success) / len(success)
        print("success rate total is {rate}".format(rate=success_rate_total))

        # interval success rate
        n_loop = int(len(start) // interval + (len(start) % interval != 0))
        print(n_loop)
        interval_start = 0
        for i in range(n_loop):
            if i == 0:
                interval_start = 0
            else:
                interval_start = interval_start + interval
            interval_end = interval_start + interval

            interval_success = success[interval_start:interval_end]
            print("success rate in {i} interval {rate}".format(i=i,
                                                               rate=np.sum(interval_success) / len(interval_success)))

    def getGraph(self, aspect, interval=100, show_range=[0, 1]):
        if aspect not in self.getKeys():
            print("there's not {aspect}".format(aspect=aspect))
            return [], []

        Y = np.array(self.log[aspect]).copy()
        start = int(show_range[0] * len(Y))
        end = int(show_range[1] * len(Y)) + 1

        Y = Y[start:end]
        Y_smooth = smooth(Y, interval)
        x = np.linspace(0, len(Y_smooth), len(Y_smooth))

        return x, Y_smooth

    def showGraph(self, aspect, interval=100, show_range=[0, 1]):
        fig1 = plt.figure()
        ax1 = plt.axes()

        x, y = self.getGraph(aspect, interval=interval, show_range=show_range)
        ax1.plot(x, y, lw=2, label=aspect)
        ax1.set_title(aspect)

        ax1.legend()
        ax1.set_xlabel("epochs")
