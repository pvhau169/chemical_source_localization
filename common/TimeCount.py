import time


class TimeCount:
    def __init__(self):
        super(TimeCount, self).__init__()
        self.current_time = time.time()
        self.time = 0

    def reset(self):
        self.current_time = time.time()

    def getTime(self):
        return self.time

    def count(self):
        self.time = float(time.time() - self.current_time)
        return self.time

    def printCount(self):
        print(self.count())
