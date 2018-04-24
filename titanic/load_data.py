import pandas as pd


class DataSet():
    def __init__(self):
        self.titanic_ds = pd.DataFrame()
        #self.test = pd.DataFrame()
        self.full_ds = pd.DataFrame()

    def get_titanic(self):
        return self.titanic_ds
    def get_full(self):
        return self.full_ds

    def load_files(self, traindata, testdata):

        train = pd.read_csv(traindata)
        test = pd.read_csv(testdata)
        self.full_ds = train.append(test, ignore_index=True)
        self.titanic_ds = self.full_ds[:891]