import sys
import os



import pandas as pd

class Data(object):


    def __init__(self, fn = "Advertising.csv"):

        # find path to data files, either called from top-level or within scripts
        if os.path.exists("../data/"+fn):
            data_path = "../data/"
        elif os.path.exists("data/"+fn):
            data_path = "data/"
        else:# fn probably includes a path
            data_path=""

        self.file = data_path+fn
        self.raw_data = self.load_data()


    def load_data(self):
        data = pd.read_csv(self.file)

        return data

    def xy_split(self, df):
        breakpoint()
        X = df.loc[:, df.columns!="Sales"]

        y = df.loc[:, df.columns=="Sales"]


        return X, y


if __name__ == "__main__":

    data_1 = Data()
    X, y = data_1.xy_split(data_1.raw_data)
