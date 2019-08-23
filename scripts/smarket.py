import sys
import os
# find filepath for data sets
if os.path.exists("../data"):
    fp = "../data/"
else:
    fp = "data/"

import pandas as pd


class Data(object):

    def __init__(self):

        self.raw_data = pd.read_csv(fp+"Smarket.csv")

        #breakpoint()



def sec_4_6_1(data):
    """
        describe data
    """

    print(data.raw_data.describe())


if __name__ == "__main__":
    data = Data()


    sec_4_6_1(data)

