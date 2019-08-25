import sys
import os

# find filepath for data sets
if os.path.exists("../data"):
    fp = "../data/"
else:
    fp = "data/"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

class Data(object):
    def __init__(self):

        # pull in data, drop "day" column for now
        self.raw_data = pd.read_csv(fp + "Smarket.csv").drop("day", axis=1)
        self.raw_data.replace({"Direction": {"Up": 1, "Down": 0}}, inplace=True)
        #self.raw_features = self.raw_data.loc[:, self.raw_data.columns != "Direction"]
        self.raw_features = self.raw_data.drop(["Direction", "Today", "Year"], axis=1)
        self.raw_y = self.raw_data.loc[:, self.raw_data.columns == "Direction"]





def get_rss(y_train, y_pred):
    """"
        residual sum of squares for the prediction
    """
    rss = 0
    for i in range(len(y_train)):
        rss += (y_test[i][0] - y_pred[i][0]) ** 2

    return (rss / (len(y_test) - 2)) ** 0.5

def get_slopes_se(self, X, y_pred):
    rss = self.get_rss(y_pred)
    x_mean = np.mean(X)
    x_var = list(map(lambda x: (x - x_mean) ** 2, X))
    x_var = sum(x_var)

    return (rss ** 2 / x_var) ** 0.5

def coef_summary(X, y_pred, coef):
    b1_std_err = self.get_slopes_se(X, y_pred)

    t_statistic = coef / b1_std_err

    p_val = scipy.stats.norm.sf(abs(t_statistic))  # one-sided
    print(p_val)
    breakpoint()
    return b1_std_err, t_statistic, p_val

def sec_4_6_1(data):
    """
        describe data, make plot to show correlation between features
    """
    print(data.raw_data.describe())

    print(data.raw_features.corr())

    # parir plot
    # sns.pairplot(data.raw_features,diag_kind="hist")
    # plt.show()


def sec_4_6_2(data):
    """
        logistic regression model to predict Direction or stock market on any given day based
        on lag1-5 and volume (previous day's)
    """
    # breakpoint()

    # x_train, x_test, y_train, y_test = train_test_split(
    #     data.raw_features, data.raw_y, test_size=0.01, random_state=0
    # )

    logit = sm.Logit(data.raw_y,data.raw_features.loc[:,["Lag1", "Lag2"]])
    # fit the model
    result = logit.fit()
    print(result.summary())
    conf_matrix = result.pred_table(0.5)


    print("           down pred   up pred")
    print(f"down true {conf_matrix[0,0]}       {conf_matrix[0,1]}  ")
    print(f"up true   {conf_matrix[1,0]}        {conf_matrix[1,1]}")

    print(f"accuracy = {(conf_matrix[0,0]+conf_matrix[1,1])/1250}")


def sec_4_6_3(data):
    """
        using LDA to predict stock prices
    """
    lda = LinearDiscriminantAnalysis(solver="svd")

    lda.fit(data.raw_features, data.raw_y.values.ravel())
    breakpoint()



if __name__ == "__main__":
    data = Data()

    #sec_4_6_1(data)
    # sec_4_6_2(data)
    sec_4_6_3(data)
