import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.model_selection import train_test_split, GridSearchCV
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn import metrics
import time

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.drop("customerID", axis=1, inplace=True)
data["TotalChareges"] = pd.to_numeric(data.TotalCharges, errors="coerce")
data.dropna(subset=["TotalChareges"], inplace=True)
data = data.reset_index().drop("index", axis=1)
data.SeniorCitizen = data.SeniorCitizen.astype("str")
data.Churn = data.Churn.map({"Yes": 1, "No": 0})
df = pd.get_dummies(data)

## KM 检测
# fig, ax = plt.subplots(figsize=(10, 8))
# kmf = KaplanMeierFitter()
# kmf.fit(data.tenure, event_observed=data.Churn)
# kmf.plot_survival_function(at_risk_counts=True, ax=ax)
# plt.show()

# obj_list = data.select_dtypes(include=["object"]).columns

# print(obj_list)

# data["MonthlyCharges_median"] = data["MonthlyCharges"].apply(
#     lambda x: "1" if x >= data["MonthlyCharges"].median() else "0")
# data["TotalCharges_median"] = data["TotalCharges"].apply(
#     lambda x: "1" if float(x) >= data["TotalCharges"].median() else "0")

# obj_list = data.select_dtypes(include=["object"]).columns
# fig, ax = plt.subplots(nrows=9, ncols=2, figsize=(24, 72))

# obj_list = obj_list.delete(-3)

## 各协变量检测
# for nrow in range(9):
#     for ncol in range(2):
#         feature = obj_list[nrow * 2 + ncol]
#         print(f"{nrow}, {ncol}, {feature}")
#         for i in data[feature].unique():
#             kmf = KaplanMeierFitter()
#             df_tmp = data.loc[data[feature] == i]
#             kmf.fit(df_tmp.tenure, event_observed=df_tmp.Churn, label=i)
#             kmf.plot_survival_function(ci_show=True, ax=ax[nrow, ncol])

#         p_value = multivariate_logrank_test(event_durations=data.tenure,
#                                             groups=data[feature],
#                                             event_observed=data.Churn).p_value
#         p_value_text = [
#             "p-value < 0.001" if p_value < 0.001 else "p-value = %.4F" %
#             p_value
#         ][0]
#         ax[nrow][ncol].set_title(
#             f"survival curves of {feature} \n logrank test: {p_value_text}")

# plt.show()
