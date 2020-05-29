import sklearn
from sklearn_pandas import DataFrameMapper
import numpy as np
import pandas as pd


def mapper_init(df):
    mapper = DataFrameMapper([

        ('手机型号', sklearn.preprocessing.LabelBinarizer()),
        ('education', sklearn.preprocessing.LabelBinarizer()),
        ('渠道', sklearn.preprocessing.LabelBinarizer()),
        ('身份证号', sklearn.preprocessing.LabelBinarizer()),

        ('好/坏（1：坏）', None),
        ('紧密联系人(通话记录次数最多的5个人)的最大逾期天数', None),
        ('年龄', None),
        ('成功借款次数', None),
        ('网龄(月)', None),
        ('L分', None),
        ('匹配到借款人近一年使用过的手机数量', None),
        ('匹配到借款人近一年使用过的手机号个数', None),
        ('用户有效联系号码(非170,171开头的手机号码)总数', None),
        ('详单通话条数', None),
        ('月平均话费', None),
        ('X拉黑', None),
        ('评分卡最终得分', None),
        ('3个月身份证关联家庭地址数', None),
        ('1个月内申请人在多个平台申请借款', None),
        ('3个月内申请人在多个平台申请借款', None),
        ('注册客户端', sklearn.preprocessing.LabelBinarizer()),

    ], input_df=True)

    np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)
    return mapper


def one_hot(df, mapper):
    data_tmp = np.round(mapper.transform(df.copy()).astype(np.double), 3)
    return pd.DataFrame(data_tmp, columns=mapper.transformed_names_)

