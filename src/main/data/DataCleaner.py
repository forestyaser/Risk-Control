import pandas

from main.utils.helpers import *
from main.utils.mapper import mapper_init, one_hot


class DataCleaner:

    def __init__(self):
        self.mapper = None

    features = [
        '好/坏（1：坏）',
        '注册客户端',
        '手机型号',
        '成功借款次数',
        '身份证号',
        'education',
        '紧密联系人(通话记录次数最多的5个人)的最大逾期天数',
        '渠道',
        '年龄',
        '网龄(月)',
        'L分',
        '匹配到借款人近一年使用过的手机数量',
        '匹配到借款人近一年使用过的手机号个数',
        '用户有效联系号码(非170,171开头的手机号码)总数',
        '详单通话条数',
        '月平均话费',
        'X拉黑',
        '评分卡最终得分',
        '3个月身份证关联家庭地址数',
        '1个月内申请人在多个平台申请借款',
        '3个月内申请人在多个平台申请借款']

    def generate_mapper_and_cleanend_training_data(self, training_data_path):
        df = self.clean_data(training_data_path)
        mapper = mapper_init(df)
        self.mapper = mapper
        df_limited_features = one_hot(df, mapper)
        return df_limited_features

    def clean_predict_data_path(self, path):
        df = pandas.read_csv(path, sep=',')
        return self.clean_predict_data(df)

    def clean_predict_data(self, df):
        predict_transformed_df = self.transform_data(df, set_mean=False)
        df_limited_features = one_hot(predict_transformed_df, self.mapper)
        return df_limited_features

    def clean_data(self, path_to_file):
        df = pandas.read_csv(path_to_file, sep=',')
        df_limited_features = self.transform_data(df)
        return df_limited_features

    def transform_data(self, df, set_mean=True):
        df_limited_features = df[DataCleaner.features]

        df_limited_features['注册客户端'] = df_limited_features['注册客户端'].apply(lambda x: unify_registed_client_end(x))
        df_limited_features['手机型号'] = df_limited_features['手机型号'].apply(lambda x: unify_phone_brand(x))
        df_limited_features['成功借款次数'] = df_limited_features['成功借款次数'].apply(lambda x: 0 if x <= 1 else 1)
        df_limited_features['sex'] = df_limited_features['身份证号'].apply(lambda x: int(x[-2]) % 2)
        df_limited_features['身份证号'] = df_limited_features['身份证号'].apply(lambda x: int(x[:2]))
        df_limited_features['渠道'] = df_limited_features['渠道'].fillna(value="OTHER")
        df_limited_features['education'] = df_limited_features['education'].apply(lambda x: unify_education(x))

        if set_mean:
            self.age_mean = df_limited_features['年龄'].mean()
            self.online_age_mean = df_limited_features['网龄(月)'].mean()
            self.L_score_mean = df_limited_features['L分'].mean()
            self.num_phone_used_in_last_year_mean = df_limited_features['匹配到借款人近一年使用过的手机数量'].mean()
            self.num_phone_num_used_mean = df_limited_features['匹配到借款人近一年使用过的手机号个数'].mean()
            self.num_useful_contact_mean = df_limited_features['用户有效联系号码(非170,171开头的手机号码)总数'].mean()
            self.num_close_bad_user = df_limited_features['紧密联系人(通话记录次数最多的5个人)的最大逾期天数'].mean()
            self.commute_time = df_limited_features['详单通话条数'].mean()
            self.commute_cost_per_month_mean = df_limited_features['月平均话费'].mean()
            self.evaluate_card_score_mean = df_limited_features['评分卡最终得分'].mean()
            self.three_month_add_change_mean = df_limited_features['3个月身份证关联家庭地址数'].mean()
            self.one_month_apply_avg = df_limited_features['1个月内申请人在多个平台申请借款'].mean()
            self.three_month_num_platform_avg = df_limited_features['3个月内申请人在多个平台申请借款'].mean()
        df_limited_features['年龄'] = df_limited_features['年龄'].fillna(value=self.age_mean)
        df_limited_features['年龄'] = df_limited_features['年龄'].apply(
            lambda x: int(x / 3) if x is not 0 else int(self.age_mean / 3))
        df_limited_features['网龄(月)'] = df_limited_features['网龄(月)'].fillna(value=self.online_age_mean)
        df_limited_features['网龄(月)'] = df_limited_features['网龄(月)'].apply(lambda x: int(x / 6))
        df_limited_features['L分'] = df_limited_features['L分'].fillna(value=self.L_score_mean)
        df_limited_features['L分'] = df_limited_features['L分'].apply(lambda x: int(x / 50))
        df_limited_features['匹配到借款人近一年使用过的手机数量'] = df_limited_features['匹配到借款人近一年使用过的手机数量'].fillna(
            value=self.num_phone_used_in_last_year_mean)
        df_limited_features['匹配到借款人近一年使用过的手机号个数'] = df_limited_features['匹配到借款人近一年使用过的手机号个数'].fillna(
            value=self.num_phone_num_used_mean)
        df_limited_features['用户有效联系号码(非170,171开头的手机号码)总数'] = df_limited_features['用户有效联系号码(非170,171开头的手机号码)总数'].fillna(
            value=self.num_useful_contact_mean)
        df_limited_features['用户有效联系号码(非170,171开头的手机号码)总数'] = df_limited_features['用户有效联系号码(非170,171开头的手机号码)总数'].apply(
            lambda x: int(x / 50))

        df_limited_features['紧密联系人(通话记录次数最多的5个人)的最大逾期天数'] = df_limited_features['紧密联系人(通话记录次数最多的5个人)的最大逾期天数'].fillna(
            value=self.num_close_bad_user)
        df_limited_features['紧密联系人(通话记录次数最多的5个人)的最大逾期天数'] = df_limited_features['紧密联系人(通话记录次数最多的5个人)的最大逾期天数'].apply(
            lambda x: x / 10)

        df_limited_features['详单通话条数'] = df_limited_features['详单通话条数'].fillna(value=self.commute_time)
        df_limited_features['详单通话条数'] = df_limited_features['详单通话条数'].apply(lambda x: int(x / 100) if x else 20)

        df_limited_features['月平均话费'] = df_limited_features['月平均话费'].fillna(value=self.commute_cost_per_month_mean)
        df_limited_features['月平均话费'] = df_limited_features['月平均话费'].apply(
            lambda x: int(x / 50) if x > 0 else int(self.commute_cost_per_month_mean / 20))

        df_limited_features['X拉黑'] = df_limited_features['X拉黑'].fillna(value=0)
        df_limited_features['X拉黑'] = df_limited_features['X拉黑'].apply(lambda x: 0 if x is 0 else 1)

        df_limited_features['评分卡最终得分'] = df_limited_features['评分卡最终得分'].fillna(value=self.evaluate_card_score_mean)
        df_limited_features['评分卡最终得分'] = df_limited_features['评分卡最终得分'].apply(lambda x: int(x / 2))

        df_limited_features['3个月身份证关联家庭地址数'] = df_limited_features['3个月身份证关联家庭地址数'].fillna(
            value=self.three_month_add_change_mean)
        df_limited_features['3个月身份证关联家庭地址数'] = df_limited_features['3个月身份证关联家庭地址数'].apply(lambda x: int(x / 1))

        df_limited_features['1个月内申请人在多个平台申请借款'] = df_limited_features['1个月内申请人在多个平台申请借款'].fillna(
            value=self.one_month_apply_avg)
        df_limited_features['1个月内申请人在多个平台申请借款'] = df_limited_features['1个月内申请人在多个平台申请借款'].apply(lambda x: int(x / 3))

        df_limited_features['3个月内申请人在多个平台申请借款'] = df_limited_features['3个月内申请人在多个平台申请借款'].fillna(
            value=self.three_month_num_platform_avg)
        df_limited_features['3个月内申请人在多个平台申请借款'] = df_limited_features['3个月内申请人在多个平台申请借款'].apply(lambda x: int(x / 3))
        return df_limited_features
