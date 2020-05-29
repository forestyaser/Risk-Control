import pandas

from flask import Flask, request, abort

from main.data.DataCleaner import DataCleaner
from main.model.RiskControlModel import RiskControlModel
from main.utils.helpers import transform_request_to_df, generate_X_y_from_df
import logging

app = Flask(__name__)

data_cleaner = DataCleaner()
training_data = data_cleaner.generate_mapper_and_cleanend_training_data(
    '/var/qindom/riskcontrol/data/risk_all_label_data.csv')
X, y = generate_X_y_from_df(training_data)
risk_control_model = RiskControlModel(threshold=0.45)
risk_control_model.fit(X, y)
logging.basicConfig(filename='/var/qindom/riskcontrol/log/myapp.log', level=logging.INFO)


@app.route("/predict", methods=['post'])
def predict():
    logging.info(request.get_json())
    try:
        df = transform_request_to_df(request)
    except Exception:
        abort(400, 'Bad request, input data cannot be found or data is not json array list')
    try:
        df_limited_features = data_cleaner.clean_predict_data(df)
    except Exception as e:
        print(e)
        abort(400, 'predict data does not match training data')
    try:
        X_test, y_test = generate_X_y_from_df(df_limited_features)
    except Exception:
        abort(400, 'Dummy y label is required')
    try:
        result = risk_control_model.predict(X_test)
    except Exception as e:
        logging.error(e)
        abort(500, 'Model exception')
    result_df = pandas.DataFrame()
    result_df['user_id'] = df['用户id']
    result_df['predict'] = result
    logging.info(result_df.to_json(orient='records'))
    return result_df.to_json(orient='records')
