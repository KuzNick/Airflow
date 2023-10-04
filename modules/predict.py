import json
import os
from datetime import datetime
import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')
def predict():
    models_list = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{models_list[-1]}', 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    files_list = os.listdir(f'{path}/data/test')
    for name in files_list:
        with open(f'{path}/data/test/{name}') as file:
            form = json.load(file)
        data = pd.DataFrame.from_dict([form])
        prediction = model.predict(data)
        dict_pred = {'car_id': data.id, 'pred': prediction}
        df = pd.DataFrame(dict_pred)
        df_pred = pd.concat([df_pred, df], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

    pass


if __name__ == '__main__':
    predict()
