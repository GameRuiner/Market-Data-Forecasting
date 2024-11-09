import lightgbm as lgb
import pandas as pd

import numpy as np

import glob
import pickle 

def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                int8 = np.iinfo(np.int8)
                int16 = np.iinfo(np.int16)
                int32 = np.iinfo(np.int32)
                if c_min > int8.min and c_max < int8.max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > int16.min and c_max < int16.max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > int32.min and c_max < int32.max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                float16 = np.finfo(np.float16)
                float32 = np.finfo(np.float32)
                if c_min > float16.min and c_max < float16.max and not float16_as32:
                    df[col] = df[col].astype(np.float16)         
                elif c_min > float32.min and c_max < float32.max:
                    df[col] = df[col].astype(np.float32)     
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

def get_X_y_w(df, dates):
    X = df[feature_names].loc[df['date_id'].isin(dates)]
    y = df['responder_6'].loc[df['date_id'].isin(dates)]
    w = df['weight'].loc[df['date_id'].isin(dates)]
    return X,y,w


def r2_lgb(y_true, y_pred, sample_weight):
    e = 1e-38
    rss = np.average((y_pred - y_true) ** 2, weights=sample_weight)
    tss = np.average(y_true ** 2, weights=sample_weight)
    r2 = 1 - rss/tss + e
    return 'r2', r2, True

if __name__ == "__main__":
    input_path = './input'

    feature_names = [f"feature_{i:02d}" for i in range(79)]
    partitions = glob.glob(f"{input_path}/train.parquet/*/*.parquet")
    partitions.sort()
    partitions_to_read = 2
    df = pd.concat([pd.read_parquet(partitions[i]) for i in range(partitions_to_read)], ignore_index=True)
    reduce_mem_usage(df, False)

    df = df.fillna(0)
    columns_with_missing_values = df.columns[df.isnull().any()]

    for feature in feature_names:
        lower_bound = df[feature].quantile(0.05)
        upper_bound = df[feature].quantile(0.95)
        df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
        df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])
    
    dates = df['date_id'].unique()
    n = len(dates)
    train_end = int(n * 0.8)
    train_dates = dates[:train_end]
    val_dates = dates[train_end:]
    
    X_train, y_train, w_train = get_X_y_w(df, train_dates)
    X_valid, y_valid, w_valid = get_X_y_w(df, val_dates)

    model = lgb.LGBMRegressor(n_estimators=500, objective='l2', gpu_use_dp=True)
    model.fit(X_train, y_train, w_train,
            eval_metric=[r2_lgb],
            eval_set=[(X_valid, y_valid, w_valid)],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(10)
            ]         
    )
    file = open("./web_service/final_model.bin", "wb")
    pickle.dump(model, file)
