import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    d1 = pd.read_csv('../data/csv/18-05-01.csv')
    d2 = pd.read_csv('../data/csv/18-05-02.csv')
    d3 = pd.read_csv('../data/csv/18-05-03.csv')
    d4 = pd.read_csv('../data/csv/18-05-04.csv')
    d5 = pd.read_csv('../data/csv/18-05-05.csv')
    d6 = pd.read_csv('../data/csv/18-05-06.csv')

    merged = d1.append(d2).append(d3).append(d4).append(d5).append(d6)

    merged = merged[merged.light < 1000]
    merged = merged[merged.sound < 750]

    averaged = merged.groupby(np.arange(len(merged))//60).mean()
    norm = normalize(averaged, axis=0)
    norm = pd.DataFrame(norm)
    norm['datetime'] = merged['datetime'].values[::60]

    norm.to_csv(path_or_buf='../data/csv/hourly_averages_normalized.csv', index=False)

    d1 = d1[d1.light < 1000]
    d1 = d1[d1.sound < 750]
    d2 = d2[d2.light < 1000]
    d2 = d2[d2.sound < 750]
    d3 = d3[d3.light < 1000]
    d3 = d3[d3.sound < 750]
    d4 = d4[d4.light < 1000]
    d4 = d4[d4.sound < 750]
    d5 = d5[d5.light < 1000]
    d5 = d5[d5.sound < 750]
    d6 = d6[d6.light < 1000]
    d6 = d6[d6.sound < 750]
    d1.to_csv(path_or_buf='../data/csv/18-05-01-filtered.csv', index=False)
    d2.to_csv(path_or_buf='../data/csv/18-05-02-filtered.csv', index=False)
    d3.to_csv(path_or_buf='../data/csv/18-05-03-filtered.csv', index=False)
    d4.to_csv(path_or_buf='../data/csv/18-05-04-filtered.csv', index=False)
    d5.to_csv(path_or_buf='../data/csv/18-05-05-filtered.csv', index=False)
    d6.to_csv(path_or_buf='../data/csv/18-05-06-filtered.csv', index=False)

