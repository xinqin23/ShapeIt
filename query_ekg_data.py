import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_time_list(step_size, len):
    li = []
    counter = 0.0
    t = 0.0
    while counter < len:
        li.append(t)
        t += step_size
        counter += 1.0

    time_list = np.asarray(li)
    return time_list


def plot_one_trace():
    file_name = "ekg_data/ekg2_1.csv"
    raw_trace = pd.read_csv(file_name, sep=r'\s*,\s*', nrows=None, engine='python') # the cut was 834
    x = raw_trace["Time"].values
    y = raw_trace["Value"].values
    plt.plot(x, y)
    plt.show()


def query_data():
    # the original six signals are of length 70.
    # ekg2 is patient007,  ekg1 is patient024

    patient1, fields = wfdb.rdsamp('s0026lre', pn_dir='ptbdb/patient007', channels=[11], sampfrom=4500, sampto=8000)
    # v5
    t_p1 = create_time_list(0.01, len(patient1))


    print(fields)
    print(len(patient1))

    patient2, fields = wfdb.rdsamp('s0083lre', pn_dir='ptbdb/patient024', channels=[11], sampfrom=4500, sampto=8000)
    t_p2 = create_time_list(0.01, len(patient2))
    print(fields)
    print(len(patient2))

    patient1 = patient1.reshape(len(patient1), 1)[:, 0]

    plt.plot(t_p1, patient1)
    plt.plot(t_p2, patient2)
    plt.show()

    return t_p1, patient1, t_p2, patient2


def cut_data():
    t_p1, patient1, t_p2, patient2 = query_data()


    step = 833
    start = 50

    # brute force writing, thus by the way check the plot for each slice
    #

    slice1_1 = start + step

    start = slice1_1
    slice1_2 = slice1_1 + step

    start = slice1_2
    slice1 = slice1_2 + step

    seg1 = patient1[start:slice1]
    t_seg1 = t_p1[start:slice1]

    plt.plot(t_seg1, seg1)
    plt.show()

    save_slice(t_seg1, seg1)


def save_slice(t, signal):
    folder = './ekg_data/ekg_more_data'
    seg = {'Time': [], 'Value': []}
    seg['Time'] = t
    seg['Value'] = signal
    df = pd.DataFrame.from_dict(seg)
    filename = os.path.join(folder, 'ekg2_7.csv')
    df.to_csv(filename)


def main():
    # query_data()
    # plot_one_trace()
    cut_data()

if __name__ == '__main__':
    main()