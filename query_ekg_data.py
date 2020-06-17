"""Cristi: Real period is important. The cut tech Cristi used works!
Xin: Tried use offset, not good idea.
"""


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


def query_data(p1=True, p2=True):
    # the original six signals are of length 70.
    # ekg2 is patient007,  ekg1 is patient024

    # deleted last three of patient2
    t_p1, t_p2, patient1, patient2 = None, None, None, None
    if p1:
        patient1, fields = wfdb.rdsamp('s0026lre', pn_dir='ptbdb/patient007', channels=[11], sampfrom=4500, sampto=None)
        # v5
        t_p1 = create_time_list(0.01, len(patient1))


        print(fields)
        print(len(patient1))

        patient1 = patient1.reshape(len(patient1), 1)[:, 0]

        plt.plot(t_p1, patient1)

    if p2:
        patient2, fields = wfdb.rdsamp('s0083lre', pn_dir='ptbdb/patient024', channels=[11], sampfrom=4500, sampto=None)
        t_p2 = create_time_list(0.01, len(patient2))
        print(fields)
        print(len(patient2))

        patient2 = patient2.reshape(len(patient1), 1)[:, 0]

        plt.plot(t_p2, patient2)
    plt.show()

    return t_p1, patient1, t_p2, patient2


def cut_data_patient1(t_p1, patient1):
    step = 833
    start = 50
    offset = 0
    index = 4
    for i in range(120):  # todo: change this to while loop
        slice1 = start + step + offset
        if slice1 >= len(patient1):
            break
        seg1 = patient1[start:slice1]
        t_seg1 = t_p1[start:slice1]

        plt.plot(t_seg1, seg1)
        plt.show()

        save_slice(t_seg1, seg1, index, patient_index=2) # note, the number is reverse by historical problem.
        index += 1
        start = slice1


def cut_data_patient2(t_p2, patient2):
    step = 950
    start = 300
    offset = 0
    index = 4
    for i in range(120):  # todo: change is to while loop
        slice1 = start + step - offset
        if slice1 >= len(patient2):
            break
        seg1 = patient2[start:slice1]
        t_seg1 = t_p2[start:slice1]

        plt.plot(t_seg1, seg1)
        plt.show()

        save_slice(t_seg1, seg1, index, patient_index=1)
        index += 1
        start = slice1


def save_slice(t, signal, index, patient_index):
    folder = './ekg_data/ekg_more_data_2'
    seg = {'Time': [], 'Value': []}
    seg['Time'] = t
    seg['Value'] = signal
    df = pd.DataFrame.from_dict(seg)
    filename = os.path.join(folder, 'ekg{}_{}.csv'.format(patient_index, index))
    df.to_csv(filename)


def plot_traces():
    for i in range(4, 14):
        file_name = "ekg_data/ekg_more_data_2/ekg1_{}.csv".format(i)
        raw_trace = pd.read_csv(file_name, sep=r'\s*,\s*', nrows=None, engine='python')  # the cut was 834
        x = raw_trace["Time"].values
        y = raw_trace["Value"].values
        plt.figure()
        plt.plot(x, y)
        plt.show()

def main():
    t_p1, patient1, t_p2, patient2 = query_data(p1=True, p2=False)  # T, F controls which patient to query

    # plot_one_trace()
    cut_data_patient1(t_p1, patient1)
    # cut_data_patient2(t_p2, patient2)
    # plot_traces()

if __name__ == '__main__':
    main()