import wfdb


def query_data():
    d, fields = wfdb.rdsamp('s0026lre', pn_dir='ptbdb/patient007', channels=[11], sampfrom=0, sampto=10000)  # v5
    print(fields)

    signals, fields = wfdb.rdsamp('s0083lre', pn_dir='ptbdb/patient024', channels=[11], sampfrom=0, sampto=4500)
    print(fields)

def main():
    query_data()

if __name__ == '__main__':
    main()