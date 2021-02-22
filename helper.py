import csv


def write2csv(file, data):
    with open('{}.csv'.format(file), mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def np2csv(file, np_arr):
    with open('{}.csv'.format(file), mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(np_arr)


if __name__ == '__main__':
    write2csv('text', [['John Smith', 'Accounting', 'November'],
                        ['John Smith', 'Accounting', '2']])
