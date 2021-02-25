import csv
import matplotlib.pyplot as plt


def write2csv(file, data):
    with open('{}.csv'.format(file), mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def np2csv(file, np_arr):
    with open('{}.csv'.format(file), mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(np_arr)

def plot_reward(file_name, total_rewards):
    plt.figure(figsize=(15, 3))

    epoch = len(total_rewards)
    plt.plot(epoch, total_rewards)
    plt.plot(epoch, avg_rewards)
    plt.savefig('fig/{}.png'.format(file_name))
    plt.clf()


if __name__ == '__main__':
    write2csv('text', [['John Smith', 'Accounting', 'November'],
                        ['John Smith', 'Accounting', '2']])
