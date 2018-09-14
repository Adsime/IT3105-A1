import matplotlib.pyplot as plt

def visualize(data):
    with plt.style.context('seaborn-colorblind'):
        print(plt.style.available)
        labels = []
        for set in data:
            x = data[set][0]
            y = data[set][1]
            plt.plot(x, y, label=set)
            labels.append(set)
        plt.legend(labels)
        plt.title('Stuff and things')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()