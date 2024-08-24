from matplotlib import pyplot as plt
def plot_data_distribution(num_train: int, num_val: int, num_test: int):
    """Plot number of data for train-set, val-set, and test-set after splitted"""

    # Create the bar chart
    bars = plt.bar(["Train", "Val", "Test"],
            [num_train, num_val, num_test], align='center', color=['green', 'red', 'blue'])

    # Add the data value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center', va='bottom')

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    plt.show()
