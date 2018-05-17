import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_examples, n_timesteps, predict, freq=None):
    """
    :param n_examples: Number of Examples
    :param n_timesteps: Number of n_timesteps for LSTM
    :param predict: Window size to be predicted
    :param random_freq: Uses random frequency if true
    :param freq: Frequency value if non random frquency to be used
    :return: sample and predicted data along with their times
    """

    A = 1  # Peak amplitude
    sampling_freq = 100

    x = np.empty((n_examples, n_timesteps))
    y = np.empty((n_examples, predict))
    time_x = np.empty((n_examples, n_timesteps))
    time_y = np.empty((n_examples, predict))
    
    for i in range(n_examples):
        
        # If this condition is met, allows generation of random frequency data for test set
        if freq is None:
            freq = np.random.rand() * 3
        # Else train on specific frequency data
        else:
            freq = freq

        t = np.random.rand() * 2 * np.pi
    
        time_x[i,:] = np.arange(0, n_timesteps) / sampling_freq
        time_y[i,:] = np.arange(n_timesteps, n_timesteps+predict) / sampling_freq
    
        x[i,:] = A*np.sin(2*np.pi*freq*(time_x[i,:] + t))
        y[i,:] = A*np.sin(2*np.pi*freq*(time_y[i,:] + t))

        y = np.float32(y)
        x = np.float32(x)

    return time_x, x, time_y, y


if __name__ == "__main__":
    n_examples = 50
    n_timesteps = 100
    predict = 50
    time_x, x, time_y, y = generate_data(n_examples, n_timesteps, predict)

    for i in range(3):
        plt.plot(time_x[i, :], x[i, :])
        plt.plot(time_y[i, :], y[i, :])
        plt.show()

