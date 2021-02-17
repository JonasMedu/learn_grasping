"""
This was an attempt to show the connection between noise and the learning capabilities
"""

from performance_analysis.paper_graphs import NoiseGaph
from scipy.stats import linregress as lng
import numpy as np
from matplotlib import pyplot as plt


def get_regression_coef():
    noise_g = NoiseGaph()
    df = noise_g.noise_score_df.sort_values('num_trajectories')
    x = df['num_trajectories'].astype(float)
    x_log = np.log(df['num_trajectories'].astype(float))
    y = df['noise'].astype(float)
    ans_non_log = lng(x=x, y=y)
    ans_log = lng(x=x_log, y=y)
    print(df)
    plt.figure()
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, ans_non_log.intercept + ans_non_log.slope * x, 'r', label='fitted line')
    plt.title('ans_non_log')
    plt.legend()
    plt.figure()
    plt.plot(x_log, y, 'o', label='original data')
    plt.plot(x_log, ans_log.intercept + ans_log.slope * x, 'r', label='fitted line')
    plt.title('ans_log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    get_regression_coef()