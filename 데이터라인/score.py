import sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import permutations 
NUM_RANKING = 3

def calculate_weighted_kendall_tau(pred, label, rnk_lst, num_rnk):
    """
    calcuate Weighted Kendall Tau Correlation
    """
    total_count = 0
    total_corr = 0
    for i in range(len(label)):
        # weighted-kendall-tau는 순위가 높을수록 큰 숫자를 갖게끔 되어있기 때문에 
        # 순위 인덱스를 반대로 변경해서 계산함 (1위 → 가장 큰 숫자)
        corr, _ = stats.weightedtau(num_rnk-1-rnk_lst[label[i]],
                                    num_rnk-1-rnk_lst[pred[i]])
        total_corr += corr
        total_count += 1
    return (total_corr / total_count)      

y_true = pd.read_csv(sys.argv[1], header=None, encoding='utf8').to_numpy()
y_pred = pd.read_csv(sys.argv[2], header=None, encoding='utf8').to_numpy()

rnk_lst = np.array(list(permutations(np.arange(NUM_RANKING), NUM_RANKING)))

# get scores
score = np.round(calculate_weighted_kendall_tau(y_pred, y_true, rnk_lst, NUM_RANKING), 7)

print("score:", score)