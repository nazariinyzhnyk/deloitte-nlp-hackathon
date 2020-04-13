import os
from collections import Counter

import pandas as pd
import numpy as np

submissions_dir = 'e_selection'
submissions = os.listdir(submissions_dir)

print(submissions)

sbm_weigths = {
    'ensembe_sat_7101.csv': 7101,
    'maks_bert_knn_6582.csv': 6582,
    'maks_magic_6984.csv': 6984,
    'mykola_best_7300.csv': 7300,
    'mykola_best2_7340.csv': 7340,
    'submission_KNN_Myk.csv': 7000,
    'mykola_knn_bert_7042.csv': 7042,
    'mykola_super_overfit_rf_7178.csv': 7178,
    'nazar_all_features_log_regr_7133.csv': 7133,
    'mykola_two_level_rf_clean_text_7049.csv': 7049,
    'nazar_rf_all_features_6848.csv': 6848
}

targets = []
files = []
for submission in submissions:
    files.append(submission)
    sbmsn = pd.read_csv(os.path.join(submissions_dir, submission))
    targets.append(np.array(sbmsn['target']))
    ids = sbmsn['id']

final_targets = []
for j in range(len(targets[0])):
    trg = [0, 0, 0]
    for t in range(len(targets)):
        trg[targets[t][j]] += sbm_weigths[files[t]]
    final_targets.append(np.argmax(trg))

submission_df = pd.DataFrame(
            {'id': ids,
             'target': list(final_targets)}
        )
submission_df.to_csv('submission_ens.csv', index=False)


# # SIMPLE VOTING

# targets = []
# for submission in submissions:
#     sbmsn = pd.read_csv(os.path.join(submissions_dir, submission))
#     targets.append(np.array(sbmsn['target']))
#     ids = sbmsn['id']
#
# targets = np.array(targets)
#
#
# def apply_mode(col):
#     return [word for word, word_count in Counter(col).most_common(1)]
#
#
# final_targets = np.apply_along_axis(apply_mode, 0, targets)
#
# submission_df = pd.DataFrame(
#             {'id': ids,
#              'target': list(final_targets)[0]}
#         )
# submission_df.to_csv('submission_ens.csv', index=False)

print(1)
