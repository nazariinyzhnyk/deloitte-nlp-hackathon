#!/usr/bin/env bash

FILEDEST="submission.csv"
TIMESTAMP=$(date +%s)
line=$(head -n 1 "score.txt")

SAVETO="all_submissions/submission_at_($TIMESTAMP)_score_($line).csv"

head -n 10 ${FILEDEST}
echo "Submitting file: $FILEDEST. Current timestamp: $TIMESTAMP"

cp ${FILEDEST} $SAVETO
echo "$FILEDEST was copied to $SAVETO"

kaggle competitions submit -c company-acceptance-prediction -f ${FILEDEST} -m "Submit at $TIMESTAMP"
