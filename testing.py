from baseline import *
import os

binary_questions = os.path.expanduser("~/Desktop/data_sub/2_questions.txt")
path = os.path.expanduser("~/Desktop/data/")

print(baseline_sum_by_question_no(binary_questions, path))

# Output
# [allmb, allwb, nomb, nowb, textmb, textwb, allmb2, allwb2, nomb2, nowb2, textmb2, textwb2]
# [all majority daily baseline, all weighted daily baseline, no text majority daily baseline, no text weighted daily baseline, text majority daily baseline, text weighted daily baseline,
# all majority daily baseline, all weighted daily baseline, no text majority daily baseline, no text weighted daily baseline, text majority daily baseline, text weighted daily baseline]
# [743.06232418 751.12440409 741.28797043 749.08179571 695.86574925
#  709.03089006 743.17306005 750.98926495 741.37500947 749.38742637
#  697.16840384 710.43812746]

