import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')

train_df = pd.read_csv('pubmed_train.csv')
train_df = train_df[['text', 'label']]

test_df = pd.read_csv('pubmed_test.csv')
test_df = test_df[['text', 'label']]


def run_test(lm):
    model = ClassificationModel('distilbert', lm, num_labels=5, args={'overwrite_output_dir': True, 'fp16': False, 'save_steps': 100000000})
    model.train_model(train_df)
    result, model_outputs, wrong_preds = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
    return result


PATH = '/'
f = open('list_of_models.txt') # this is a list of model paths/names
t = [PATH + s.strip() for s in f.readlines()]
f.close()

for lm in t:
    l = []
    for i in range(3):
        l.append(run_test(lm)['acc'])

    f = open('results.txt', 'a+')
    f.write('{}: {:0.2f}, {:0.2f}\n'.format(lm, np.mean(l)*100, np.std(l)*100))
    f.close()
