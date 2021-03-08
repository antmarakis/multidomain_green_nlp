from simpletransformers.classification import ClassificationModel
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')

torch.cuda.set_device(0)

train_df = pd.read_csv('hyperpartisan_train.csv')
test_df = pd.read_csv('hyperpartisan_test.csv')

def run_test(lm):
    model = ClassificationModel('distilbert', lm, num_labels=2, args={'num_train_epochs': 2, 'overwrite_output_dir': True, 'fp16': False, 'save_steps':100000})
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
