import numpy as np
from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')

train_df = pd.read_csv('agnews_train.csv', header=None, names=['label', 'title', 'body'], skiprows=[0])
train_df['text'] = train_df['body'] + ' ' + train_df['title']
train_df = train_df[['text', 'label']]
train_df['label'] = train_df['label'].apply(lambda x:x-1)

test_df = pd.read_csv('agnews_test.csv', header=None, names=['label', 'title', 'body'], skiprows=[0])
test_df['text'] = test_df['body'] + ' ' + test_df['title']
test_df = test_df[['text', 'label']]
test_df['label'] = test_df['label'].apply(lambda x:x-1)

def run_test(lm):
    model = ClassificationModel('distilbert', lm, num_labels=4,
                                args={'overwrite_output_dir': True, 'fp16': False,
                                'save_steps':-1})

    # Train the model
    model.train_model(train_df)
    result, model_outputs, predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)

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
