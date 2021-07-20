from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt

import shutil

import Algorithms

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

#training_df = Algorithms.main()
#testing_df = Algorithms.getTest()
#features = ['ToGo', 'Down']
'''
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Algorithms.main()#tf.cast(training_df[features].values, tf.int32) pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval =  pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')#Algorithms.getTest()#tf.cast(testing_df[features].values, tf.int32)  #pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')#tf.cast(training_df['Success'].values, tf.int32)
y_eval = dfeval.pop('survived')#tf.cast(testing_df['Success'].values, tf.int32)
'''

dftrain = Algorithms.main() #pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = Algorithms.getTest() #pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#y_train = dftrain.pop('Success')
#y_eval = dfeval.pop('Success')

print(dftrain)
#print(y_train)



'''
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare'] #['ToGo', 'Down', 'YardLine']
'''

fc = tf.feature_column
CATEGORICAL_COLUMNS = []#['sex', 'n_siblings_spouses', 'parch', 'class', 'deck','embark_town', 'alone']
NUMERIC_COLUMNS = ['Quarter', 'Time', 'ToGo', 'Down', 'YardLine', 'PType'] #['age', 'fare']


def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))


def predict(dfeval, importedModel):
    colNames = dfeval.columns
    dtypes = dfeval.dtypes
    predictions = []
    for row in dfeval.iterrows():
        example = tf.train.Example()
        for i in range(len(colNames)):
            dtype = dtypes[i]
            colName = colNames[i]
            value = row[1][colName]
            if dtype == "object":
                value = bytes(value, "utf-8")
                example.features.feature[colName].bytes_list.value.extend(
                    [value])
            elif dtype == "float":
                example.features.feature[colName].float_list.value.extend(
                    [value])
            elif dtype == "int":
                example.features.feature[colName].int64_list.value.extend(
                    [value])

        predictions.append(
            importedModel.signatures["predict"](
                examples=tf.constant([example.SerializeToString()])
            )
        )

    return predictions


feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))


for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
#print('Feature value: "{}"'.format(example['class'].iloc[0]))
#print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

NUM_EXAMPLES = len(dftrain)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

def make_train_input_fn(df, num_epochs):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = df,
    y = df["Success"],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )

def make_prediction_input_fn(df):
  return tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )


#train_input_fn = make_input_fn(dftrain, y_train)
#eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=10)

print(feature_columns)

feature_columns2 = tf.feature_column.numeric_column('ToGo', dtype=tf.float32)
linear_est = tf.estimator.LinearClassifier(feature_columns)

dir = os.listdir("modelsAndCheckpoints")

if not os.path.exists("modelsAndCheckpoints/linear_est"):
    linear_est.train(make_train_input_fn(dftrain, num_epochs=10))
    #linear_est.train(train_input_fn, max_steps=100)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec(feature_columns))

    OUTDIR = 'modelsAndCheckpoints'
    shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each
    timemodelBasePath = os.path.join(OUTDIR, "linear_est")
    modelPath = linear_est.export_saved_model(timemodelBasePath, serving_input_fn)
    print(modelPath)
    print(dfeval.columns)

    savedModelPath = modelPath
    importedModel = tf.saved_model.load(savedModelPath)

    dfeval.drop(columns=["Success"], inplace=True)
    predictions = predict(dfeval, importedModel)
    #linear_est = tf.saved_model.load(savedModelPath)
    #export_dir = 'modelsAndCheckpoints'
    #subdirs = [x for x in Path(export_dir).iterdir()
    #           if x.is_dir() and 'temp' not in str(x)]
    #latest = str(sorted(subdirs)[-1])
    #linear_est = keras.models.load_model("modelsAndCheckpoints/")
    #linear_est = tf.keras.models.load_model(latest)#tf.saved_model.load(latest)
    #linear_est = tf.keras.models.load_model("modelsAndCheckpoints/1625790552/saved_model.pb")
    newPreds = []
    for pred in predictions[:10]:
        # change 'probabilities' with 'predictions' in case
        # of regression model.
        newPreds.append(np.argmax(pred["probabilities"]))
    print(newPreds)


'''
predDicts = list(linear_est.predict(make_prediction_input_fn(dfeval)))
preds = []
for pred in predDicts[:10]:
    preds.append(np.argmax(pred["probabilities"]))
print(preds)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

#linear_est.export_saved_model('saved_model', train_input_fn)

# Evaluation.
result1 = linear_est.evaluate(eval_input_fn)

#print(linear_est)

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)
# The model will stop training once the specified number of trees is built, not
# based on the number of steps.


est.train(train_input_fn, max_steps=100)
#logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)





train_hooks_list = [logging_hook] # implemented by yourself
train_spec = tf.estimator.TrainSpec(train_input_fn, hooks=train_hooks_list)  # implemented by yourself
eval_spec = tf.estimator.train_and_evaluate(est, train_spec)


# Eval.
#result = est.evaluate(eval_input_fn)
result = list(est.predict(input_fn=eval_input_fn))
result2 = est.evaluate(eval_input_fn)


#train_result = est.evaluate(train_input_fn)

#print(dfeval)

#for p in result:
#    print(p)

print(pd.Series(result1))
print(pd.Series(result2))
#print(result)

plt.plot(10, pd.Series(result2)[7])
plt.show()

#pred_dicts = list(est.predict(eval_input_fn))
#suc = pd.Series([pred['Success'][1] for pred in pred_dicts])

#suc.plot(kind='hist', bins =20, title='predicted success')
#plt.show()

#print(train_result)

'''