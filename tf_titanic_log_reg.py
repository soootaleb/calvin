import pandas as pd
import tensorflow as tf


def input_fn(data_file, num_epochs, shuffle):
    data_file = pd.read_csv(tf.gfile.Open(data_file), names=[
        'PassengerId',
        'Survived',
        'Pclass',
        'Name',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Ticket',
        'Fare',
        'Cabin',
        'Embarked'
    ], skiprows=1).dropna(how='any', axis=0)
    return tf.estimator.inputs.pandas_input_fn(
        x=data_file,
        y=data_file['Survived'],
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1
    )


sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])
pclass = tf.feature_column.categorical_column_with_vocabulary_list('Pclass', [1, 2, 3])
sibsp = tf.feature_column.numeric_column('SibSp')
parch = tf.feature_column.numeric_column('Parch')
age = tf.feature_column.numeric_column('Age')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[16, 30, 40, 60])

model = tf.estimator.LinearClassifier(model_dir='./tmp/titanic_log_reg', feature_columns=[sex, pclass, sibsp, parch, age])

model.train(input_fn=input_fn('data/titanic/train.csv', None, True), steps=100)

results = model.evaluate(input_fn=input_fn('data/titanic/test.csv', 1, False), steps=None)

for key in sorted(results):
    print('%s: %s' % (key, results[key]))