from tensorflow.keras.models import load_model

import functions


X = [
    ' 20+ 3+3',
    '3- 1 - 5  ',
    '22 +11 - 2 ',
    ' 45 -35+ 5  ',
    ' 2+20-2 '
]

y = [str(abs(eval(x))) for x in X]  # calculate y as str
max_len = max(list(map(len, y)))
y = list(map(lambda x: format(x, f">{max_len}"), y))  # set y lengthes to max_len

model = load_model('checkpoint.h5')
y_pred = functions.predict(X, model)

print('The result of translating phrases (real-y, pred-y): ')
print(list(zip(list(map(int, y)), list(map(int, y_pred)))))
