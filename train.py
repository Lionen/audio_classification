import argparse
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
import process

parser = argparse.ArgumentParser(description='Read file and process audio')
parser.add_argument('label_path', type=str, help='File to read and process')
parser.add_argument('--model_path', type=str, default='model.h5', help='File to save the model')

args = parser.parse_args()
label_path = args.label_path
model_path = args.model_path

batch_size = 128
epochs = 10

# 获取数据集
x, y = process.preprocessing(label_path)
# 随机抽取20%的测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

feature_size = x_train.shape[1]
classes_num = y_train.shape[1]

# 构建模型
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(feature_size,), name='layer_1'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1,
          shuffle=True)

# 模型评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(model.predict_classes(x_test))
# 模型保存
model.save(model_path)
