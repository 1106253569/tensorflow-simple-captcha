import numpy as np
from PIL import Image  
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 336])
    W = tf.Variable(tf.zeros([336, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    save_path = "model/model"

    saver.restore(sess, save_path)

    correct_prediction = tf.argmax(y, 1)

    while True:
        print('---------')
        file_path = input('图片路径: ')
        img = Image.open(file_path).convert('L')  # 读取图片并灰度化
        print('---------')

        img = img.crop((2, 1, 66, 22))  # 裁掉边变成 64x21

        # 分离数字
        img1 = img.crop((0, 0, 16, 21))
        img2 = img.crop((16, 0, 32, 21))
        img3 = img.crop((32, 0, 48, 21))
        img4 = img.crop((48, 0, 64, 21))

        img1 = np.array(img1).flatten()
        img1 = list(map(lambda x: 1 if x <= 180 else 0, img1))
        img2 = np.array(img2).flatten()
        img2 = list(map(lambda x: 1 if x <= 180 else 0, img2))
        img3 = np.array(img3).flatten()
        img3 = list(map(lambda x: 1 if x <= 180 else 0, img3))
        img4 = np.array(img4).flatten()
        img4 = list(map(lambda x: 1 if x <= 180 else 0, img4))

        result = sess.run(correct_prediction, feed_dict={
            x: [img1, img2, img3, img4]})

        print('---------')
        print("'{}' 识别结果为: ".format(file_path), end='')
        for num in result:
            print(num, end='')
        print('')
        print('---------')

        yn = input("是否继续(Y/N): ")
        if yn != "y" and yn != "Y":
            print("正在退出......")
            break

    sess.close()
