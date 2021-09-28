# CNNを用いたVAE
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py

# ======================================================
# システム設定（学習には影響ない設定）
# ======================================================
flag_display2dMAP = 0  # 二次元マップを表示させるかどうか
data_folder_name = 'data_output'  # モデル、重み、パラメーターなどを保存するフォルダ名

# ======================================================
# 学習パラメータ設定
# ======================================================
# 学習対象フォルダ名
list_folder_name = data_training

# 学習データの画像読み込みサイズ
# img_target_size = 64

# network parameters
# epochs = 3000
# batch_size = 64
# filters = 32

# conv2Dの繰り返し回数
conv2D_repeat_num = 3

# =====================================================
# ファイル名につけるための現在時刻（日本時間）を得る
# =====================================================
import datetime
import os

# 日本時間のオフセット
offset = datetime.timedelta(hours=+9)
jst = datetime.timezone(offset)

# 現在時刻を文字列化
datetime_now_tz = datetime.datetime.now(tz=jst)
filename_time = datetime_now_tz.strftime('d%y%m%d_%H%M')

# モデルと重みのファイル名に現在時刻を追加
model_file_name = filename_time + 'model_vae.json'
weights_file_name = filename_time + 'weights_vae.h5'

# 保存先のフォルダを連結
model_file_name = os.path.join(data_folder_name, model_file_name)
weights_file_name = os.path.join(data_folder_name, weights_file_name)

# =====================================================
# パラメーターをJSON形式で保存
# =====================================================
import json

# 古いPython versionだと順番が保持されずOrderedDictを使う必要があるが、Python3.6では問題なさそう
param_dict = {}
param_dict['list_folder'] = list_folder_name
param_dict['img_target_size'] = img_target_size
param_dict['epochs'] = epochs
param_dict['batch_size'] = batch_size
param_dict['filters'] = filters
# param_dict['data_augmentation_num'] = data_augmentation_num
# param_dict['DataGenerator'] ={}
# param_dict['DataGenerator']['width_shift_range'] = datagen.width_shift_range
# param_dict['DataGenerator']['height_shift_range'] = datagen.height_shift_range
# param_dict['DataGenerator']['horizontal_flip'] = datagen.horizontal_flip
# param_dict['DataGenerator']['rotation_range'] = datagen.rotation_range
param_dict['conv2D_repeat_num'] = conv2D_repeat_num

# パラメーターファイルの名前
param_file_name = filename_time + 'param.json'

# 保存先のフォルダを連結
param_file_name = os.path.join(data_folder_name, param_file_name)

# 保存
with open(param_file_name, mode='w') as f:
    json.dump(param_dict, f, indent=4)  # indentを入れることで改行されて見やすくなる

# =====================================================
# VAE
# =====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import cifar10  # ＊
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models, data, batch_size=128):
    encoder, decoder = models
    x_test, y_test = data

    filename = os.path.join(data_folder_name, filename_time + "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    # flagに応じて2次元マップの表示
    # 画像データが大きいとRAM容量不足でセッションがクラッシュするので注意
    if flag_display2dMAP == 1:
        filename = os.path.join(data_folder_name, filename_time + "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = img_target_size  # ＊
        figure = np.zeros((digit_size * n, digit_size * n, 3))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size, 3)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()


# ======================================
# オリジナルデータを読み込み
# ======================================
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob

# 対象のフォルダの中にある画像を順次読み込む
x_data = []
y_data = []
for num, folder in enumerate([list_folder_name]):
    print("Load images... " + str(num))
    list_img = glob.glob(folder + "/*.jpg")
    for j, img in enumerate(list_img):
        img = load_img(img, target_size=(img_target_size, img_target_size))
        img = img_to_array(img)
        x_data.append(img)
        y_data.append(num)

        # プログレス表示
        print(str(j) + ', ', end='')
        if (j % 20) == 0:
            print('')  # 改行

print('Done')

# データ形式をCNNモデルに合わせる（ndarrayへ変更）
x_data = np.array(x_data)
y_data = np.array(y_data)

# データを0.0～1.0へ正規化
x_data = x_data.astype('float32') / 255.0

print("Done")

# データを訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

# 訓練データを訓練データと評価データに分割
# ★valid未使用 x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# network parameters（こっちはあまり変更しない）
image_size = x_train.shape[1]
input_shape = (image_size, image_size, 3)
kernel_size = 3
latent_dim = 2

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(conv2D_repeat_num):  # パラメータ化
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file=data_folder_name + '/vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(conv2D_repeat_num):  # パラメータ化
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file=data_folder_name + '/vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# ===================================================
# 　一定間隔で進捗表示
# ===================================================
from tensorflow import keras

interval_num = 100


class IntervalProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch % interval_num) == 0:
            # print(str(epoch) + 'epochs, loss = '+ str(logs.get('loss')) + ', val_loss = ' + str(logs.get('val_loss')))
            print('%6d epochs, loss = %8.2f, val_loss = %8.2f' % (epoch, logs.get('loss'), logs.get('val_loss')))


# ===================================================
# main
# ===================================================
if __name__ == '__main__':
    models = (encoder, decoder)
    data = (x_test, y_test)

    # reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    plot_model(vae, to_file=data_folder_name + '/vae_cnn.png', show_shapes=True)

    # インスタンス生成
    interval_progress = IntervalProgress()

    # VAEでのリアルタイムデータ拡張は不整合が起きるため行わない
    history = vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None), verbose=0,
                      callbacks=[interval_progress])

    # ===================================================
    # モデルと重みの保存
    # ===================================================
    open(model_file_name, 'w').write(vae.to_json())
    vae.save_weights(weights_file_name)

    # ===================================================
    # plot_results
    # ===================================================
    plot_results(models, data, batch_size=batch_size)

    # ===================================================
    # グラフ出力
    # ===================================================
    # Loss
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    nb_epoch = len(train_loss)
    plt.plot(range(nb_epoch), train_loss, marker='.', label='train_loss')
    plt.plot(range(nb_epoch), valid_loss, marker='.', label='valid_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()