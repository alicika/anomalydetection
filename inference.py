# CNNを用いたVAEの推論版
# モデルはファイル読み込みではなく、ネットワークを作る
# 推論対象フォルダと推論データ時刻文字列を指定するだけでOKにした
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ======================================================
# 推論対象の設定
# ======================================================
folder_name_predict = data_inference

# 推論データ時刻文字列の指定（学習直後に推論する場合はコメアウトOK）
# filename_time = 'd190806_0204'

print(filename_time)
print('--------------------------')

# ======================================================
# システム設定
# ======================================================
# data_folder_name = 'data_output' #モデル、重み、パラメーターなどを保存するフォルダ名（学習直後に推論する場合はコメアウトOK）


# ======================================================
# 重みファイル読み込み
# ======================================================
# 読み込む重みファイル名
weights_file_name = filename_time + 'weights_vae.h5'

# 読み込む先のフォルダを連結
weights_file_name = os.path.join(data_folder_name, weights_file_name)

# ======================================================
# パラメーター読み込み
# ======================================================
# パラメーターファイルの名前
param_file_name = filename_time + 'param.json'

# 保存先のフォルダを連結
param_file_name = os.path.join(data_folder_name, param_file_name)

# 読み込み
with open(param_file_name, mode='r') as f:
    param_dict = json.load(f)

# 表示
for k, v in param_dict.items():
    print(k + ' : ' + str(v))

print('--------------------------')

# ======================================================
# パラメータ反映
# ======================================================
# 推論データの画像読み込みサイズ
img_target_size = param_dict['img_target_size']

# フィルター
filters = param_dict['filters']

# conv2Dの繰り返し回数
conv2D_repeat_num = param_dict['conv2D_repeat_num']


# ======================================================
# Lambdaで使用
# ======================================================
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ======================================================
# モデル読み込み
# モデルファイル読み込みだとエラーになるので、学習時と全く同じモデルを作る
# ======================================================
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K

# network parameters
input_shape = (img_target_size, img_target_size, 3)
kernel_size = 3
latent_dim = 2

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(conv2D_repeat_num):
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
# encoder.summary()
# plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(conv2D_repeat_num):
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
# decoder.summary()
# plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# ===================================================
# 重みの読み込み。損失関数は読み込み不要
# ===================================================
# 重みを読み込む
vae.load_weights(weights_file_name)

# ======================================
# データ読み込み
# ======================================
from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import random

# 推論画像リストを取得しランダムに並び替え
list_file = glob.glob(folder_name_predict + "/*.jpg")
random.shuffle(list_file)

# 推論画像リスト全てのファイルに推論
for n, filename in enumerate(list_file):

    # 指定の推論数を超えたらブレイク
    if n >= inference_times:
        break

    # 指定サイズで画像読み込み
    input_img = load_img(filename, target_size=(img_target_size, img_target_size))

    # データ形式をCNNモデルに合わせる（3次元のPILから3次元のndarrayへ変更）
    input_img = img_to_array(input_img)

    # データを0.0～1.0へ正規化
    input_img = input_img.astype('float32') / 255.0

    # 次元を合わせる
    input_img = np.expand_dims(input_img, axis=0)

    # =====================================================
    # 推論
    # =====================================================
    output_img = vae.predict(input_img)

    # =====================================================
    # 結果表示
    # 次元を上げてリスト化しているので[0]で指定する
    # =====================================================
    import matplotlib.pyplot as plt
    import cv2

    # 差分
    diff_img = cv2.absdiff(input_img[0], output_img[0])

    # 3つ並べて表示
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 3, 1)
    filename2 = filename.split('/')[1]  # 簡易的にフォルダ名は削除
    plt.title(filename2)
    plt.imshow(input_img[0])

    plt.subplot(1, 3, 2)
    plt.title('generated')
    plt.imshow(output_img[0])

    plt.subplot(1, 3, 3)
    var_sum = str(np.sum(diff_img))
    plt.title('diff_img : ' + var_sum)
    plt.imshow(diff_img)
