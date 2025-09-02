import numpy as np
np.random.seed(1337)  # for reproducibility

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# -----------------------------
# 讀取圖片資料
# -----------------------------
img_size = 128
latent_dim = 64  # 壓縮後向量維度

datagen = ImageDataGenerator(rescale=1./255 )  # 正規化到 [-0.5,0.5]

train_generator = datagen.flow_from_directory(
    'images',                  # images/bob, images/platypus
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode=None,           # 不需要標籤
    batch_size=32,
    shuffle=False              # 保持順序方便對應
)

# 把所有圖片合併成 numpy array
x_data = np.concatenate([next(train_generator) for _ in range(len(train_generator))])


# 攤平成 1 維，方便輸入 Dense
x_data_flat = x_data.reshape((x_data.shape[0], -1))

# -----------------------------
# 建立 encoder
# -----------------------------
input_img = Input(shape=(img_size*img_size,))  # 攤平後的 128*128
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoder_output = Dense(latent_dim, activation='relu')(encoded)  # 壓縮到 latent_dim

decoded = Dense(64, activation='relu')(encoder_output)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(img_size*img_size, activation='sigmoid')(decoded)

# 建 encoder
autoencoder = Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(
    x_data_flat, x_data_flat,
    epochs=20,
    batch_size=32,
    shuffle=True
)
encoder = Model(inputs=input_img, outputs=encoder_output)  # 只取向量

# -----------------------------
# 提取向量
# -----------------------------
latent_vectors = encoder.predict(x_data_flat)



# -----------------------------
# 可選：分組存成兩個 npy 檔
# -----------------------------
num_bob = len([f for f in train_generator.filenames if f.startswith('bob')])
bob_vectors = latent_vectors[:num_bob]
platypus_vectors = latent_vectors[num_bob:]

np.save('bob_vectors.npy', bob_vectors)
np.save('platypus_vectors.npy', platypus_vectors)
