

# %%
import tensorflow as tf
# Hide tensorflow warnings
tf.get_logger().setLevel('ERROR')
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist # this is a hand written digits dataset
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input
from tensorflow.keras import Model, Sequential

import numpy as np
np.random.seed(0)
import matplotlib
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
img_shape = x_train[0].shape
latent_dim = 100 # This is the dimension of the random noise that the generator nn will use
batch_size = 128
epochs = 1000


# %%
# create the image generator laers
generator_layers = [
    Dense(256, activation='relu'),
    BatchNormalization(momentum=0.8),
    Dense(512, activation='relu'),
    BatchNormalization(momentum=0.8),
    Dense(1024, activation='relu'),
    BatchNormalization(momentum=0.8),
    Dense(784, activation='tanh'),
    Reshape(img_shape)
]
# make the generator
generator = Sequential(generator_layers)

# %%
# create a discriminator nn layers
discriminator_layers = [
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid'),
]
# add layers to sequential model
discriminator = Sequential(discriminator_layers)


# %%
# compile the discriminator nn
discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model make sure to only train generator when called
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# make a combined model witth discriminator and generator
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer="adam")


# %%
# Rescale training data to values between -1 to 1
x_train = x_train / 127.5 - 1.
# x_train = np.expand_dims(x_train, axis=3)

# Adversarial ground truths
valid = np.ones((batch_size, 1)) # an array of ones that is the same size as 1 batch
print(valid[0])
fake = np.zeros((batch_size, 1)) # an array of zeros that is the same size as 1 batch
print(fake[0])

# %%

# Number of eopchs designated at start of code along with batch size
eRange = 0
for epoch in range(epochs):

    # ---------------------
    #  Train the Discriminator
    # ---------------------

    print("Epoch Range: " + eRange)

    # Select a random batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    
    imgs = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generate a batch of new images
    gen_imgs = generator.predict(noise)

    # Train the discriminator
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train the Generator
    # ---------------------

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (to have the discriminator label samples as valid)
    discriminator.trainable = False
    g_loss = combined.train_on_batch(noise, valid)
    
    if epoch % 50 == 0:
        r, c = 5,5
        noise = np.random.normal(0, 1, (r*c, latent_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.suptitle(f"Epoch: {epoch+1}")
        plt.show()

    eRange += 1


# %%
# Generate another "digit"
noise = np.random.normal(0, 1, (1, latent_dim))
gen_img = generator.predict(noise)[0]
plt.imshow(gen_img, cmap='gray')
plt.show()

