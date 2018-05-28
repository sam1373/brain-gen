import numpy as np

from brainpedia.brainpedia import Brainpedia
from brainpedia.fmri_processing import invert_preprocessor_scaling

import argparse

from nilearn import plotting
import matplotlib.pyplot as plt


from imlgen import imlgenModel

from sklearn import mixture


import torch

from torch.autograd import Variable

"""
parser = argparse.ArgumentParser(description="Train ICW_FMRI_GAN.")
parser.add_argument('train_data_dir', help='the directory containing real fMRI data to train on')
parser.add_argument('train_data_dir_cache', help='the directory to use as a cache for the train_data_dir preprocessing')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()
"""
argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, help='number of epochs for autoencoder training', default=200)
argparser.add_argument('--batch_size', type=int, help='size of batches for autoencoder training', default=128)
argparser.add_argument('--seed', type=int, help='random seed', default=1)
argparser.add_argument('--comp', type=int, help='gaussian mixture components', default=30)
argparser.add_argument('--inits', type=int, help='gaussian mixture fit initializations', default=15)
argparser.add_argument('--dim', type=int, help='dimensionality of latent representations', default=100)
argparser.add_argument('--dir', type=str, help='directory of data', default="/home/samuel/Data/CelebAligned/")

arguments = argparser.parse_args()

train_data_dir = "/home/samuel/nilearn_data/neurovault/collection_1952/"
train_data_dir_cache = "data/collection_1952_train_cache/"

CUDA = torch.cuda.is_available()
if CUDA:
    print("Using GPU optimizations!")

# ========== Hyperparameters ==========
DOWNSAMPLE_SCALE = 0.25
MULTI_TAG_LABEL_ENCODING = False
TRAINING_STEPS = 200000
BATCH_SIZE = 50
MODEL_DIMENSIONALITY = 64
CONDITONING_DIMENSIONALITY = 5
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 1
LAMBDA = 10
VISUALIZATION_INTERVAL = 1000
NOISE_SAMPLE_LENGTH = 128

brainpedia = Brainpedia(data_dirs=[train_data_dir],
                        cache_dir=train_data_dir_cache,
                        scale=DOWNSAMPLE_SCALE,
                        multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
all_brain_data, all_brain_data_tags = brainpedia.all_data()
brainpedia_generator = Brainpedia.batch_generator(all_brain_data, all_brain_data_tags, BATCH_SIZE, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

print(all_brain_data.shape)
print(brain_data_shape, brain_data_tag_shape)
batch = next(brainpedia_generator)
print(batch[0].shape, batch[1].shape)


for i in range(0):
    real_brain_img_axes = plt.subplot(2, 1, 1)
    real_sample_data = batch[0][i][0]
    print(np.min(real_sample_data), np.max(real_sample_data), np.mean(real_sample_data), np.std(real_sample_data))
    #real_sample_data[real_sample_data < 0.3] = 0
    print("sample", i, "label:", np.argmax(batch[1][i]))
    #real_sample_data = np.expand_dims(real_sample_data, axis=-1)
    upsampled_real_brain_img = invert_preprocessor_scaling(real_sample_data, brainpedia.preprocessor)
    plotting.plot_glass_brain(upsampled_real_brain_img, threshold='auto', title="[REAL]", axes=real_brain_img_axes)
    plt.show()


w, h, z = brain_data_shape[-3:]
print("object size:", w, h, z)
model = imlgenModel(w, h, z, mixture.GaussianMixture(n_components=arguments.comp, verbose=1, n_init=arguments.inits, max_iter = 100))

model.train_autoencoder(brainpedia_generator, steps=2500)
#model.train_autoencoder(x_train, x_test, arguments.epochs, arguments.batch_size, weights_filepath="celeb_model100")
real_samples = batch[0]
encoded_samples = model.encode(real_samples)
decoded_samples = model.decode(encoded_samples)

print(real_samples.shape, encoded_samples.shape, decoded_samples.shape)

for i in range(5):
    real_brain_img_axes = plt.subplot(2, 1, 1)
    real_sample = real_samples[i][0]
    decoded_axes = plt.subplot(2, 1, 2)
    decoded_sample = decoded_samples[i][0]
    #print(np.min(real_sample_data), np.max(real_sample_data), np.mean(real_sample_data), np.std(real_sample_data))
    #print("sample", i, "label:", np.argmax(batch[1][i]))
    real_sample = invert_preprocessor_scaling(real_sample, brainpedia.preprocessor)
    decoded_sample = invert_preprocessor_scaling(decoded_sample, brainpedia.preprocessor)
    plotting.plot_glass_brain(real_sample, threshold='auto', title="real", axes=real_brain_img_axes)
    plotting.plot_glass_brain(decoded_sample, threshold='auto', title="decoded", axes=decoded_axes)
    plt.show()

#displayImageTable2(x_test, decoded_imgs)
all_brain_data = torch.Tensor(all_brain_data).cuda()
model.fit_distribution(all_brain_data)

n_gen = 1000
n_display = 64

gen = model.generate(n_gen)

for i in range(5):
    gen_brain_img_axes = plt.subplot(2, 1, 1)
    gen_sample = gen[i][0]

    gen_sample = invert_preprocessor_scaling(gen_sample, brainpedia.preprocessor)
    plotting.plot_glass_brain(gen_sample, threshold='auto', title="gen", axes=gen_brain_img_axes)
    plt.show()
    #displayImageTable(gen[i * n_display : (i + 1) * n_display], n_display=n_display)
#plotting.plot_glass_brain(synthetic_sample_brain_img, threshold='auto', title="[SYNTHETIC] " + title, axes=synthetic_brain_img_axes)