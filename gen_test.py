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
import pickle
import nibabel
import json
import random

def noise(size, cuda=False):
    noise = torch.from_numpy(np.random.normal(0.0, size=size)).float()
    if cuda:
        noise = noise.cuda()
    return noise


train_data_dir = "/home/samuel/nilearn_data/neurovault/collection_1952/"
train_data_dir_cache = "data/collection_1952_train_cache/"
gen_data_dir = "data/gen_same/"

CUDA = torch.cuda.is_available()
if CUDA:
    print("Using GPU optimizations!")

# ========== Hyperparameters ==========
DOWNSAMPLE_SCALE = 0.25
MULTI_TAG_LABEL_ENCODING = False
BATCH_SIZE = 50


brainpedia = Brainpedia(data_dirs=[train_data_dir],
                        cache_dir=train_data_dir_cache,
                        scale=DOWNSAMPLE_SCALE,
                        multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
all_brain_data, all_brain_data_tags = brainpedia.all_data()
brainpedia_generator = Brainpedia.batch_generator(all_brain_data, all_brain_data_tags, BATCH_SIZE, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

unique_brain_data_tags = np.unique(all_brain_data_tags, axis=0)
print(unique_brain_data_tags)
print(unique_brain_data_tags.shape)
print(np.sum(unique_brain_data_tags, axis=1))

for i in range(len(unique_brain_data_tags)):
    class_data = all_brain_data[np.argmax(all_brain_data_tags, axis=1) == np.argmax(unique_brain_data_tags[i])]
    if len(class_data) > 50:
        print(i, class_data.shape)
    """
    for j in range(3):
        class_brain_img_axes = plt.subplot(2, 1, 1)
        class_sample = class_data[j][0]
        class_sample = invert_preprocessor_scaling(class_sample, brainpedia.preprocessor)
        plotting.plot_glass_brain(class_sample, threshold='auto', axes=class_brain_img_axes)
        plt.show()
    """



print(all_brain_data.shape)
print(brain_data_shape, brain_data_tag_shape)
batch = next(brainpedia_generator)
print(batch[0].shape, batch[1].shape)



w, h, z = brain_data_shape[-3:]
print("object size:", w, h, z)

retrain = arguments.retrain



n_gen = 1000

#gen = model.generate(n_gen)

for i in range(1000):
    idx = random.randint(0, len(all_brain_data))
    gen = all_brain_data[idx]
    gen_tags = all_brain_data_tags[idx]
    #noise = np.random.normal(scale=0.1, size=gen.shape)
    #gen += noise

    # Upsample synthetic brain image data
    synthetic_sample_data = gen.squeeze()
    upsampled_synthetic_brain_img = invert_preprocessor_scaling(synthetic_sample_data, brainpedia.preprocessor)

    # Save upsampled synthetic brain image data
    synthetic_sample_output_path = "{0}image_{1}.nii.gz".format(gen_data_dir, i)
    nibabel.save(upsampled_synthetic_brain_img, synthetic_sample_output_path)

    # Save synthetic brain image metadata
    with open("{0}image_{1}_metadata.json".format(gen_data_dir, i), 'w') as metadata_f:
        tags = ""
        for sample_label in brainpedia.preprocessor.decode_label(gen_tags):
            tags += sample_label + ','

        json.dump({'tags': tags}, metadata_f)
        if i == 0:
            print(tags)
    if (i + 1) % 10 == 0:
        print(i + 1, "generated")