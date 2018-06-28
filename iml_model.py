from autoencoder3d import AutoEncoder
import numpy as np

from torch.autograd import Variable

import matplotlib.pyplot as plt

from sklearn import mixture

import torch

class imlgenModel():

    def __init__(self, w, h, z, distr=None):
        self.distr = distr
        self.w = w
        self.h = h
        self.z = z
        self.autoencoder = AutoEncoder((w, h, z))

        if distr is None:
            self.distr = mixture.GaussianMixture(n_components=30, verbose=0, n_init=20, max_iter = 200)

    #def load_autoencoder(self, filename):
    #    self.autoencoder.load(filename)

    def train_autoencoder(self, images_gen, steps=250, batch_size=50, weights_filepath="model"):
        for i in range(steps):
            batch = next(images_gen)[0]
            batch = Variable(batch)
            cur_loss = self.autoencoder.train(batch)
            if (i + 1) % 10 == 0:
                print("Step", i + 1, "Loss = %.3f" % cur_loss.data[0])

    def encode(self, x):
        #x = torch.Tensor(x)
        x = Variable(x)
        return self.autoencoder.encode(x).data

    def decode(self, x):
        #x = torch.Tensor(x)
        x = Variable(x)
        return self.autoencoder.decode(x).data

    def fit_distribution(self, x_train):
        x_encoded = self.encode(x_train)
        self.distr.fit(x_encoded)

    def generate(self, n_samples):
        gen = self.distr.sample(n_samples=n_samples)[0]
        gen = torch.Tensor(gen).cuda()
        gen_decoded = self.decode(gen)
        return gen_decoded.cpu()