# tinygrad version of https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# wget https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip

# %%
# == Imports / Constants / Model Definition ==

import os
import pathlib
import numpy as np
import time
from tinygrad import nn
from tinygrad.nn import optim
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class Generator:
  def __init__(self):
    self.main = [
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      Tensor.relu,
      # state size. ``(ngf*8) x 4 x 4``
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      Tensor.relu,
      # state size. ``(ngf*4) x 8 x 8``
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      Tensor.relu,
      # state size. ``(ngf*2) x 16 x 16``
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      Tensor.relu,
      # state size. ``(ngf) x 32 x 32``
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      Tensor.tanh]
  def __call__(self, x): return x.sequential(self.main)

class Discriminator:
  def __init__(self):
    self.main = [
      # input is ``(nc) x 64 x 64``
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      lambda x: x.leakyrelu(0.2),
      # state size. ``(ndf) x 32 x 32``
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      lambda x: x.leakyrelu(0.2),
      # state size. ``(ndf*2) x 16 x 16``
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      lambda x: x.leakyrelu(0.2),
      # state size. ``(ndf*4) x 8 x 8``
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      lambda x: x.leakyrelu(0.2),
      # state size. ``(ndf*8) x 4 x 4``
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)]
  def __call__(self, x): return x.sequential(self.main)

from datasets.imagenet import _iterate
import torchvision.transforms.functional as F
def celeba_iterate():
  PATH = pathlib.Path(__file__).parent.parent / "datasets/img_align_celeba"
  return _iterate([PATH / x for x in os.listdir(PATH)], 32, True, image_size, image_size)

# %%
# == Trainer ==

import matplotlib.pyplot as plt

from tinygrad.jit import TinyJit

@TinyJit
def train_step(img, noise):
  optimizerD.zero_grad()
  optimizerG.zero_grad()

  # generate images
  fake = netG(noise)

  # train discriminator
  errD_real = netD(img).mean()
  errD_fake = (1-netD(fake.detach())).mean()
  errD = errD_real + errD_fake
  errD.backward()
  optimizerD.step()

  # train generator
  errG = netD(fake).mean()
  errG.backward()
  optimizerG.step()

  return fake.realize(), errD.realize(), errG.realize()

if __name__ == "__main__":
  netG = Generator()
  netD = Discriminator()
  optimizerD = optim.Adam(optim.get_parameters(netD), lr=lr, b1=beta1)
  optimizerG = optim.Adam(optim.get_parameters(netG), lr=lr, b1=beta1)

  from extra.helpers import cross_process
  st = time.perf_counter()
  for x,_ in cross_process(celeba_iterate):
    GlobalCounters.reset()

    img = Tensor((x.astype(np.float32)-128)/255).permute([0,3,1,2])
    noise = Tensor.randn(x.shape[0], nz, 1, 1)

    fake, errD, errG = train_step(img, noise)

    et = time.perf_counter()
    print(f"{(et-st)*1000:7.2f} ms:", img.shape, fake.shape, errD.numpy(), errG.numpy())
    st = et

    #plt.imshow(fake[0].numpy())

# %%
