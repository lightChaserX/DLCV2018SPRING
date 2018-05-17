import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from utils import *
from init import *

'''
TODO: Define the LeakyReLUBNConv2d unit
'''
class LeakyReLUBNConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        model += [nn.BatchNorm2d(n_out)]
        model +=[nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        
    def forward(self, x):
        return self.model(x)

'''
TODO: Define the LeakyReLUBNConvTranspose2d unit
'''
class LeakyReLUBNConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        model += [nn.BatchNorm2d(n_out)]
        model +=[nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        
    def forward(self, x):
        return self.model(x)

'''
TODO: Define the Flatten unit
'''
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

'''
TODO: Define the unFlatten unit
'''
class unFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1)//16, 4, 4)

'''
TODO: Define the VAE
'''
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            LeakyReLUBNConv2d(  3,  32, kernel_size=4, stride=2, padding=1),
            LeakyReLUBNConv2d( 32,  64, kernel_size=4, stride=2, padding=1),
            LeakyReLUBNConv2d( 64, 128, kernel_size=4, stride=2, padding=1),
            LeakyReLUBNConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 4096),
            unFlatten(),
            LeakyReLUBNConvTranspose2d( 256, 128, kernel_size=6, stride=2),
            LeakyReLUBNConvTranspose2d( 128,  64, kernel_size=6, stride=2),
            LeakyReLUBNConvTranspose2d(  64,  32, kernel_size=6, stride=2),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.muParametrize     = nn.Linear(256*4*4, 512)
        self.logvarParametrize = nn.Linear(256*4*4, 512)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def bottleneck(self, h):
        mu = self.muParametrize(h)
        logvar = self.logvarParametrize(h)
        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar

    def forward(self, x, test=False):
        x = self.encoder(x)
        x, mu, logvar = self.bottleneck(x)
        x = self.reparameterize(mu, logvar)
        if test:
            x = mu
        x = self.decoder(x)
        return x, mu, logvar
    
    def save(self, dirname, iterations):
        filename = os.path.join(dirname, 'model_%08d.pkl' % (iterations + 1))
        torch.save(self.state_dict(), filename)
        
    def resume(self, dirname):
        last_model_name = get_model_list(dirname, "model")
        if last_model_name is None:
            return 0
        self.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12:-4])
        print('Resume from iteration %d' % iterations)
        return iterations

'''
TODO: Define the discriminator of GAN
'''
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        print(x.shape)
        return x.view(-1, 1).squeeze(1)
    
    def save(self, dirname, iterations):
        filename = os.path.join(dirname, 'dis_%08d.pkl' % (iterations + 1))
        torch.save(self.state_dict(), filename)
        
    def resume(self, dirname):
        last_model_name = get_model_list(dirname, "dis")
        if last_model_name is None:
            return 0
        self.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12:-4])
        print('Resume from iteration %d' % iterations)
        return iterations

'''
TODO: Define the generator of GAN
'''
class generator(nn.Module):
    def __init__(self, num_z):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(num_z, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x
    
    def save(self, dirname, iterations):
        filename = os.path.join(dirname, 'gen_%08d.pkl' % (iterations + 1))
        torch.save(self.state_dict(), filename)
        
    def resume(self, dirname):
        last_model_name = get_model_list(dirname, "gen")
        if last_model_name is None:
            return 0
        self.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12:-4])
        print('Resume from iteration %d' % iterations)
        return iterations

'''
TODO: Define the generator of ACGAN
'''
class netG_ACGAN(nn.Module):
    def __init__(self, nz):
        super(netG_ACGAN, self).__init__()
        self.nz = nz

        # first linear layer
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(nz+1, 384, 4, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input, label):
        x = torch.cat([input, label.view(-1,1,1,1)],1)
        tconv1 = self.tconv1(x)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output
    
    def save(self, dirname, iterations):
        filename = os.path.join(dirname, 'gen_%08d.pkl' % (iterations + 1))
        torch.save(self.state_dict(), filename)
        
    def resume(self, dirname):
        last_model_name = get_model_list(dirname, "gen")
        if last_model_name is None:
            return 0
        self.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12:-4])
        print('Resume from iteration %d' % iterations)
        return iterations

'''
TODO: Define the discriminator of ACGAN
'''
class netD_ACGAN(nn.Module):
    def __init__(self, num_classes=2):
        super(netD_ACGAN, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 4*4*512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes
    
    def save(self, dirname, iterations):
        filename = os.path.join(dirname, 'gen_%08d.pkl' % (iterations + 1))
        torch.save(self.state_dict(), filename)
        
    def resume(self, dirname):
        last_model_name = get_model_list(dirname, "gen")
        if last_model_name is None:
            return 0
        self.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12:-4])
        print('Resume from iteration %d' % iterations)
        return iterations