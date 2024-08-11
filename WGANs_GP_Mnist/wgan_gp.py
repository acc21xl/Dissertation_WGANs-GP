import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad

# Setup directories
os.makedirs("saved_images", exist_ok=True)

# Setup argument parser with multiple hyperparameters
def setup_parser():
    parser = argparse.ArgumentParser(description="Train a WGAN on MNIST")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of each batch")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for Adam optimizer")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of CPU threads for data loading")
    parser.add_argument("--z_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--image_size", type=int, default=28, help="Height and width of the images")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
    parser.add_argument("--disc_steps", type=int, default=5, help="Number of discriminator steps per generator step")
    parser.add_argument("--weight_clip", type=float, default=0.01, help="Clipping value for discriminator weights")
    parser.add_argument("--sample_freq", type=int, default=400, help="Sampling frequency for generated images")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Gradient penalty lambda hyperparameter")
    return parser.parse_args()


# Compute gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device) # Random weight for interpolation
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True) # Interpolated samples
    d_interpolates = D(interpolates)  # Interpolated samples' output
    fake = Variable(torch.ones(d_interpolates.size(), device=real_samples.device), requires_grad=False)  # Fake output
    gradients = grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, 
                     create_graph=True, retain_graph=True, only_inputs=True)[0]  # Gradients of interpolated samples
    gradients = gradients.view(gradients.size(0), -1)  # Flatten gradients
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()        # Compute gradient penalty
    return gradient_penalty


# Generator and Discriminator classes
class ImageGenerator(nn.Module):
    def __init__(self, z_dim, img_shape):
        super(ImageGenerator, self).__init__()
        self.model = nn.Sequential(    #MLp generator With layers number of 5
            self.gen_block(z_dim, 128, normalize=False),
            self.gen_block(128, 256),
            self.gen_block(256, 512),
            self.gen_block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),  #only this layer is not needed to normalize
            nn.Tanh() 
        )

    #this function is used to create a block of layers, which is used in the generator,
    # linear layer, batch normalization, and leaky relu activation function
    def gen_block(self, in_features, out_features, normalize=True):
        layers = [nn.Linear(in_features, out_features)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_features, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, z):  #forward function to generate the image
        img = self.model(z)
        return img.view(img.shape[0], *img_shape) #return the image in the shape of the input image

# Define the discriminator
class ImageDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(ImageDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),  #input the image from the generator in training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  
        )

    def forward(self, img):
        return self.model(img.view(img.shape[0], -1))  #return the output of the discriminator, which is the validity of the image

# Initialize and configure
args = setup_parser()  
img_shape = (args.channels, args.image_size, args.image_size) #set to 1, 28, 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = ImageGenerator(args.z_dim, img_shape).to(device)   #initialize the generator
discriminator = ImageDiscriminator(img_shape).to(device)  # initialize the discriminator
data_loader = DataLoader(   #load the mnist dataset
    datasets.MNIST("../../data/mnist", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(args.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])
                   ])),
    batch_size=args.batch_size, shuffle=True
)

#initialize the optimizer for the generator and discriminator
opt_gen = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
opt_disc = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

# Training loop 200*938
def train():
    for epoch in range(args.epochs):
        for i, (imgs, _) in enumerate(data_loader):
            real_imgs = imgs.to(device)

            # Train Discriminator
            opt_disc.zero_grad()
            z = torch.randn(imgs.size(0), args.z_dim, device=device)
            fake_imgs = generator(z).detach()
            real_validity = discriminator(real_imgs)  #real image validity
            fake_validity = discriminator(fake_imgs) #fake image validity
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data) #compute the gradient penalty

            loss_disc = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty
            loss_disc.backward()  #backpropagate the loss
            opt_disc.step()  #update the weights of the discriminator

            # Train Generator every 'n_critic' steps
            if i % args.disc_steps == 0:  #train the generator every 5 steps
                opt_gen.zero_grad()
                generated_imgs = generator(z)
                loss_gen = -torch.mean(discriminator(generated_imgs))
                loss_gen.backward()
                opt_gen.step()

                print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(data_loader)}] [D loss: {loss_disc.item()}] [G loss: {loss_gen.item()}]")

                if i % args.sample_freq == 0:
                    save_image(generated_imgs.data[:25], f"saved_images/{epoch}_{i}.png", nrow=5, normalize=True)

if __name__ == "__main__":
    train()
