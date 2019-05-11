import os
import argparse
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import utils
from torchvision import models
from model import *
import metrics


class Dict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--generator_checkpoint", type=str, default="")
parser.add_argument("--discriminator_checkpoint", type=str, default="")
args = parser.parse_args()

hyper_params = Dict(
    latent_size=512,
    generator_learning_rate=2e-3,
    generator_beta1=0.0,
    generator_beta2=0.99,
    generator_epsilon=1e-8,
    discriminator_learning_rate=2e-3,
    discriminator_beta1=0.0,
    discriminator_beta2=0.99,
    discriminator_epsilon=1e-8,
    real_gradient_penalty_weight=5.0,
    fake_gradient_penalty_weight=0.0,
)

generator = Generator(
    min_resolution=4,
    max_resolution=256,
    min_channels=16,
    max_channels=512,
    embedding_param=Dict(num_embeddings=10, embedding_dim=512),
    linear_params=[
        Dict(in_features=1024, out_features=512),
        *[Dict(in_features=512, out_features=512)] * 8
    ],
    num_features=512,
    out_channels=3
)
print(generator)

if args.generator_checkpoint:
    generator.load_state_dict(torch.load(args.generator_checkpoint))

discriminator = Discriminator(
    min_resolution=4,
    max_resolution=256,
    min_channels=16,
    max_channels=512,
    num_classes=10,
    in_channels=3
)
print(discriminator)

if args.discriminator_checkpoint:
    discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))

generator_optimizer = optim.Adam(
    params=generator.parameters(),
    lr=hyper_params.generator_learning_rate,
    betas=(
        hyper_params.generator_beta1,
        hyper_params.generator_beta2
    ),
    eps=hyper_params.generator_epsilon
)

discriminator_optimizer = optim.Adam(
    params=discriminator.parameters(),
    lr=hyper_params.discriminator_learning_rate,
    betas=(
        hyper_params.discriminator_beta1,
        hyper_params.discriminator_beta2
    ),
    eps=hyper_params.discriminator_epsilon
)

dataset = datasets.LSUN(
    root="lsun",
    classes="train",
    transform=transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True
)

for epoch in range(args.num_epochs):
    for reals, labels in data_loader:

        discriminator.zero_grad()

        reals.requires_grad_(True)
        real_logits = discriminator(reals, labels)

        latents = torch.randn(reals.size(0), hyper_params.latent_size)
        fakes = generator(latents, labels)
        fake_logits = discriminator(fakes.detach(), labels)

        real_losses = nn.functional.softplus(-real_logits)
        fake_losses = nn.functional.softplus(fake_logits)
        discriminator_losses = real_losses + fake_losses

        if hyper_params.real_gradient_penalty_weight:
            real_gradients = torch.autograd.grad(
                outputs=real_logits,
                inputs=reals,
                grad_outputs=torch.ones_like(real_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            real_gradient_penalties = torch.sum(real_gradients ** 2, dim=(1, 2, 3))
            discriminator_losses += real_gradient_penalties * hyper_params.real_gradient_penalty_weight

        if hyper_params.fake_gradient_penalty_weight:
            fake_gradients = torch.autograd.grad(
                outputs=fake_logits,
                inputs=fakes,
                grad_outputs=torch.ones_like(fake_logits),
                retain_graph=True,
                create_graph=True
            )
            fake_gradient_penalties = torch.sum(fake_gradients ** 2, dim=(1, 2, 3))
            discriminator_losses += fake_gradient_penalties * hyper_params.fake_gradient_penalty_weight

        discriminator_loss = discriminator_losses.mean()
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        generator.zero_grad()

        fake_losses = nn.functional.softplus(-fake_logits)
        generator_losses = fake_losses

        generator_loss = generator_losses.mean()
        generator_loss.backward(retain_graph=False)
        generator_optimizer.step()

        print(f"epoch: {epoch} discriminator_loss: {discriminator_loss} generator_loss: {generator_loss}")

    torch.save(generator.state_dict(), f"model/generator/epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"model/discriminator/epoch_{epoch}.pth")

dataset = datasets.LSUN(
    root="lsun",
    classes="test",
    transform=transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=False
)

inception_v3 = models.inception_v3(pretrained=True)

for param in inception_v3.parameters():
    param.requires_grad = False

inception_v3.fc = nn.Identity()


def data_generator():

    inception_v3.eval()

    for reals, labels in data_loader:

        with torch.no_grad():

            latents = torch.randn(reals.size(0), hyper_params.latent_size)
            fakes = generator(latents, labels)

            reals = nn.functional.interpolate(
                input=reals,
                size=(299, 299),
                mode="bilinear"
            )
            fakes = nn.functional.interpolate(
                input=fakes,
                size=(299, 299),
                mode="bilinear"
            )

            real_features = inception_v3(reals)
            fake_features = inception_v3(fakes)

            yield real_features, fake_features


real_features, fake_features = map(torch.cat, zip(*data_generator()))
frechet_inception_distance = metrics.frechet_inception_distance(real_features, fake_features)
print(f"frechet_inception_distance: {frechet_inception_distance}")
