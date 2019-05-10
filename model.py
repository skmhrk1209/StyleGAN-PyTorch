import torch
from torch import nn
import numpy as np
import functools
from attrdict import AttrDict


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(out_features, in_features))

        std = np.sqrt(variance_scale / in_features)
        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            weight *= std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)

        if bias:
            bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(bias)
        else:
            bias = None

        self.weight = weight
        self.bias = bias

    def forward(self, inputs):

        outputs = nn.functional.linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias
        )

        return outputs


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        std = np.sqrt(variance_scale / num_embeddings)
        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            weight *= std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)

        self.weight = weight

    def forward(self, inputs):

        outputs = nn.functional.embedding(
            input=inputs,
            weight=self.weight
        )

        return outputs


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        std = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)
        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            weight *= std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)

        if bias:
            bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(bias)
        else:
            bias = None

        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, inputs):

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )

        return outputs


class ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))

        std = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)
        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            weight *= std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)

        if bias:
            bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(bias)
        else:
            bias = None

        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, inputs):

        outputs = nn.functional.conv_transpose2d(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )

        return outputs


class PixelNorm(nn.Module):

    def __init__(self, epsilon=1e-12):

        super().__init__()

        self.epsilon = epsilon

    def forward(self, inputs):

        outputs = inputs * torch.rsqrt((inputs ** 2).mean(dim=1, keepdim=True) + self.epsilon)

        return outputs


class BatchStddev(nn.Module):

    def __init__(self, groups, epsilon=1e-12):

        super().__init__()

        self.groups = groups
        self.epsilon = epsilon

    def forward(self, inputs):

        outputs = torch.reshape(inputs, [self.groups, -1, *inputs.shape[1:]])
        outputs -= torch.mean(outputs, axis=0, keepdims=True)
        outputs = torch.mean(outputs ** 2, axis=0)
        outputs = torch.sqrt(outputs + self.epsilon)
        outputs = torch.reduce_mean(outputs, axis=[1, 2, 3], keepdims=True)
        outputs = torch.repeat(outputs, [self.groups, 1, *inputs.shape[2:]])
        outputs = torch.cat((outputs, inputs), dim=1)

        return outputs


class LearnedConstant(nn.Module):

    def __init__(self, num_channels, resolution):

        super().__init__()

        self.constant = nn.Parameter(torch.ones(1, num_channels, resolution, resolution))

    def forward(self, inputs):

        outputs = self.constant.repeat(inputs.shape[0], *[1 for _ in self.constant.shape[1:]])

        return outputs


class LearnedNoise(nn.Module):

    def __init__(self, num_channels):

        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, inputs):

        noise = torch.randn(inputs.shape[0], 1, *inputs.shape[2:])
        outputs = inputs + noise * self.weight

        return outputs


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, num_features, num_channels, bias, variance_scale, weight_scale):

        super().__init__()

        self.instance_norm2d = nn.InstanceNorm2d(
            num_features=num_channels,
            affine=False
        )

        self.linear1 = Linear(
            in_features=num_features,
            out_features=num_channels,
            bias=bias,
            variance_scale=variance_scale,
            weight_scale=weight_scale
        )

        self.linear2 = Linear(
            in_features=num_features,
            out_features=num_channels,
            bias=bias,
            variance_scale=variance_scale,
            weight_scale=weight_scale
        )

    def forward(self, inputs, styles):

        outputs = self.instance_norm2d(inputs)

        gamma = self.linear1(styles)
        beta = self.linear2(styles)

        outputs *= gamma.unsqueeze(-1).unsqueeze(-1)
        outputs += beta.unsqueeze(-1).unsqueeze(-1)

        return outputs


class Concat(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dim = dim

    def forward(self, *inputs):

        outputs = torch.cat(inputs, self.dim)

        return outputs


class MappingNetwork(nn.Module):

    def __init__(self, embedding_param, linear_params):

        super().__init__()

        self.modules = nn.ModuleDict(AttrDict(
            embedding_block=nn.ModuleDict(AttrDict(
                embedding=Embedding(
                    num_embeddings=embedding_param.num_embeddings,
                    embedding_dim=embedding_param.embedding_dim,
                    variance_scale=1,
                    weight_scale=True
                ),
                concat=Concat(dim=1),
                pixel_norm=PixelNorm()
            )),
            linear_blocks=nn.ModuleList([
                nn.ModuleDict(AttrDict(
                    linear=Linear(
                        in_features=linear_param.in_features,
                        out_features=linear_param.out_features,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    leaky_relu=nn.LeakyReLU()
                ))
                for linear_param in linear_params
            ])

        ))

    def forward(self, latents, labels=None):

        labels = self.modules.embedding_block.embedding(labels)
        outputs = self.modules.embedding_block.concat(latents, labels)
        outputs = self.modules.embedding_block.pixel_norm(outputs)

        for linear_block in self.modules.linear_blocks:
            outputs = linear_block.linear(outputs)
            outputs = linear_block.leaky_relu(outputs)

        return outputs


class SynthesisNetwork(nn.Module):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels, num_features):

        super().__init__()

        min_depth = int(np.log2(min_resolution // min_resolution))
        max_depth = int(np.log2(max_resolution // min_resolution))

        def resolution(depth): return min_resolution << depth
        def num_channels(depth): return min(max_channels, min_channels << (max_depth - depth))

        self.modules = nn.ModuleDict(AttrDict(
            conv_block=nn.ModuleDict(AttrDict(
                first=nn.ModuleDict(AttrDict(
                    leaned_constant=LearnedConstant(
                        num_channels=num_channels(min_depth),
                        resolution=resolution(min_depth)
                    ),
                    learned_noise=LearnedNoise(
                        num_channels=num_channels(min_depth)
                    ),
                    leaky_relu=nn.LeakyReLU(),
                    adaptive_instance_norm=AdaptiveInstanceNorm(
                        num_features=num_features,
                        num_channels=num_channels(min_depth),
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                )),
                second=nn.ModuleDict(AttrDict(
                    conv2d=Conv2d(
                        in_channels=num_channels(min_depth),
                        out_channels=num_channels(min_depth),
                        kernel_size=3,
                        stride=1,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    learned_noise=LearnedNoise(
                        num_channels=num_channels(min_depth)
                    ),
                    leaky_relu=nn.LeakyReLU(),
                    adaptive_instance_norm=AdaptiveInstanceNorm(
                        num_features=num_features,
                        num_channels=num_channels(min_depth),
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                ))
            )),
            conv_blocks=nn.ModuleList([
                nn.ModuleDict(AttrDict(
                    first=nn.ModuleDict(AttrDict(
                        conv_transpose2d=ConvTranspose2d(
                            in_channels=num_channels(depth) >> 1,
                            out_channels=num_channels(depth),
                            kernel_size=3,
                            stride=2,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        learned_noise=LearnedNoise(
                            num_channels=num_channels(depth)
                        ),
                        leaky_relu=nn.LeakyReLU(),
                        adaptive_instance_norm=AdaptiveInstanceNorm(
                            num_features=num_features,
                            num_channels=num_channels(depth),
                            bias=True,
                            variance_scale=1,
                            weight_scale=True
                        )
                    )),
                    second=nn.ModuleDict(AttrDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(depth),
                            out_channels=num_channels(depth),
                            kernel_size=3,
                            stride=1,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        learned_noise=LearnedNoise(
                            num_channels=num_channels(depth)
                        ),
                        leaky_relu=nn.LeakyReLU(),
                        adaptive_instance_norm=AdaptiveInstanceNorm(
                            num_features=num_features,
                            num_channels=num_channels(depth),
                            bias=True,
                            variance_scale=1,
                            weight_scale=True
                        )
                    ))
                )) for depth in range(1, max_depth + 1)
            ]),
            color_block=nn.ModuleDict(AttrDict(
                conv2d=Conv2d(
                    in_channels=num_channels(max_depth),
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    variance_scale=1,
                    weight_scale=True
                ),
                tanh=nn.Tanh()
            ))
        ))

    def forward(self, latents):

        outputs = self.modules.conv_block.first.leaned_constant(latents)
        outputs = self.modules.conv_block.first.learned_noise(outputs)
        outputs = self.modules.conv_block.first.leaky_relu(outputs)
        outputs = self.modules.conv_block.first.adaptive_instance_norm(outputs, latents)

        outputs = self.modules.conv_block.second.conv2d(outputs)
        outputs = self.modules.conv_block.second.learned_noise(outputs)
        outputs = self.modules.conv_block.second.leaky_relu(outputs)
        outputs = self.modules.conv_block.second.adaptive_instance_norm(outputs, latents)

        for conv_block in self.modules.conv_blocks:

            outputs = conv_block.first.conv_transpose2d(outputs)
            outputs = conv_block.first.learned_noise(outputs)
            outputs = conv_block.first.leaky_relu(outputs)
            outputs = conv_block.first.adaptive_instance_norm(outputs, latents)

            outputs = conv_block.second.conv2d(outputs)
            outputs = conv_block.second.learned_noise(outputs)
            outputs = conv_block.second.leaky_relu(outputs)
            outputs = conv_block.second.adaptive_instance_norm(outputs, latents)

        outputs = self.modules.color_block.conv2d(outputs)
        # outputs = self.modules.color_block.tanh(outputs)

        return outputs


class Generator(nn.Module):

    def __init__(self, mapping_network, synthesis_network):

        self.mapping_network = mapping_network
        self.synthesis_network = synthesis_network

    def forward(self, latents, labels=None):

        outputs = self.mapping_network(latents, labels)
        outputs = self.synthesis_network(outputs)

        return outputs
