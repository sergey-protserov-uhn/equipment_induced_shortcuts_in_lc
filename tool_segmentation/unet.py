# Adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch

# Original license included below

# BEGIN ORIGINAL LICENSE

# MIT License

# Copyright (c) 2019 mateuszbuda

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# END ORIGINAL LICENSE

from itertools import pairwise

import torch as t
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        *,
        n_input_channels,
        n_output_channels,
        n_first_block_output_channels,
        depth,
        pool_class,
        activation_class,
        activation_args,
        activation_kwargs,
        dropout_p,
    ):
        super().__init__()

        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.n_first_block_output_channels = n_first_block_output_channels
        self.depth = depth
        self.pool_class = pool_class
        self.activation_class = activation_class
        self.activation_args = activation_args
        self.activation_kwargs = activation_kwargs
        self.dropout_p = dropout_p

        channel_counts = [self.n_input_channels] + [
            self.n_first_block_output_channels * (2**i)
            for i in range(self.depth)
        ]

        self.encoder_blocks = nn.ModuleList(
            [
                self.get_block(
                    n_input_channels=n_block_input_channels,
                    n_output_channels=n_block_output_channels,
                    activation_class=self.activation_class,
                    activation_args=self.activation_args,
                    activation_kwargs=self.activation_kwargs,
                    dropout_p=self.dropout_p,
                )
                for n_block_input_channels, n_block_output_channels in pairwise(
                    channel_counts,
                )
            ],
        )
        self.pools = nn.ModuleList(
            [
                self.pool_class(kernel_size=2, stride=2)
                for _ in range(self.depth - 1)
            ],
        )
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    n_upconv_input_channels,
                    n_upconv_output_channels,
                    kernel_size=2,
                    stride=2,
                    output_padding=p,
                )
                for (
                    n_upconv_input_channels,
                    n_upconv_output_channels,
                ), p in zip(
                    pairwise(
                        reversed(channel_counts[1:]),
                    ),
                    ((0, 1), 0, 0),
                    strict=True,
                )
            ],
        )
        self.decoder_blocks = nn.ModuleList(
            [
                self.get_block(
                    n_input_channels=n_block_input_channels,
                    n_output_channels=n_block_output_channels,
                    activation_class=self.activation_class,
                    activation_args=self.activation_args,
                    activation_kwargs=self.activation_kwargs,
                    dropout_p=self.dropout_p,
                )
                for n_block_input_channels, n_block_output_channels in pairwise(
                    reversed(channel_counts[1:])
                )
            ],
        )

        self.final_conv = nn.Conv2d(
            channel_counts[1],
            self.n_output_channels,
            kernel_size=1,
        )

    def encode(self, x):
        encoder_block_outputs = []
        for encoder_block, pool in zip(
            self.encoder_blocks,
            nn.ModuleList([nn.Identity()]) + self.pools,
            strict=True,
        ):
            x = encoder_block(pool(x))
            encoder_block_outputs.append(x)
        return encoder_block_outputs

    def forward(self, x):
        encoder_block_outputs = self.encode(x)
        x = encoder_block_outputs[-1]
        for (
            upconv,
            decoder_block,
            encoder_block_output,
        ) in zip(
            self.upconvs,
            self.decoder_blocks,
            reversed(encoder_block_outputs[:-1]),
            strict=True,
        ):
            x = decoder_block(
                t.cat(
                    (upconv(x), encoder_block_output),
                    dim=1,
                )
            )

        return self.final_conv(x)

    @staticmethod
    def get_block(
        *,
        n_input_channels,
        n_output_channels,
        activation_class,
        activation_args,
        activation_kwargs,
        dropout_p,
    ):
        return nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                n_output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(n_output_channels),
            nn.Dropout2d(dropout_p),
            activation_class(*activation_args, **activation_kwargs),
            nn.Conv2d(
                n_output_channels,
                n_output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(n_output_channels),
            nn.Dropout2d(dropout_p),
            activation_class(*activation_args, **activation_kwargs),
        )
