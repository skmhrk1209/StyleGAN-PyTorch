import torch
import tensorboardX as tbx


class Dict(dict):

    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


class SummaryWriter(tbx.SummaryWriter):

    def add_images(self, main_tag, tag_images_dict, global_step=None,
                   walltime=None, dataformats="NCHW", mean=None, std=None):

        def unnormalize(images, mean, std):
            if std:
                std = torch.Tensor(std).to(images.device)
                images *= std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            if mean:
                mean = torch.Tensor(mean).to(images.device)
                images += mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return images

        for tag, images in tag_images_dict.items():
            self.file_writer.add_summary(
                summary=tbx.summary.image(
                    tag=f"{main_tag}/{tag}",
                    tensor=unnormalize(images, mean, std),
                    dataformats=dataformats
                ),
                global_step=global_step,
                walltime=walltime
            )
