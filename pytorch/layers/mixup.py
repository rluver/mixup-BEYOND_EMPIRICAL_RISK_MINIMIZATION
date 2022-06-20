import torch
import torch.nn as nn


class MixUp(nn.Module):
    def __init__(self, alpha1=0.2, alpha2=0.2):
        super(MixUp, self).__init__()
        self.alpha1=alpha1
        self.alpha2=alpha2

    def forward(self, data1, data2):
        images_one, labels_one = data1
        images_two, labels_two = data2
        batch_size = images_one.shape[0]

        beta = torch.distributions.beta.Beta(self.alpha2, self.alpha1).expand(batch_shape=(batch_size,)).sample()
        x_l = beta.reshape(-1, 1, 1, 1)
        y_l = beta.reshape(-1, 1)
        
        new_images = images_one * x_l + images_two * (1 - x_l)
        new_labels = labels_one * y_l + labels_two * (1 - y_l)
        
        return new_images, new_labels
