import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad

class AutoEncoder(nn.Module):
    
    def __init__(self, object_shape, repr_dim=128, dimensionality=64):
        super().__init__()

        object_shape = (1,) + object_shape
        #print(object_shape)
        #input()
        #hm

        self.repr_dim = repr_dim

        # Conv 1
        #self.fc_labels_1 = nn.Linear(num_classes, (13 * 15 * 11))
        self.conv_1 = nn.Conv3d(1, dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2))

        # Conv 2
        #self.fc_labels_2 = nn.Linear(num_classes, (7 * 8 * 6))
        self.conv_2 = nn.Conv3d(dimensionality, 2 * dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv_2_bn = nn.BatchNorm3d(2 * dimensionality)

        # Conv 3
        #self.fc_labels_3 = nn.Linear(num_classes, (3 * 4 * 3))
        self.conv_3 = nn.Conv3d(2 * dimensionality, 4 * dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 1, 2))
        self.conv_3_bn = nn.BatchNorm3d(4 * dimensionality)

        # Linear and reshape
        self.fc_1 = nn.Linear((2 * 2 * 2) * 4 * dimensionality, repr_dim)#128

        # Linear
        #self.fc_2 = nn.Linear(128, 1)
        
        # Linear and reshape
        self.fc_2 = nn.Linear(repr_dim, (2 * 2 * 2) * 4 * dimensionality)
        #self.fc_labels_1 = nn.Linear(num_classes, (2 * 2 * 2))

        # Deconv 1
        self.deconv_1 = nn.ConvTranspose3d(4 * dimensionality,
                                           2 * dimensionality,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1, 1))
        self.deconv_1_bn = nn.BatchNorm3d(2 * dimensionality)

        # Deconv 2
        #self.fc_labels_2 = nn.Linear(num_classes, (4 * 4 * 4))
        self.deconv_2 = nn.ConvTranspose3d(2 * dimensionality,
                                           dimensionality,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1, 1))
        self.deconv_2_bn = nn.BatchNorm3d(dimensionality)

        # Deconv 3
        #self.fc_labels_3 = nn.Linear(num_classes, (8 * 8 * 8))
        self.deconv_3 = nn.ConvTranspose3d(dimensionality,
                                           1,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1, 1))

        # Linear and reshape
        output_shape_dimensionality = 1
        for s in list(object_shape):
            output_shape_dimensionality *= s
        self.fc_3 = nn.Linear(16 * 16 * 16, output_shape_dimensionality)

        self.input_shape = object_shape
        self.output_shape = object_shape
        self.dimensionality = dimensionality
        #self.num_classes = num_classes
        #self.conditioning_dimensionality = conditioning_dimensionality
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))
        #self.cudaEnabled = cudaEnabled
        self.cuda()
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        out = images.view(-1, 1, 13, 15, 11)

        # Conv 1
        out = self.conv_1(out)
        out = F.leaky_relu(out, inplace=True)

        # Conv 2
        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Conv 3
        out = self.conv_3(out)
        out = self.conv_3_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Linear and reshape
        out = out.view(out.shape[0], (2 * 2 * 2) * 4 * self.dimensionality)
        out = self.fc_1(out)

        #out is currently repr_dim

        # Linear
        #out = torch.cat((out, labels), 1)
        #out = self.fc_2(out)

        return out
    
    def decode(self, code):
        # Concatenate noise and labels for input layer:

        #out = torch.cat((noise, labels), 1)
        out = code.view(-1, 1, self.repr_dim)

        # Linear and reshape brain data:
        out = self.fc_2(out)
        out = F.leaky_relu(out, inplace=True)
        out = out.view(-1, 4 * self.dimensionality, 2, 2, 2)

        # Deconv 1
        out = self.deconv_1(out)
        out = self.deconv_1_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Deconv 2
        out = self.deconv_2(out)
        out = self.deconv_2_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Deconv 3
        out = self.deconv_3(out)
        out = F.tanh(out)

        # Linear and reshape
        out = out.view(-1, 16 * 16 * 16)
        out = self.fc_3(out)

        batch_output_shape = (-1,) + self.output_shape
        out = out.view(batch_output_shape)
        return out

    def train(self, images):
        self.zero_grad()
        out, code = self(images)

        loss_fn = nn.MSELoss()
        loss = loss_fn(out, images)
        loss.backward()

        self.optimizer.step()
        return loss