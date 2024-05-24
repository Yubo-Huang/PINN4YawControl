# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_output=3, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(4, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.estimator = nn.Linear(num_features, num_output)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.estimator(out)
        return out
    
class ALMLossFun(nn.Module):
    def __init__(self) -> None:
        super(ALMLossFun).__init__()
        self.area = torch.tensor([159.89, 178.40, 168.68, 149.28, 128.99, 108.96, 77.68])
        self.chord = torch.tensor([10.0, 9.40, 8.68, 7.28, 6.99, 5.96, 3.68])
        
    def forward(self, equivalent_wind, 
                body_velocity, 
                normal_vec, 
                tangent_vec, 
                normal_coef,
                tangent_coef,
                moment_coef,
                normal_force,
                tangent_force,
                moment_torque):
        tensor_shape = body_velocity.shape
        # batch_size, num_blades, num_elements = tensor_shape[0], tensor_shape[1], tensor_shape[2]
        
        # equal_wind_expand = equivalent_wind.unsqueeze(1).expand(-1, 7, -1)
        # equal_wind_expand = equal_wind_expand.unsqueeze(1).expand(-1, 3, -1, -1)
        equal_wind_expand = torch.unsqueeze(torch.unsqueeze(equivalent_wind, dim=1), dim=2)
        
        relative_wind = torch.sub(equal_wind_expand, body_velocity)
        normal_velocity = torch.sum(torch.mul(relative_wind, normal_vec), dim=3)
        tangent_velocity = torch.sum(torch.mul(relative_wind, tangent_vec), dim=3)
        relative_wind_squared = torch.add(torch.pow(normal_velocity, 2), torch.pow(tangent_velocity, 2))
        
        # np_equivalent_wind = equivalent_wind.numpy()
        # np_equal_wind_expand = np.tile(np_equivalent_wind[:, np.newaxis, np.newaxis, :], (1, 3, 7, 1))
        # np_relative_wind = np_equal_wind_expand - body_velocity.numpy()
        # np_normal_velocity = np.sum(np_relative_wind * normal_vec.numpy(), axis=3)
        # np_tangent_velocity = np.sum(np_relative_wind * tangent_vec.numpy(), axis=3)
        # np_relative_wind_squared = np_normal_velocity**2 + np_tangent_velocity**2
        # print('Numpy:')
        # print(np_relative_wind_squared)
        # print('Pytorch:')
        # print(relative_wind_squared.numpy())
        # print(np.sum(np_relative_wind - relative_wind.numpy()))
        # print(np.sum(np_relative_wind_squared - relative_wind_squared.numpy()))
        
        normal_para = torch.mul(torch.mul(0.5, normal_coef), self.area) 
        equal_normal_force = torch.mul(normal_para, relative_wind_squared)
        tangent_para = torch.mul(torch.mul(0.5, tangent_coef), self.area)
        equal_tangent_force = torch.mul(tangent_para, relative_wind_squared)
        moment_para = torch.mul(torch.mul(torch.mul(0.5, moment_coef), self.chord), self.area) 
        equal_moment_torque = torch.mul(moment_para, relative_wind_squared)  
        
        # np_equal_normal_force = 0.5 * normal_coef.numpy() * self.area.numpy() * relative_wind_squared.numpy()
        # np_equal_tangent_force = 0.5 * tangent_coef.numpy() * self.area.numpy() * relative_wind_squared.numpy()
        # np_equal_moment_torque = 0.5 * moment_coef.numpy() * self.chord.numpy() * self.area.numpy() * relative_wind_squared.numpy()
        
        # print(np.sum(equal_normal_force.numpy() - np_equal_normal_force))
        # print(np.sum(equal_tangent_force.numpy() - np_equal_tangent_force))
        # print(np.sum(equal_moment_torque.numpy() - np_equal_moment_torque))
        ALMLoss = torch.add(torch.add(F.mse_loss(equal_normal_force, normal_force), F.mse_loss(equal_tangent_force, tangent_force)), F.mse_loss(equal_moment_torque, moment_torque))
        
        # equal_wind_expand = torch.unsqueeze(torch.unsqueeze(equivalent_wind, dim=1), dim=2)
        # relative_wind = equal_wind_expand - body_velocity
        # normal_velocity = torch.sum(relative_wind*normal_vec, dim=1)
        # tangent_velocity = torch.sum(relative_wind*tangent_vec, dim=1)
        # relative_wind_squared = (normal_velocity**2) * (tangent_velocity**2)
        
        # normal_para = 0.5 * normal_coef * self.area 
        # equal_normal_force = normal_para * relative_wind_squared
        # tangent_para = 0.5 * tangent_coef * self.area
        # equal_tangent_force =  tangent_para * relative_wind_squared
        # moment_para = 0.5 * moment_coef * self.chord * self.area 
        # equal_moment_torque = moment_para * relative_wind_squared  
        
        # ALMLoss = F.mse_loss(equal_normal_force, normal_force) + F.mse_loss(equal_tangent_force, tangent_force) + F.mse_loss(equal_moment_torque, moment_torque)
        
        return ALMLoss
        
         