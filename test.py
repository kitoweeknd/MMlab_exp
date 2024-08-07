import torch
in_channels = [2, 3, 5, 7]
scales = [340, 170, 84, 43]
inputs = [torch.rand(1, c, s, s)
               for c, s in zip(in_channels, scales)]
print(inputs[0].shape)
print(inputs[1].shape)
print(inputs[2].shape)
print(inputs[3].shape)

# self = ChannelMapper(in_channels, 11, 3).eval()
# outputs = self.forward(inputs)
print(inputs)