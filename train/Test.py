from model_test.model_hourglass import PoseNet
# from model_test.model_hourglass_2d import PomseNet
from model_test.StackedHourGlass import StackedHourGlass
import torch

# image_ct = sitk.ReadImage(image_path, sitk.sitkInt16)
# image_array = sitk.GetArrayFromImage(image_ct)
# image_array = image_array.transpose((2, 1, 0))
# image_size = image_array.shape
# scale = [config.image_size[i] / image_size[i] for i in range(3)]
# image_array = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
# image_resized = F.interpolate(image_array, size=config.image_size, mode='trilinear', align_corners=True)
# image_resized = image_resized.squeeze(0)

# test_input = torch.randn(1, 3, 256, 256)
# model = StackedHourGlass(
#     nChannels= 256,
#     nStack= 2,
#     nModules= 2,
#     numReductions= 4,
#     nJoints=24
# ).to('cuda')
# test_input = torch.randn(1,3, 256, 256)
# model = PoseNet(
#     inp_dim=3,
#     oup_dim=24,
#     nstack=2
# ).to('cuda')
# nstack = 2
# test_input = torch.randn(1, 1, 256,256, 256)
# model = PoseNet(
#     in_channels=1,
#     out_channels=24,
#     nstack=2
# ).to('cuda')
# model.eval()
# test_input=test_input.to('cuda')
# with torch.no_grad():
#     output = model(test_input)
# for i in range(nstack):
#     single_output = output[:,i,:,:,:]
#     print(single_output.shape)
# print(output.shape)
test_input = torch.randn(1, 2,3, 20,20,20)
for i in range(test_input.size(1)):
    input = test_input[:,i,:,:,:,:]
    print(input.shape)

idx = torch.argmax(test_input,dim=1)
print(idx.shape)
# print(idx)
