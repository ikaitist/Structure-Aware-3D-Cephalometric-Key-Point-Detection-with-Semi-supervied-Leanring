import torch
import matplotlib.pyplot as plt

# 指定保存的张量文件路径
save_path = "myTensor3.pth"

# 使用 torch.load() 加载保存的张量
heatmap = torch.load(save_path)
# heatmap_slice = heatmap[0, 3, :, :, 20]
# heatmap_grayscale = 1 - heatmap_slice
# indices = torch.where(heatmap > 1)
#
# # 获取对应的行列坐标
# x_indices = indices[2]
# y_indices = indices[3]
# z_indices = indices[4]
indices = torch.where(heatmap > 0.99)
print(indices)
print(heatmap[heatmap > 0.99])
heatmap_slice = heatmap[1, 4, :, :, 10]
heatmap_grayscale = 1 - heatmap_slice
#
# # 打印位置信息
# for y in range(5):
#     for i in range(len(x_indices)):
#         print(print("位置：({}, {}, {})，值：{}".format(x_indices[i], y_indices[i], z_indices[i], heatmap[0,y, x_indices[i], y_indices[i], z_indices[i]])))
#     # 可视化热力图


plt.imshow(heatmap_grayscale.squeeze().detach().numpy(), cmap='gray' ,vmin=0, vmax=1)  # 使用灰度色彩图
plt.imshow()
plt.colorbar()  # 添加颜色条

# 显示图像
plt.show()
