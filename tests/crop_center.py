#
# import SimpleITK as sitk
#
# # Read the image
# image = sitk.ReadImage('/data/8T/lyh/data/data_K/Train/6408/6408.nii')
# image_size = image.GetSize()
# # Define the center and patch size
# center = (512, 512, 0)
#
# patch_size = (200, 200, 200)
#
# # Calculate the starting index
# start = [int(center[i] - patch_size[i] // 2) for i in range(3)]
# for i in range(3):
#     if start[i] < 0:
#         start[i] = 0  # 起始点不能小于 0
#     if start[i] + patch_size[i] > image_size[i]:
#         patch_size = list(patch_size)  # 转为可修改列表
#         start[i] = image_size[i] - patch_size[i]  # 调整裁剪大小
# # Use SimpleITK's RegionOfInterest function to crop the patch
# patch = sitk.RegionOfInterest(image, size=patch_size, index=start)
#
# # Output patch information
# print(patch.GetSize())  # (patch_size, patch_size, patch_size)
# sitk.WriteImage(patch, f'/data/8T/lyh/data/{center}_{patch_size[0]}_center.nii')


