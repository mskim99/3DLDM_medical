import nibabel as nib
import numpy as np

'''
image_data = []
for i in range (0, 8):
    image_data_p = nib.load('J:/test/real_0_' + str(i) + '.nii.gz').get_fdata()    
    print(image_data_p.shape)
    image_data.append(image_data_p)

image_data = np.concatenate(image_data, axis=2)
print(image_data.shape)

fake_nii = nib.Nifti1Image(image_data, None)
nib.save(fake_nii, 'J:/test/test_m.nii.gz')
'''

# Case 1 : OVERLAPPING (2)
# Resolution 128
tres = 128
z_res = 16
intv = 2
image_data = np.zeros([tres, tres, tres])
prev_idx = 0
for i in range(0, 9):
    image_data_p = nib.load('J:/test/generated_20000_' + str(i) + '.nii.gz').get_fdata()
    image_data_p = 256. * (image_data_p - image_data_p.min()) / (image_data_p.max() - image_data_p.min())
    # image_data_p = nib.load('J:/Dataset/SynthRAD2023/Task1_proc/brain_64_s_16_pd_2/mask/' + str(i) + '/57_' + str(i).zfill(4) + '_s_8.nii.gz').get_fdata()
    print(image_data_p.shape)

    cur_idx = prev_idx + z_res

    if i == 0:
        image_data[:, :, 0:z_res] = image_data_p
    else:
        image_data[:, :, prev_idx:prev_idx + intv] = (image_data_p[:, :, 0:intv] + image_data[:, :,
                                                                                   prev_idx:prev_idx + intv]) / 2
        image_data[:, :, prev_idx + intv:cur_idx] = image_data_p[:, :, intv:z_res]

    # image_data[:, :, prev_idx:cur_idx] = image_data_p

    print(prev_idx)
    print(cur_idx)

    prev_idx = cur_idx - intv

# image_data = image_data.swapaxes(0, 1)

fake_nii = nib.Nifti1Image(image_data, None)
nib.save(fake_nii, 'J:/test/generated_20000_m.nii.gz')
# nib.save(fake_nii, 'J:/test/57_grad_m.nii.gz')

'''
image_data = np.zeros([256, 256, 254])
prev_idx = 0
for i in range (0, 18):
    image_data_p = nib.load('J:/test/real_0_' + str(i) + '.nii.gz').get_fdata()
    # image_data_p = nib.load('J:/Dataset/CHAOS_nibabel_norm_res_256_s_16_pd_2/' + str(i) + '/1_' + str(i).zfill(4) + '_s_16.nii.gz').get_fdata()
    print(image_data_p.shape)

    cur_idx = prev_idx + 16

    if i == 0:
        image_data[:, :, 0:16] = image_data_p
    else:
        image_data[:, :, prev_idx:prev_idx+2] = (image_data_p[:, :, 0:2] + image_data[:, :, prev_idx:prev_idx+2]) / 2
        image_data[:, :, prev_idx+2:cur_idx] = image_data_p[:, :, 2:32]

    print(prev_idx)
    print(cur_idx)

    prev_idx = cur_idx - 2

fake_nii = nib.Nifti1Image(image_data, None)
nib.save(fake_nii, 'J:/test/real_0_m.nii.gz')
'''

# Case 2 : OVERLAPPING (8)
'''
image_data = np.zeros([128, 128, 128])

for i in range (0, 15):
    image_data_p = nib.load('J:/test/generated_105000_' + str(i) + '.nii.gz').get_fdata()
    print(image_data_p.shape)

    if i == 0:
        image_data[:, :, 0:16] = image_data_p
    else:
        image_data[:, :, (8*i):(8*i+8)] = (image_data_p[:, :, 0:8] + image_data[:, :, (8*i):(8*i+8)]) / 2
        image_data[:, :, (8*i+8):(8*i+16)] = image_data_p[:, :, 8:16]

fake_nii = nib.Nifti1Image(image_data, None)
nib.save(fake_nii, 'J:/test/generated_105000_m.nii.gz')
'''

'''
for i in range (1, 40):
    image_data_p = nib.load('J:/Dataset/CHAOS_nibabel_norm/' + str(i) + '.nii.gz').get_fdata()
    print(image_data_p.shape)
'''

# Case 4 : OVERLAPPING (2) & all direction (64)
# Resolution 126
'''
tres = 64
intv=0
image_data = np.zeros([tres, tres, tres])

image_data_ps = []
for i in range (0, 8):
    image_data_ps.append(nib.load('J:/test/generated_15000_' + str(i) + '.nii.gz').get_fdata())
    # image_data_ps.append(nib.load('J:/Dataset/HCP_1200_norm_res_64_crop2_mask_xyz_64_pd_1/' + str(i) + '/50_' + str(i).zfill(4) + '_s_xyz_64.nii.gz').get_fdata())

image_data_ps_z_accom = []
for i in range (0, 4):
    image_data_z_accom = np.zeros([int(tres/2), int(tres/2), tres-intv])
    image_data_z_accom[:, :, 0:int(tres/2)-intv] = image_data_ps[2*i][:, :, 0:int(tres/2)-intv]
    image_data_z_accom[:, :, int(tres/2):tres-intv] = image_data_ps[2*i+1][:, :, intv:int(tres/2)]
    image_data_z_accom[:, :, int(tres/2)-intv:int(tres/2)] = 0.5 * (image_data_ps[2*i][:, :, int(tres/2)-intv:int(tres/2)] + image_data_ps[2*i+1][:, :, 0:intv])
    image_data_ps_z_accom.append(image_data_z_accom)

image_data_ps_y_accom = []
for i in range (0, 2):
    image_data_y_accom = np.zeros([int(tres/2), tres-intv, tres-intv])
    image_data_y_accom[:, 0:int(tres/2)-intv, :] = image_data_ps_z_accom[2*i][:, 0:int(tres/2)-intv, :]
    image_data_y_accom[:, int(tres/2):tres-intv, :] = image_data_ps_z_accom[2*i+1][:, intv:int(tres/2), :]
    image_data_y_accom[:, int(tres/2)-intv:int(tres/2), :] = 0.5 * (image_data_ps_z_accom[2*i][:, int(tres/2)-intv:int(tres/2), :] + image_data_ps_z_accom[2*i+1][:, 0:intv, :])
    image_data_ps_y_accom.append(image_data_y_accom)

image_data[0:int(tres/2)-intv, :, :] = image_data_ps_y_accom[0][0:int(tres/2)-intv, :, :]
image_data[int(tres/2):tres-intv, :, :] = image_data_ps_y_accom[1][intv:int(tres/2), :, :]
image_data[int(tres/2)-intv:int(tres/2), :, :] = 0.5 * (image_data_ps_y_accom[0][int(tres/2)-intv:int(tres/2), :, :] + image_data_ps_y_accom[1][0:intv, :, :])

fake_nii = nib.Nifti1Image(image_data, None)
nib.save(fake_nii, 'J:/test/generated_15000_m.nii.gz')
'''

# Grid switch sampling
'''
tres = 64
s_z = 8
z_num = int(tres / s_z)
tres_p = int(tres / 2)

prev_idx = 0
img_parts_accum = []
for i in range (0, int(z_num/2)):

    img = nib.load('J:/test/generated_8750_' + str(2*i) + '.nii.gz')
    # img = nib.load('J:/Dataset/HCP_1200_norm_res_64_crop2_gs_half_xy/' + str(2*i) + '/100_' + str(2*i).zfill(4) + '_gs_half_xy.nii.gz')
    img_data = img.get_fdata()

    img_next = nib.load('J:/test/generated_8750_' + str(2*i+1) + '.nii.gz')
    # img_next = nib.load('J:/Dataset/HCP_1200_norm_res_64_crop2_gs_half_xy/' + str(2*i+1) + '/100_' + str(2*i+1).zfill(4) + '_gs_half_xy.nii.gz')
    img_next_data = img_next.get_fdata()

    cur_idx = prev_idx + s_z
    cur_idx_next = cur_idx + s_z

    print(prev_idx)
    print(cur_idx)

    slice_x = [0, tres_p, tres_p, tres_p * 2]
    slice_y = [0, tres_p, tres_p, tres_p * 2]

    img_parts = []
    img_parts.append(img_data[slice_x[0]:slice_x[1], slice_y[0]:slice_y[1], :])
    img_parts.append(img_next_data[slice_x[2]:slice_x[3], slice_y[0]:slice_y[1], :])
    img_parts.append(img_next_data[slice_x[0]:slice_x[1], slice_y[2]:slice_y[3], :])
    img_parts.append(img_data[slice_x[2]:slice_x[3], slice_y[2]:slice_y[3], :])

    img_parts_x_concat = []
    img_parts_x_concat.append(np.concatenate([img_parts[0], img_parts[1]], axis=0))
    img_parts_x_concat.append(np.concatenate([img_parts[2], img_parts[3]], axis=0))

    img_recon = np.concatenate([img_parts_x_concat[0], img_parts_x_concat[1]], axis=1)
    img_parts_accum.append(img_recon)

    img_parts = []
    img_parts.append(img_next_data[slice_x[0]:slice_x[1], slice_y[0]:slice_y[1], :])
    img_parts.append(img_data[slice_x[2]:slice_x[3], slice_y[0]:slice_y[1], :])
    img_parts.append(img_data[slice_x[0]:slice_x[1], slice_y[2]:slice_y[3], :])
    img_parts.append(img_next_data[slice_x[2]:slice_x[3], slice_y[2]:slice_y[3], :])

    img_parts_x_concat = []
    img_parts_x_concat.append(np.concatenate([img_parts[0], img_parts[1]], axis=0))
    img_parts_x_concat.append(np.concatenate([img_parts[2], img_parts[3]], axis=0))

    img_recon = np.concatenate([img_parts_x_concat[0], img_parts_x_concat[1]], axis=1)
    img_parts_accum.append(img_recon)

img_output = np.dstack(img_parts_accum)
img_output_nib = nib.Nifti1Image(img_output, None)
nib.save(img_output_nib, 'J:/test/generated_8750_m.nii.gz')
'''