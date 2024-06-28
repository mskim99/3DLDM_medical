from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.folder import make_dataset
from torchvision.io import read_video

data_location = '/data/jionkim'
from tools.data_utils import *

import nibabel as nib
from skimage.transform import resize

class Image3DDataset(Dataset):
    def __init__(self,
                 root,
                 train,
                 resolution,
                 n_frames=128,
                 skip=1,
                 fold=1,
                 use_labels=False,    # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,    # True for evaluating FVD
                 seed=42,
                 ):

        image_3d_root = osp.join(os.path.join(root))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = image_3d_root
        name = image_3d_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.classes = list(natsorted(p for p in os.listdir(image_3d_root) if osp.isdir(osp.join(image_3d_root, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(image_3d_root, class_to_idx, ('nii.gz',), is_valid_file=None)
        image_3d_list = [x[0] for x in self.samples]
        self.image_3d_list = image_3d_list
        self._use_labels = use_labels
        self._label_shape = None
        self._raw_labels = None
        self._raw_shape = [len(self.image_3d_list)] + [16, resolution, resolution]
        self.num_channels = 1
        self.return_vid = return_vid

        frames_between_clips = skip
        print(root, frames_between_clips, n_frames)
        # self.indices = [i for i in range(len(self.image_3d_list))]
        self.indices = self._select_fold(self.image_3d_list, self.path, fold, train)

        random.seed(seed)

        self.size = len(self.indices)
        print(self.size)
        self.shuffle_indices = [i for i in range(self.size)]
        random.shuffle(self.shuffle_indices)

        self._need_init = True

    def _select_fold(self, video_list, path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        img_idx = self.indices[idx]

        img_path = self.image_3d_list[img_idx]

        # 3D image load
        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_data = img_data.swapaxes(0, 2)
        img_data = np.expand_dims(img_data, axis=1)

        return img_data, idx


class Image3DDatasetCond(Dataset):
    def __init__(self,
                 root,
                 train,
                 resolution,
                 n_frames=16,
                 skip=1,
                 fold=1,
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,  # True for evaluating FVD
                 seed=42,
                 ):
        image_3d_root = osp.join(os.path.join(root))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = image_3d_root
        name = image_3d_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.classes = list(natsorted(p for p in os.listdir(image_3d_root) if osp.isdir(osp.join(image_3d_root, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(image_3d_root, class_to_idx, ('nii.gz',), is_valid_file=None)

        image_3d_list = [x[0] for x in self.samples]
        vol_num_list = [int(os.path.basename(x[0]).split('_')[0]) for x in self.samples] # Extract volume number from path
        vol_num_list_unique = list(set(vol_num_list))
        slices_num_list = [x[1] for x in self.samples]

        # Bind image_3d_list and slice_num_list with same data
        image_3d_list_c = []
        slices_num_list_c = []
        for vn_l in vol_num_list_unique:
            img_3d_list = []
            sls_num_list = []
            for idx, val in enumerate(vol_num_list):
                if val == vn_l:
                    img_3d_list.append(image_3d_list[idx])
                    sls_num_list.append(slices_num_list[idx])

            image_3d_list_c.append(img_3d_list)
            slices_num_list_c.append(sls_num_list)

        self.image_3d_list = image_3d_list_c
        self.vol_num_list = vol_num_list
        self.slices_num_list = slices_num_list_c
        self._use_labels = use_labels
        self._label_shape = None
        self._raw_labels = None
        self._raw_shape = [len(self.image_3d_list)] + [16, resolution, resolution]
        self.num_channels = 1
        self.return_vid = return_vid

        frames_between_clips = skip
        print(root, frames_between_clips, n_frames)
        if train:
            self.indices = [i for i in range(0, int(len(self.image_3d_list) * 0.7))]
        else:
            self.indices = [i for i in range(int(len(self.image_3d_list) * 0.7), len(self.image_3d_list))]
        # self.indices = self._select_fold(self.image_3d_list, self.path, fold, train)

        random.seed(seed)

        self.size = len(self.indices)
        print(self.size)
        self.shuffle_indices = self.indices

        self._need_init = True

    def _select_fold(self, video_list, path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_idx = self.indices[idx]

        img_paths = self.image_3d_list[img_idx]
        slice_nums = self.slices_num_list[img_idx]

        # 3D image load
        img_datas = []
        for img_path in img_paths:
            img = nib.load(img_path)
            img_data = img.get_fdata()
            img_data = img_data.swapaxes(0, 2)
            img_data = np.expand_dims(img_data, axis=1)
            img_datas.append(img_data)

        return img_datas, slice_nums, idx

class Image3DDatasetCondMask(Dataset):
    def __init__(self,
                 root,
                 root_mask,
                 train,
                 resolution,
                 n_frames=16,
                 skip=1,
                 fold=1,
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,  # True for evaluating FVD
                 seed=42,
                 ):
        image_3d_root = osp.join(os.path.join(root))
        image_3d_root_mask = osp.join(os.path.join(root_mask))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = image_3d_root
        self.path_mask = image_3d_root_mask
        name = image_3d_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.classes = list(natsorted(p for p in os.listdir(image_3d_root) if osp.isdir(osp.join(image_3d_root, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(image_3d_root, class_to_idx, ('nii.gz',), is_valid_file=None)
        self.samples_mask = make_dataset(image_3d_root_mask, class_to_idx, ('nii.gz',), is_valid_file=None)

        image_3d_list = [x[0] for x in self.samples]
        image_3d_list_mask = [x[0] for x in self.samples_mask]
        vol_num_list = [int(os.path.basename(x[0]).split('_')[0]) for x in self.samples] # Extract volume number from path
        vol_num_list_unique = list(set(vol_num_list))
        slices_num_list = [x[1] for x in self.samples]

        # Bind image_3d_list and slice_num_list with same data
        image_3d_list_c = []
        image_3d_list_mask_c = []
        slices_num_list_c = []
        for vn_l in vol_num_list_unique:
            img_3d_list = []
            img_m_3d_list = []
            sls_num_list = []
            for idx, val in enumerate(vol_num_list):
                if val == vn_l:
                    img_3d_list.append(image_3d_list[idx])
                    img_m_3d_list.append(image_3d_list_mask[idx])
                    sls_num_list.append(slices_num_list[idx])

            image_3d_list_mask_c.append(img_m_3d_list)
            image_3d_list_c.append(img_3d_list)
            slices_num_list_c.append(sls_num_list)

        self.image_3d_list = image_3d_list_c
        self.image_3d_list_mask = image_3d_list_mask_c
        self.vol_num_list = vol_num_list
        self.slices_num_list = slices_num_list_c
        self._use_labels = use_labels
        self._label_shape = None
        self._raw_labels = None
        self._raw_shape = [len(self.image_3d_list)] + [16, resolution, resolution]
        self.num_channels = 1
        self.return_vid = return_vid

        frames_between_clips = skip
        print(root, frames_between_clips, n_frames)
        if train:
            self.indices = [i for i in range(0, int(len(self.image_3d_list) * 0.7))]
        else:
            self.indices = [i for i in range(int(len(self.image_3d_list) * 0.7), len(self.image_3d_list))]
        # self.indices = self._select_fold(self.image_3d_list, self.path, fold, train)

        random.seed(seed)

        self.size = len(self.indices)
        print(self.size)
        self.shuffle_indices = self.indices

        self._need_init = True

    def _select_fold(self, video_list, path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_idx = self.indices[idx]

        img_paths = self.image_3d_list[img_idx]
        img_paths_mask = self.image_3d_list_mask[img_idx]
        slice_nums = self.slices_num_list[img_idx]

        # 3D image load
        img_datas = []
        img_datas_mask = []

        for idx in range (0, img_paths.__len__()):
            img = nib.load(img_paths[idx])
            img_data = img.get_fdata()
            img_data = img_data.swapaxes(0, 2)
            img_data = np.expand_dims(img_data, axis=1)

            img_mask = nib.load(img_paths_mask[idx])
            img_data_mask = img_mask.get_fdata()
            img_data_mask = img_data_mask.swapaxes(0, 2)
            img_data_mask = np.expand_dims(img_data_mask, axis=1)

            img_datas.append(img_data)
            img_datas_mask.append(img_data_mask)

        return img_datas, img_datas_mask, slice_nums, idx

def get_loaders(rank, imgstr, resolution, timesteps, skip, batch_size=1, n_gpus=1, seed=42,  cond=False):

    if imgstr == 'CHAOS':
        train_dir = os.path.join(data_location, 'CHAOS_res_128_s_16_c')
    elif imgstr == 'CHAOS_OL_0_5':
        train_dir = os.path.join(data_location, 'CHAOS_res_128_s_16_ol_0_5')
    elif imgstr == 'CHAOS_PD_2':
        train_dir = os.path.join(data_location, 'CHAOS_res_128_s_16_pd_2')
        train_dir_mask = os.path.join(data_location, 'CHAOS_res_128_s_16_pd_2_mask')
    elif imgstr == 'CHAOS_32_PD_2':
        train_dir = os.path.join(data_location, 'CHAOS_res_128_s_32_pd_2')
        train_dir_mask = os.path.join(data_location, 'CHAOS_res_128_s_32_pd_2_mask')
    elif imgstr == 'CHAOS_PD_1_RES_256':
        train_dir = os.path.join(data_location, 'CHAOS_res_256_s_16_pd_1')
    elif imgstr == 'CHAOS_PD_2_RES_256':
        train_dir = os.path.join(data_location, 'CHAOS_res_256_s_16_pd_2')
    elif imgstr == 'HCP_PD_2_RES_256':
        train_dir = os.path.join(data_location, 'HCP_res_256_s_16_pd_2')
    elif imgstr == 'HCP':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_s_16')
    elif imgstr == 'HCP_res_64':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_64')
    elif imgstr == 'HCP_OL_0_5':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_s_16_ol_0_5')
    elif imgstr == 'HCP_PD_2':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_s_16_pd_2')
        train_dir_mask = os.path.join(data_location, 'HCP_1200_norm_res_128_s_16_pd_2_mask')
    elif imgstr == 'HCP_PD_2_ALL_DIR':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_s_xyz_64_pd_2')
    elif imgstr == 'HCP_32_PD_2':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_s_32_pd_2')
    elif imgstr == 'HCP_XZ_SWP_PD_2':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_xz_swp_s_16_pd_2')
    elif imgstr == 'HCP_YZ_SWP_PD_2':
        train_dir = os.path.join(data_location, 'HCP_1200_norm_res_128_yz_swp_s_16_pd_2')
    elif imgstr == 'CT_ORG':
        train_dir = os.path.join(data_location, 'CT-ORG_res_128_norm_s_16')
    else:
        raise NotImplementedError()

    if cond:
        print("here")
        timesteps *= 2  # for long generation

    trainset = Image3DDatasetCond(train_dir, train=True, resolution=resolution)
    print(len(trainset))
    testset = Image3DDatasetCond(train_dir, train=False, resolution=resolution)
    print(len(testset))

    '''
    trainset = Image3DDatasetCondMask(train_dir, train_dir_mask, train=True, resolution=resolution)
    print(len(trainset))
    testset = Image3DDatasetCondMask(train_dir, train_dir_mask, train=False, resolution=resolution)
    print(len(testset))
    '''

    trainset_sampler = InfiniteSampler(dataset=trainset, rank=0, num_replicas=n_gpus, seed=seed)
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size, pin_memory=False, num_workers=4, prefetch_factor=2)
    testset_sampler = InfiniteSampler(testset, num_replicas=n_gpus, rank=0, seed=seed)
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size, pin_memory=False, num_workers=4, prefetch_factor=2)


    return trainloader, trainloader, testloader 


