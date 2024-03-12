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
                 n_frames=16,
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
        slices_num_list = [x[1] for x in self.samples]
        self.image_3d_list = image_3d_list
        self.slices_num_list = slices_num_list
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
        slice_num = self.slices_num_list[img_idx]

        # 3D image load
        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_data = img_data.swapaxes(0, 2)
        img_data = np.expand_dims(img_data, axis=1)

        return img_data, slice_num, idx

def get_loaders(rank, imgstr, resolution, timesteps, skip, batch_size=1, n_gpus=1, seed=42,  cond=False):

    if imgstr == 'CHAOS':
        train_dir = os.path.join(data_location, 'CHAOS_res_128_s_16')
        # train_dir = os.path.join(data_location, 'CHAOS_res_256_s_16')
        # test_dir = os.path.join(data_location, 'CHAOS_res_128')
        if cond:
            print("here")
            timesteps *= 2  # for long generation
        # trainset = Image3DDataset(train_dir, train=True, resolution=resolution)
        trainset = Image3DDatasetCond(train_dir, train=True, resolution=resolution)
        print(len(trainset))
        # testset = Image3DDataset(train_dir, train=False, resolution=resolution)
        testset = Image3DDatasetCond(train_dir, train=False, resolution=resolution)
        print(len(testset))

    else:
        raise NotImplementedError()    

    trainset_sampler = InfiniteSampler(dataset=trainset, rank=0, num_replicas=n_gpus, seed=seed)
    # trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=4, prefetch_factor=2)
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size, pin_memory=False, num_workers=4, prefetch_factor=2)
    
    testset_sampler = InfiniteSampler(testset, num_replicas=n_gpus, rank=0, seed=seed)
    # testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=4, prefetch_factor=2)
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size, pin_memory=False, num_workers=4, prefetch_factor=2)

    return trainloader, trainloader, testloader 


