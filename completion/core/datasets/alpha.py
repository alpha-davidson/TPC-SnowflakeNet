import torch
import numpy as np
import json
from .utils import Compose

def collate_fn(batch):

    experiment = []
    data = {}

    for sample in batch:
        experiment.append(sample[0])
        _data = sample[1]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        if not isinstance(v, torch.Tensor):
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v.copy()).float()
            else:
                v = torch.Tensor(v)
        data[k] = torch.stack(v, 0)

    return experiment, data


class ALPhADataset(torch.utils.data.dataset.Dataset):
    def __init__(self, config, file_list, transforms=None):
        self.config = config
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.config['required_items']:
            file_path = sample[f'{ri}_path']
            data[ri] = np.load(file_path)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['experiment'], data
    

class ALPhaDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.dataset_categories = []
        with open(config.dataset.category_file_path, 'r') as f:
            self.dataset_categories = json.loads(f.read())
        if not self.config.include_o:
            self.dataset_categories = self.dataset_categories[1:]

    def get_datset(self, subset):

        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()

        return ALPhADataset({'required_items' : ['partial_cloud', 'gt_cloud'],
                             'shuffle' : subset != 'test'},
                              file_list, transforms)

    def _get_file_list(self, subset):

        file_list = []

        for dc in self.dataset_categories:
            print(f"Collecting {dc['experiment']} files")
            samples = dc[subset]

            for s in samples:
                for c in ['center', 'rand', 'down']:
                    file_list.append({
                        'experiment' : dc['experiment'],
                        'partial_cloud_path' : self.config.dataset.partial_cloud_path % (subset, s, c),
                        'gt_cloud_path' : self.config.dataset.gt_cloud_path % (subset, s)
                    })

        return file_list
    
    def _get_transforms(self):
        return Compose([{
                'callback': 'DownUpSamplePoints',
                'parameters': {
                    'n_points': self.config.dataset.partial_points
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'DownUpSamplePoints',
                'parameters': {
                    'n_points': self.config.dataset.complete_points
                },
                'objects': ['gt_cloud']
            }, {
                'callback': 'MinMaxDownScale',
                'parameters': {'config': self.config},
                'objects': ['partial_cloud', 'gt_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gt_cloud']
            }])