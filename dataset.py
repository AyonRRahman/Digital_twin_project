import torch
from torch.utils.data import Dataset
import pandas as pd
import os


# Define a seed for model re-implementation
torch.manual_seed(42)


class DigitalTwinDataset(Dataset):
    '''
        the dataset classs takes a directory with following structure:
        root_dir:
                -no_screw
                -with_screw
                    -screw_1
                    -screw_2
                    -screw_3
    '''

    def __init__(self, root_dir, sample_length=500, device=torch.device("cpu"), shuffle=True):

        super(DigitalTwinDataset, self).__init__()
        self.root_dir = root_dir
        self.sample_length = sample_length
        self.device = device
        self.shuffle = shuffle

        self.files = self._get_files()

        self.samples = self._get_samples()

        self.data, self.labels = self._get_data()
        if self.shuffle:

            # Shuffle the data and labels using the same permutation
            if len(self.data) != len(self.labels):
                raise ValueError("Data and labels must have the same length.")

            self.permutation = torch.randperm(len(self.data))
            self.data = [self.data[i] for i in self.permutation]
            self.labels = [self.labels[i] for i in self.permutation]

    def _get_files(self):
        root_dir = self.root_dir
        files = {}
        no_screw_list = os.listdir(os.path.join(root_dir, 'no_screw'))
        files['no_screw'] = [os.path.join(
            root_dir+'no_screw', x) for x in no_screw_list]
        with_screw_folders = os.listdir(root_dir+'with_screw/')

        for folder in with_screw_folders:
            with_screw_list = []

            this_screw_list = (os.listdir(root_dir+f'with_screw/{folder}'))
            this_screw_list = [os.path.join(
                root_dir+f"with_screw/{folder}", x) for x in this_screw_list]
            with_screw_list = with_screw_list+this_screw_list
            files[folder] = with_screw_list

        return files

    def _get_samples(self):

        samples = {x: [] for x in self.files.keys()}

        for folder in self.files.keys():

            for files in self.files[folder]:

                df = pd.read_csv(files)
                x1 = torch.tensor(df['X_1 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y1 = torch.tensor(df['Y_1 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                x2 = torch.tensor(df['X_2 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y2 = torch.tensor(df['Y_2 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                x3 = torch.tensor(df['X_3 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y3 = torch.tensor(df['Y_3 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)

                data = torch.cat((x1, y1, x2, y2, x3, y3), 1)

                number_of_rows, _ = data.shape

                for x in range(0, number_of_rows, self.sample_length):
                    if x+self.sample_length <= number_of_rows:
                        samples[folder].append(data[x:x+self.sample_length])

        return samples

    def _get_data(self):
        data = []
        labels = []
        for label in self.samples:
            for labeled_data in self.samples[label]:
                if label == 'no_screw':
                    labels.append(torch.tensor(0,  dtype=torch.float64))
                elif label == 'screw_2':
                    labels.append(torch.tensor(2, dtype=torch.float64))
                elif label == 'screw_1':
                    labels.append(torch.tensor(1, dtype=torch.float64))
                else:
                    labels.append(torch.tensor(3, dtype=torch.float64))

                data.append(labeled_data)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        return sample


class AutoEncoderDataset(Dataset):
    '''
        the dataset classs takes a directory with following structure:
        root_dir:
                -no_screw
                -with_screw
                    -screw_1
                    -screw_2
                    -screw_3
    '''

    def __init__(self, root_dir, sample_length=500, device=torch.device("cpu"), shuffle=True):

        super(AutoEncoderDataset, self).__init__()
        self.root_dir = root_dir
        self.sample_length = sample_length
        self.device = device
        self.shuffle = shuffle

        self.files = self._get_files()
        self.samples = self._get_samples()

        self.data, self.labels = self._get_data()
        if self.shuffle:

            # Shuffle the data and labels using the same permutation
            if len(self.data) != len(self.labels):
                raise ValueError("Data and labels must have the same length.")

            self.permutation = torch.randperm(len(self.data))
            self.data = [self.data[i] for i in self.permutation]
            self.labels = [self.labels[i] for i in self.permutation]

    def _get_files(self):
        root_dir = self.root_dir
        files = {}
        no_screw_list = os.listdir(os.path.join(root_dir, 'no_screw'))
        files['no_screw'] = [os.path.join(
            root_dir+'no_screw', x) for x in no_screw_list]
        with_screw_folders = os.listdir(root_dir+'with_screw/')

        for folder in with_screw_folders:
            with_screw_list = []

            this_screw_list = (os.listdir(root_dir+f'with_screw/{folder}'))
            this_screw_list = [os.path.join(
                root_dir+f"with_screw/{folder}", x) for x in this_screw_list]
            with_screw_list = with_screw_list+this_screw_list
            files[folder] = with_screw_list

        return files

    def _get_samples(self):

        samples = {x: [] for x in self.files.keys()}

        for folder in self.files.keys():

            for files in self.files[folder]:

                df = pd.read_csv(files)
                x1 = torch.tensor(df['X_1 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y1 = torch.tensor(df['Y_1 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                x2 = torch.tensor(df['X_2 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y2 = torch.tensor(df['Y_2 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                x3 = torch.tensor(df['X_3 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y3 = torch.tensor(df['Y_3 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)

                data = torch.cat((x1, y1, x2, y2, x3, y3), 1)

                number_of_rows, _ = data.shape

                for x in range(0, number_of_rows, self.sample_length):
                    if x+self.sample_length <= number_of_rows:
                        samples[folder].append(data[x:x+self.sample_length])

        return samples

    def _get_data(self):
        data = []
        labels = []
        for label in self.samples:
            for labeled_data in self.samples[label]:
                if label == 'no_screw':
                    labels.append(torch.tensor(0,  dtype=torch.float64))
                elif label == 'screw_2':
                    labels.append(torch.tensor(2, dtype=torch.float64))
                elif label == 'screw_1':
                    labels.append(torch.tensor(1, dtype=torch.float64))
                else:
                    labels.append(torch.tensor(3, dtype=torch.float64))

                data.append(labeled_data)

                data.append(labeled_data)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        return sample


class TestSet(Dataset):
    def __init__(self, data_dict):

        super(TestSet, self).__init__()
        self.data_dict = data_dict
        self.data, self.labels = self._get_data()
        print(
            f"Dataset with {len(self.data)} data and {len(self.labels)} label")


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA (GPU) is available.")
    else:
        device = torch.device("cpu")
        print("CUDA (GPU) is not available. Using CPU.")

    # Trainset = AutoEncoderDataset('/home/ayon/Desktop/Digital Twin data/new_data/AutoEncoder_data/Train/', device=device)
    Trainset = DigitalTwinDataset(
        '/home/ayon/Desktop/Digital Twin data/new_data/AutoEncoder_data/Test/', device=device, shuffle=False)
    # print(Trainset.files)
    # print(Trainset.samples)

    labels = (Trainset.labels)
    print([x.item() for x in labels])
