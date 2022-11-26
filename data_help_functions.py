import torch
from torch.utils.data import Dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split


class RegressionDataset(Dataset):
    """Simple Regression dataset"""
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x_sample, y_sample = self.features[index], self.labels[index]
        if self.transform:
            x_sample = self.transform(x_sample)
        sample = (x_sample, y_sample)
        return sample


def make_regression_data(number_samples, number_features, noise_value=0.0, random_state=42):
    X, y = datasets.make_regression(n_samples=number_samples,
                                    n_features=number_features,
                                    random_state=random_state,
                                    shuffle=True,
                                    noise=noise_value)
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float).unsqueeze(dim=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)
    return X_train, X_test, y_train, y_test
