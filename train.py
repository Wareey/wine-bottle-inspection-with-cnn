import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms,datasets
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

