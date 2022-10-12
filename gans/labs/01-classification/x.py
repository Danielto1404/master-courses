import copy
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from torchvision.datasets import ImageFolder
from torchvision.models import Inception3
from tqdm import tqdm

warnings.filterwarnings("ignore")
