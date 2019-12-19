import argparse
from process.data_fusion import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart

run_check_train_data()