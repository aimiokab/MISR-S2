import matplotlib

from tasks.srdiff import SRDiffTrainer
from utils.dataloader import BreizhSRDataset

matplotlib.use('Agg')

class SRDiffSat(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = BreizhSRDataset