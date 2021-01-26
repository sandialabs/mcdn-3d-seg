import os

import numpy as np

import ctseg.ctutil.norm


# @todo(Tyler) is this used anywhere or can we get rid of it?
def normalize1(raw, key, data_dir, norm_data_dir):
    eps = 1e-7
    norm = raw - np.min(raw) + eps
    lognorm = np.log(norm)
    norm = lognorm/np.mean(lognorm)
    new_name = norm_data_dir + key + ".npy"
    print("saving to ", new_name)
    print(norm.shape)
    np.save(new_name, norm.astype(np.float32))


def norm_log(data, dtype=np.float32):
    # copy data and cast as float
    data = data.astype(dtype)
    data -= np.min(data)
    mean = np.mean(data)
    if mean:
        data /= mean
    return np.log(data + 10 ** -7)


def find_q_for_tgt(tgt_file):
    print(tgt_file)
    tgt = np.load(tgt_file)
    count0 = tgt[:, :, :, 0].sum()
    q = 100. * count0 / np.prod(tgt[:, :, :, 0].shape)
    print(q)
    return q


def magic_norm_log(raw, q, dtype=np.float32):
    '''
    Shifts min to 0
    Picks percentile of the first target (usually air).
    By dividing by the value at that percentile
    '''
    raw = raw.astype(dtype)
    norm = raw - raw.min()
    magic_num = np.percentile(norm, q)
    print("MAGIC NUM", magic_num)
    print("Q", q)
    norm = norm / magic_num
    return np.log(norm + 1e-7)


class BaseNormalize():
    def __init__(self, save_dir, targets_dir):
        self.meta_file_path = os.path.join(save_dir, self.__class__.__name__ + ".npy")
        self.prepare_metadata(targets_dir=targets_dir)

    @property
    def cls_name(self):
        return self.__class__.__name__

    def prepare_metadata(self, targets_dir):
        self.metadata = self.load_metadata()
        if self.metadata:
            print("Using saved metadata")
            return
        print("No saved metadata found")
        self.init_metadata(targets_dir)
        if self.metadata:
            print("Saving generated metadata")
            self.save_metadata()

    def init_metadata(self, targets_dir):
        pass

    def load_metadata(self):
        try:
            return np.load(self.meta_file_path).item()
        except FileNotFoundError:
            return False

    def save_metadata(self):
        np.save(self.meta_file_path, self.metadata)

    def normalize(self, raw):
        raise NotImplementedError("Subclass must implement normalize()")

    @staticmethod
    def check_npy_valid(array):
        if np.isnan(array).any():
            raise ValueError('nan in array')
        elif np.isinf(array).any():
            raise ValueError('inf in array')
    
    def normalize_file(self, inpath, cache_dir):
        cache_dir = os.path.join(cache_dir, self.__class__.__name__)
        os.makedirs(cache_dir, mode=0o775, exist_ok=True)
        basename = os.path.basename(inpath)
        newname = os.path.join(cache_dir, basename)
        if not os.path.exists(newname):
            raw = np.load(inpath, fix_imports=True, encoding='bytes')
            self.check_npy_valid(raw)
            norm = self.normalize(raw)
            self.check_npy_valid(norm)
            np.save(newname, norm.astype(np.float16))


# -mean, / std
class ZeroMeanUnitVar(BaseNormalize):
    def normalize(self, raw):
        return ctseg.ctutil.norm.standardize(raw)


# min -> 0, / mean, log
class NormLog(BaseNormalize):
    def normalize(self, raw):
        return norm_log(raw)


# min -> 0, / percentile of 0 label in train targets, log
class MagicNormLog(BaseNormalize):
    def init_metadata(self, targets_dir):
        target_paths = [
            os.path.join(targets_dir, f)
            for f in os.listdir(targets_dir)
            if f.endswith(".npy")
            and not f.endswith("counts.npy")
            and not f.endswith("probs.npy")
        ]
        if not target_paths:
            raise ValueError(f"No target files found in dir: {targets_dir}")
        qs = [find_q_for_tgt(path) for path in target_paths]
        self.metadata = {"q": np.array(qs).mean()}

    def normalize(self, raw):
        return magic_norm_log(raw, self.metadata["q"])
