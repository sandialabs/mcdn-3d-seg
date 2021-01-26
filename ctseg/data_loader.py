import logging
import os

import numpy as np
from tqdm import tqdm

from ctseg.battleship import BattleShip
import ctseg.norm
from ctseg.sampler import BattleShipSampler


logger = logging.getLogger(__name__)


def validate_array(array):
    assert not np.isnan(array).any()


class DataLoader:
    def __init__(
        self,
        data_dir,
        data_config,
        num_classes,
        sampler,
        targets_dir=None,
        mode="train",
        normalization="",
        normalizer_metadata_dir=None,
        key_limit=100,
        normalized_image_dir=None,
    ):
        if targets_dir is None and mode != "infer":
            raise (
                RuntimeError("Input 'targets_dir' must be set if mode is not 'infer'")
            )

        self.data_dir = data_dir
        self.targets_dir = targets_dir
        self._normalization = normalization
        self._normalizer_metadata_dir = normalizer_metadata_dir
        self._normalized_image_dir = normalized_image_dir
        self._sample_cache = {}
        self._labels_cache = {}
        self._bship_cache = {}

        if mode == "infer":
            # in order to reconstruct an prediction fully from chunks, can only
            # handle one image at a time
            key_limit = 1

        self.keys = self._initialize_filekeys(data_dir, key_limit)

        self._sampler = sampler

        self.mode = mode
        self._normalize_images()

        self.flip_x = data_config["flip_x"]
        self.flip_y = data_config["flip_y"]
        self.flip_z = data_config["flip_z"]
        self.flip_validation_axis = data_config["flip_validation_axis"]

        self.num_classes = num_classes

    @property
    def normalizer(self):
        if not hasattr(self, "_normalizer"):
            if self._normalization:
                if self._normalizer_metadata_dir is None:
                    raise ValueError(
                        "'normalizer_metadata_dir' must be set if using normalization."
                    )

                if self._normalized_image_dir is None:
                    raise ValueError(
                        "'normalized_image_dir' must be set if using normalization."
                    )

                normalizer_cls = getattr(ctseg.norm, self._normalization)
                self._normalizer = normalizer_cls(
                    self._normalizer_metadata_dir, self.targets_dir
                )
            else:
                self._normalizer = None

        return self._normalizer

    @property
    def sampler(self):
        return self._sampler

    def __len__(self):
        return len(self._chunks)

    def get_target_path(self, key, extension):
        return os.path.join(self.targets_dir, key + extension)

    def get_raw_image_path(self, key, extension):
        return os.path.join(self.data_dir, key + extension)

    def get_image_path(self, key, extension):
        if not self._normalization:
            return self.get_raw_image_path(key, extension)
        else:
            return os.path.join(
                self._normalized_image_dir, self._normalization, key + extension
            )

    @staticmethod
    def load_npy_file_from_path(filename):
        logger.info(f"loading {filename}")
        # mmap_mode allows for the file to be kept on disk for much faster loading
        return np.load(filename, fix_imports=True, encoding="bytes", mmap_mode="r")

    def get_all_label_freq(self):
        counts = np.zeros(self.num_classes)
        for key in self.keys:
            counts += self.get_label_freq(key)
        self.target_label_freq = counts

    def get_label_freq(self, key):
        path = self.get_target_path(key, "_counts.npy")
        try:
            label_count = np.load(path, fix_imports=True, encoding="bytes")
        except FileNotFoundError:
            label_count = self._calc_label_freq(key)
            np.save(path, label_count)
        return label_count

    def get_class_weights(self):
        total_instances = np.sum(self.target_label_freq)
        class_ratio = 1 / (self.target_label_freq / total_instances)
        return class_ratio

    @staticmethod
    def validate_image_label_shape(image, label):
        ix, iy, iz = image.shape
        lx, ly, lz, nc = label.shape
        assert (ix, iy, iz) == (lx, ly, lz)

    def validate_dataset(self):
        """Check all images for inf/nan and output mean/var"""
        for key in self.keys:
            self._log(f"Validating {key}")
            image_path = self.get_image_path(key, ".npy")
            image = self.load_npy_file_from_path(image_path)
            target_path = self.get_target_path(key, ".npy")
            label = self.load_npy_file_from_path(target_path)
            validate_array(image)
            self._log(f"Image: {image.mean()} +/-{image.std()}")
            self.validate_image_label_shape(image, label)
            validate_array(label)
            self._log(f"Label: {label.mean()} +/-{label.std()}")

    def gen_chunk_list(self):
        self._log("Generating new chunk list")

        # load new samples
        keys = self.sampler.select_keys(self.keys)
        self._update_data_caches(keys)

        # use sampler to select chunks
        self._chunks = self.sampler.select_chunks(self._sample_cache, self._bship_cache)

        self._log(f"Number of chunks: {len(self._chunks)}")

    def get_key(self, index):
        return self._chunks[index][0]

    def get_coords(self, index):
        return self._chunks[index][1]

    def get_image(self, index):
        return self._sample_cache[self.get_key(index)]

    def get_chunk(self, index):
        key = self.get_key(index)
        img = self.get_image(index)

        x_coords, y_coords, z_coords = self.get_coords(index)
        slc = (slice(*x_coords), slice(*y_coords), slice(*z_coords))
        chunk = img[slc]
        
        chunk_shape = self.sampler.chunk_shape

        chunk = chunk.reshape((1, *chunk_shape, 1)).astype(np.float32)

        if self.mode == "infer":
            return chunk

        label = self._labels_cache[key]
        label_chunk = label[(*slc, slice(None))]
        label_chunk = label_chunk.reshape((1, *chunk_shape, self.num_classes))

        if self.flip_x and np.random.randint(2):
            chunk = np.flip(chunk, 1)
            label_chunk = np.flip(label_chunk, 1)
        if self.flip_y and np.random.randint(2):
            chunk = np.flip(chunk, 2)
            label_chunk = np.flip(label_chunk, 2)
        if self.flip_z and np.random.randint(2):
            chunk = np.flip(chunk, 3)
            label_chunk = np.flip(label_chunk, 3)

        if self.mode == "test" and self.flip_validation_axis:
            chunk = np.flip(chunk, self.flip_validation_axis)
            label_chunk = np.flip(label_chunk, self.flip_validation_axis)

        return chunk, label_chunk

    def create_generator(self, batch_size):
        """Creates a generator that infinitely yields batches of (data, target) tuples
        (to be used when fitting a model).

        Args:
            batch_size: the length of the first dimension, i.e. number of samples, of
                the `data` and `target` yielded

        Returns:
            a generator object
        """
        chunk_shape = self.sampler.chunk_shape

        # get the number of classes
        if not self._labels_cache:
            self.gen_chunk_list()
        x, y = self.get_chunk(0)
        num_classes = y.shape[-1]

        # sample chunks
        self.gen_chunk_list()
        chunk_idx = 0

        # define batch data/target
        X = np.zeros((batch_size, *chunk_shape, 1))
        Y = np.zeros((batch_size, *chunk_shape, num_classes))
        batch_idx = 0

        skipped_chunks = 0
        total_chunks = 0

        while True:
            if chunk_idx == len(self):
                # resample chunks
                self.gen_chunk_list()
                chunk_idx = 0

            x, y = self.get_chunk(chunk_idx)
            chunk_idx += 1

            total_chunks += 1
            skip_chunk = False
            for c in range(num_classes):
                skip_chunk = np.all(y[..., c] == 1)
                if skip_chunk:
                    break
            if skip_chunk:
                skipped_chunks += 1
                if skipped_chunks % 20 == 1:
                    self._log(
                        f"Skipping chunk because all same label. Total"
                        f" {skipped_chunks}/{total_chunks} chunks skipped"
                    )
                continue

            X[batch_idx, ...] = x
            Y[batch_idx, ...] = y
            batch_idx += 1

            if batch_idx == batch_size:
                yield X, Y
                batch_idx = 0

    def _log(self, msg, *args, **kwargs):
        logger.info(f"{self.mode} - {msg}", *args, **kwargs)

    def _normalize_images(self):
        if not self.normalizer:
            return

        self._log(f"Normalizing via {self.normalizer.cls_name}")
        for key in tqdm(self.keys):
            # normalize key
            filename = self.get_raw_image_path(key, ".npy")
            self.normalizer.normalize_file(
                filename, cache_dir=self._normalized_image_dir
            )

    def _calc_label_freq(self, key):
        self._log(f"calculating freq for {key}")
        counts = np.zeros(self.num_classes)
        path = self.get_target_path(key, ".npy")
        tgt = np.load(path, fix_imports=True, encoding="bytes")
        for i in range(self.num_classes):
            counts[i] = tgt[:, :, :, i].sum()
        return counts

    def _update_data_caches(self, keys):
        keys_to_remove = [k for k in self._sample_cache if k not in keys]

        for key in keys_to_remove:
            del self._sample_cache[key]
            self._labels_cache.pop(key, None)
            self._bship_cache.pop(key, None)

        for key in keys:
            if key not in self._sample_cache:
                # load sample
                image_path = self.get_image_path(key, ".npy")
                self._sample_cache[key] = self.load_npy_file_from_path(image_path)

                # load label
                if self.mode != "infer":
                    target_path = self.get_target_path(key, ".npy")
                    self._labels_cache[key] = self.load_npy_file_from_path(target_path)
                    self.validate_image_label_shape(
                        self._sample_cache[key], self._labels_cache[key]
                    )

            # initialize battleship
            if (
                isinstance(self.sampler, BattleShipSampler)
                and key not in self._bship_cache
            ):
                self._bship_cache[key] = BattleShip(trainSet=self, key=key)

        # normalize any new images
        self._normalize_images()

    @staticmethod
    def _initialize_filekeys(data_dir, key_limit):
        keys = []
        filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        for f in filenames:
            key = f[:-4]
            if len(keys) < key_limit:
                keys.append(key)
        if len(keys) == 0:
            raise (
                FileNotFoundError("Found 0 images: " + os.path.join(data_dir) + "\n")
            )
        return keys
