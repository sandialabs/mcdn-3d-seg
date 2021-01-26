import logging

import numpy as np

from ctseg.config import ex

logger = logging.getLogger(__name__)


VOXEL_GRID_WIDTH = 20
MAX_GRID_VALUE = 1000


def forceMinMax(list, min, max):
    list[np.where(list < min)] = min
    list[np.where(list > max)] = max
    return list


class BattleShip():
    def __init__(self, trainSet=None, key=None):
        if trainSet is None:
            raise (RuntimeError("trainSet must be set"))
        if key is None:
            raise (RuntimeError("key must be set"))
        self.trainSet = trainSet
        self.key = key

        self.max_x, self.max_y, self.max_z, self.num_classes = self.target().shape

        self.xVoxelGridLength = int(self.max_x / VOXEL_GRID_WIDTH)
        self.yVoxelGridLength = int(self.max_y / VOXEL_GRID_WIDTH)
        self.zVoxelGridLength = int(self.max_z / VOXEL_GRID_WIDTH)

        self.getProbabilities()

    def target(self):
        return self.trainSet._labels_cache[self.key]

    def getXYZfromVoxelIndex(self, index):
        z = (index / (self.xVoxelGridLength * self.yVoxelGridLength)).astype(int)
        index -= z * (self.xVoxelGridLength * self.yVoxelGridLength)
        y = (index / self.xVoxelGridLength).astype(int)
        index -= y * self.xVoxelGridLength
        x = index
        return (x, y, z)

    def getVoxelIndexFromXYZ(self, x, y, z):
        return x + y * self.xVoxelGridLength + z * (
                    self.xVoxelGridLength * self.yVoxelGridLength)

    @ex.capture
    def selectVoxelPoints(self, model_config, selections=1):
        x_max, y_max, z_max = model_config["architecture_config"]["input_shape"]

        probs = self.voxelGrid / float(np.sum(self.voxelGrid))
        index = np.random.choice(self.voxelList, selections, p=probs)

        x, y, z = self.getXYZfromVoxelIndex(index)
        x = x * VOXEL_GRID_WIDTH + np.random.randint(0, high=(63), size=x.size) - 32
        y = y * VOXEL_GRID_WIDTH + np.random.randint(0, high=(63), size=y.size) - 32
        z = z * VOXEL_GRID_WIDTH + np.random.randint(0, high=(63), size=z.size) - 32
        forceMinMax(x, 0, self.max_x - 1 - x_max)
        forceMinMax(y, 0, self.max_y - 1 - y_max)
        forceMinMax(z, 0, self.max_z - 1 - z_max)

        return np.array([x, y, z]).T

    def forceVoxelRange(self, index_list):
        for index in index_list:
            if (index < 0 or index > self.voxelGrid.size - 1):
                continue
            if (self.voxelGrid[index] < 1):
                self.voxelGrid[index] = 1
            if (self.voxelGrid[index] > MAX_GRID_VALUE):
                self.voxelGrid[index] = MAX_GRID_VALUE

    def getVoxelIndexFromDataXYZ(self, x, y, z):
        return self.getVoxelIndexFromXYZ(int(x / VOXEL_GRID_WIDTH),
                                         int(y / VOXEL_GRID_WIDTH),
                                         int(z / VOXEL_GRID_WIDTH))

    def changeVoxelAt(self, x, y, z, value):
        index = self.getVoxelIndexFromDataXYZ(x, y, z)
        self.voxelGrid[index] += value
        self.forceVoxelRange([index])

    def getProbabilities(self):
        path = self.trainSet.get_target_path(self.key, "_probs.npy")
        try:
            self.voxelGrid = np.load(path, fix_imports=True, encoding='bytes')
            self.set_class_weights()
            self.voxelList = np.arange(self.voxelGrid.size)
        except IOError:
            self.initializeProbabilities()

    def initializeProbabilities(self):
        self.resetVoxelGrid()
        for i in range(self.voxelGrid.size):
            x, y, z = self.getXYZfromVoxelIndex(np.array([i]))
            x = x.sum() * VOXEL_GRID_WIDTH
            y = y.sum() * VOXEL_GRID_WIDTH
            z = z.sum() * VOXEL_GRID_WIDTH

            for j in range(len(self.class_weights)):
                j_label_count = self.target()[x:(x + VOXEL_GRID_WIDTH),
                                y:(y + VOXEL_GRID_WIDTH), z:(z + VOXEL_GRID_WIDTH), j].sum()
                self.voxelGrid[i] += j_label_count * self.class_weights[j]
        self.voxelGrid /= self.voxelGrid.sum()
        path = self.trainSet.get_target_path(self.key, "_probs.npy")
        np.save(path, self.voxelGrid)

    def set_class_weights(self):
        weights = self.trainSet.get_label_freq(self.key)
        class_weights = weights.sum() / weights
        self.class_weights = class_weights / class_weights.sum()

    def resetVoxelGrid(self):
        self.voxelGrid = np.ones(
            (self.xVoxelGridLength * self.yVoxelGridLength * self.zVoxelGridLength))
        self.voxelList = np.arange(self.voxelGrid.size)
        self.set_class_weights()
