from ctseg.config import ex
from ctseg.data_loader import DataLoader
from ctseg.sampler import Sampler, OverlapSampler


@ex.automain
def main(model_config, data_config, num_classes, train_config, test_config):
    """Validate the train and validation datasets"""
    train_data_dir = train_config["inputs"]["data_dir"]
    train_targets_dir = train_config["inputs"]["targets_dir"]
    test_data_dir = test_config["inputs"]["data_dir"]
    test_targets_dir = test_config["inputs"]["targets_dir"]

    input_shape = model_config["architecture_config"]["input_shape"]
    sampler_config = data_config["sampler_config"]

    train_set = DataLoader(
        data_dir=train_data_dir,
        data_config=data_config,
        num_classes=num_classes,
        sampler=Sampler.from_config(sampler_config, chunk_shape=input_shape),
        targets_dir=train_targets_dir,
        mode="train",
    )
    test_set = DataLoader(
        data_dir=test_data_dir,
        data_config=data_config,
        num_classes=num_classes,
        sampler=OverlapSampler.from_config(sampler_config, chunk_shape=input_shape),
        targets_dir=test_targets_dir,
        mode="test",
    )

    train_set.validate_dataset()
    test_set.validate_dataset()
