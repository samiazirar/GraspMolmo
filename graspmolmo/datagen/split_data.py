import os
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

@hydra.main(version_base=None, config_path="../../config", config_name="split.yaml")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")
    print(OmegaConf.to_yaml(cfg))

    rng = np.random.RandomState(cfg.seed)

    with open(cfg.all_categories_path, "r") as f:
        all_categories = set(f.read().splitlines())

    with open(cfg.blacklist, "r") as f:
        blacklist = set(f.read().splitlines())

    assets: dict[str, list[str]] = {}  # category -> list of object ids
    for fn in os.listdir(os.path.join(cfg.data_dir, "grasps")):
        if not fn.endswith(".h5"):
            continue
        category, obj_id = fn[:-len(".h5")].split("_", 1)
        if category not in all_categories or f"{category}_{obj_id}" in blacklist:
            continue

        if category not in assets:
            assets[category] = []
        assets[category].append(obj_id)

    test_data: dict[str, list[str]] = {}
    train_data: dict[str, list[str]] = {}

    for category in cfg.test_categories:
        obj_ids = assets.pop(category)
        for obj_id in obj_ids:
            if category not in test_data:
                test_data[category] = []
            test_data[category].append(f"{category}_{obj_id}")

    for category in sorted(assets.keys()):
        obj_ids = assets[category]
        rng.shuffle(obj_ids)
        n_test = int(np.ceil(len(obj_ids) * cfg.test_frac))

        if category not in test_data:
            test_data[category] = []
        test_data[category].extend(obj_ids[:n_test])

        if category not in train_data:
            train_data[category] = []
        train_data[category].extend(obj_ids[n_test:])

    print(f"Test categories: {len(test_data)}, instances: {sum(len(v) for v in test_data.values())}")
    print(f"Train categories: {len(train_data)}, instances: {sum(len(v) for v in train_data.values())}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    with open(os.path.join(cfg.out_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

if __name__ == "__main__":
    main()
