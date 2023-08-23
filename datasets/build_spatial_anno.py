import os
import torch
from grounding_datasets import ReferSegDataset
from refer_segmentation import build_refcoco_segmentation


def load_prepositions(file="data/prepositions.txt"):
    assert os.path.exists(file), f"{file} not found!"
    with open(file) as f:
        lines = f.readlines()
    preps = [line.lower().strip() for line in lines]
    preps = sorted(set(preps))
    return preps


def has_preposition(input, preps):
    flag = False
    input = input.lower()
    for prep in preps:
        if prep+" " in input:
            flag = True
            break
    return flag


def filter_anns(root="data/refcoco/anns", 
               im_dir="./data/refcoco/images/train2014",
               seg_dir="./data/refcoco/masks",
               out_root="data/refcoco/anns_spatial",
               preps=None):
    ds = ReferSegDataset(data_root=root, im_dir=im_dir, seg_dir=seg_dir)
    DATASETS = ds.SUPPORTED_DATASETS
    for version, meta_data in DATASETS.items():
        dataset_dir = version.split('_')[0]
        # dataset_path = os.path.join(root, dataset_dir)
        out_dir = os.path.join(out_root, dataset_dir)
        os.makedirs(out_dir, exist_ok=True)
        for split in meta_data["splits"]:
            if split == "trainval":
                continue
            # process .pth file
            imgset_file = '{0}_{1}.pth'.format(version, split)
            pth_path = os.path.join(out_dir, imgset_file)
            dataset = build_refcoco_segmentation(split=split, version=version, data_root=root)
            
            if not os.path.exists(pth_path):
                dataset_filtered = []
                for i, item in enumerate(dataset.images):
                    phrase = item[3]
                    has_prep = has_preposition(phrase, preps)
                    if has_prep:
                        dataset_filtered.append(item)
                torch.save(dataset_filtered, pth_path)
                print(f"Saved file {pth_path}")



if __name__ == "__main__":
    preps = load_prepositions()
    filter_anns(preps=preps)
