import os
import json
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


SUPPORTED_SR_DATASETS = {
    'refcoco_unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'refcocog_google': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    }
}


def convert_to_dict(dataset):
    """Each image as a key"""
    result = {}
    for data in dataset:
        image_name, mask, box, phrase = data
        if image_name not in result:
            result[image_name] = {"box": {}, "mask": []}
        box = tuple(box)
        if box not in result[image_name]["box"]:
            result[image_name]["box"][box] = set()
            result[image_name]["mask"].append(mask)
        result[image_name]["box"][box].add(phrase)
    return result


def filter_by_box_number(data_dict, min_box_num=3):
    result = {}
    for k, v in data_dict.items():
        if len(v["box"]) < min_box_num:
            continue
        result[k] = v
    return result


def filter_anns(root="data/refcoco/anns", 
               im_dir="./data/refcoco/images/train2014",
               seg_dir="./data/refcoco/masks",
               out_root="data/refcoco/anns_spatial",
               preps=None):
    DATASETS = SUPPORTED_SR_DATASETS
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
            
            dataset_filtered = []
            for i, item in enumerate(dataset.images):
                phrase = item[3]
                has_prep = has_preposition(phrase, preps)
                if has_prep:
                    dataset_filtered.append(item)
            # torch.save(dataset_filtered, pth_path)
            # print(f"Saved file {pth_path}")
            dataset_dict = convert_to_dict(dataset_filtered)
            data_filtered_by_box_num = filter_by_box_number(dataset_dict)
            
            num_phrase = 0
            for k, v in data_filtered_by_box_num.items():
                for kk, vv in v["box"].items():
                    num_phrase += len(vv)

            save_file = os.path.join(out_dir, f'{version}_{split}_dict.pth')
            torch.save(data_filtered_by_box_num, save_file)
            print(f"{version}/{split},{len(data_filtered_by_box_num)} images,{num_phrase} phrase after filtering")


def traverse_datasets(root="data/refcoco/anns_spatial", 
                    im_dir="./data/refcoco/images/train2014",
                    seg_dir="./data/refcoco/masks"):
    DATASETS = SUPPORTED_SR_DATASETS
    out_str = "version,split,num_images,num_phrase\n"
    for version, meta_data in DATASETS.items():
        for split in meta_data["splits"]:
            if split == "trainval":
                continue
            dataset = build_refcoco_segmentation(split=split, version=version, data_root=root)
            # generate result string
            num_images = len(dataset.box_phrases_dict)
            num_phrase = 0
            for item in dataset.img_infos:
                num_phrase += len(item["phrase"])
            line = version + "," + split + "," + str(num_images) + "," + str(num_phrase) + "\n"
            out_str = out_str + line
            # Visualize images
            for i in range(0, 30, 10):
                dataset.visualize_image_info(i, draw_phrase=True)
    out_csv_file = "output/info.csv"
    with open(out_csv_file, "w") as fp:
        fp.write(out_str)


if __name__ == "__main__":
    preps = load_prepositions()
    filter_anns(preps=preps)
    # traverse_datasets()
