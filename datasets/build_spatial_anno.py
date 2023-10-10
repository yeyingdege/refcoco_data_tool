import os
import json
import torch
import csv
from refer_segmentation import build_refcoco_segmentation
from SpatialReasoningDataset import SpatialReasoningDataset
from util.plot import visualize_image_info


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
    'refcocog_umd': {
        'splits': ('train', 'val', 'test'),
        'params': {'dataset': 'refcocog', 'split_by': 'umd'}
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


def traverse_datasets(root="data/refcoco/anns_spatial", num_vis_image=5):
    out_str = "version,split,num_images,num_phrase\n"
    for version, meta_data in SUPPORTED_SR_DATASETS.items():
        for split in meta_data["splits"]:
            if split == "trainval":
                continue
            dataset = SpatialReasoningDataset(split=split, version=version, ann_root=root)
            # generate result string
            num_images = len(dataset.anns)
            num_phrase = len(dataset)
            line = version + "," + split + "," + str(num_images) + "," + str(num_phrase) + "\n"
            out_str = out_str + line
            # Visualize images
            for i in range(num_vis_image):
                iname = dataset.idx_to_iname[i]
                ann = dataset.anns[iname]
                box = ann["box"]
                visualize_image_info(iname, collect_box_text=box, 
                                     draw_phrase=True, out_dir=f'output/{version}/{split}')
    out_csv_file = "output/info.csv"
    with open(out_csv_file, "w") as fp:
        fp.write(out_str)


def write_dict_to_csv(json_file):
    # save caption to csv for better viewing
    d = json.load(open(json_file))
    csv_file = json_file.replace("json", "csv")
    col_name = ["image_name", "caption", "version", "split"]
    with open(csv_file, 'w') as csvFile:
        wr = csv.DictWriter(csvFile, fieldnames=col_name)
        wr.writeheader()
        for iname, v in d.items():
            row = {col_name[0]:iname, col_name[1]:v[col_name[1]], 
                   col_name[2]:v[col_name[2]], col_name[3]:v[col_name[3]]}
            wr.writerow(row)


def get_image_caption_qwen(qwen, im_dir="./data/refcoco/images/train2014",
                           out_root="data/refcoco/anns_spatial"):
    captions = {}
    curr_dir = os.getcwd()
    for version, meta_data in SUPPORTED_SR_DATASETS.items():
        dataset_dir = version.split('_')[0]
        out_dir = os.path.join(out_root, dataset_dir)
        os.makedirs(out_root, exist_ok=True)
        for split in meta_data["splits"]:
            if split in ["train", "trainval"]:
                continue
            input_file = os.path.join(out_dir, f'{version}_{split}_dict.pth')
            input_data = torch.load(input_file)
            for iname, v in input_data.items():
                if iname in captions:
                    continue
                full_path = os.path.join(curr_dir, im_dir, iname)
                response = qwen.get_response(full_path, text=qwen.default_grounding_prompt)
                captions[iname] = {"caption": response["processed_response"],
                                   "raw_response": response["raw_response"], 
                                   "bboxes": response["bboxes"],
                                   "version": version, "split": split}
                print(response["processed_response"])
    # save dict to json file
    save_name = os.path.join(out_root, "captions_with_grounding.json") # 4163
    with open(save_name, "w") as fp:
        json.dump(captions, fp)
    write_dict_to_csv(save_name)


if __name__ == "__main__":
    # preps = load_prepositions()
    # filter_anns(preps=preps)
    # traverse_datasets(num_vis_image=5)

    from llms.qwen_vl import Qwen_VL
    qwen = Qwen_VL()
    get_image_caption_qwen(qwen)
