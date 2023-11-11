import os
import sys
import copy
import json
import torch
import numpy as np
sys.path.append(os.getcwd())
from llms.chatgpt import Chatgpt
from llms.prompts import (gen_absent_object_prompt, gen_absent_unsuitable_object_prompt, \
                          gen_halluc_phrase_prompt)
from util.plot import visualize_image_caption_abs_obj, read_image_pil
from hallucination.detection import GroundedDetection
from datasets.SpatialReasoningDataset import SpatialReasoningDataset


class HallucTarget():
    def __init__(self, llm, caption_file):
        self.llm = llm
        self.captions = self._load_qwen_caption(caption_file)
        self.images = list(self.captions.keys())


    def _load_qwen_caption(self, path):
        captions = json.load(open(path))
        return captions


    def gen_absent_object(self, caption):
        prompt = gen_absent_object_prompt(caption)
        result = self.llm.get_completion(prompt)
        result = self.llm.parse_str_to_list(result)
        return result
    
    def gen_absent_unsuitable_object(self, caption):
        prompt = gen_absent_unsuitable_object_prompt(caption)
        result = self.llm.get_completion(prompt)
        result = self.llm.parse_str_to_list(result)
        return result
    

    def gen_absent_object_all(self, out_dir, num_vis=10):
        os.makedirs(out_dir, exist_ok=True)
        result_dict = {}
        for i in range(len(self.images)):
            print(i)
            image = self.images[i]
            out_file = os.path.join(out_dir, f"{i}.json")
            if os.path.exists(out_file):
                curr_dict = json.load(open(out_file))
            else:
                caption_info = self.captions[image].copy()
                caption = caption_info["caption"]
                abs_objs = self.gen_absent_object(caption)

                curr_dict = {"caption": caption,
                            "absent_suitable_objs": abs_objs}
                with open(out_file, "w") as fp:
                    json.dump(curr_dict, fp)

            result_dict[image] = curr_dict

            if i < num_vis:
                visualize_image_caption_abs_obj(image, caption, abs_objs, out_dir=out_dir)
        out_file = os.path.join(out_dir, "absent_objects.json")
        with open(out_file, "w") as fp:
            json.dump(result_dict, fp)
        return result_dict


def get_obj_grounding_score(image, objects, detector):
    detection_score = []
    for obj in objects:
        boxes, scores = detector.inference(image, obj)
        if len(scores) == 0:
            detection_score.append(0)
        else:
            detection_score.append(max(scores))
        print(obj, scores)
    return detection_score


def get_obj_grounding_score_all(file="output/absent_objects.json", 
                                im_dir="data/refcoco/images/eval_image"):
    detector = GroundedDetection()
    infos = json.load(open(file))
    for iname, v in infos.items():
        if v["grounding_score"] == []:
            img = read_image_pil(os.path.join(im_dir, iname))
            objects = v["absent_suitable_objs"]
            scores = get_obj_grounding_score(img, objects, detector)
            v["grounding_score"] = scores
        else:
            continue
    with open(file, "w") as fp:
        json.dump(infos, fp)
    return infos


def gen_halluc_phrase(llm, halluc_file="output/absent_objects.json", 
                      version="refcoco_unc", split="testA",
                      out_dir="data/refcoco/anns_spatial/"):
    halluc_anns = json.load(open(halluc_file))
    SR = SpatialReasoningDataset(version, split)
    anns = SR.anns
    result_dict = {}
    tmp_dir = f"output/response_halluc_phrase_{version}_{split}"
    os.makedirs(tmp_dir, exist_ok=True)
    for iname, v in anns.items():
        print(iname)
        result_per_image = os.path.join(tmp_dir, f"{iname[:-4]}.pth")
        if os.path.exists(result_per_image):
            result_dict[iname] = torch.load(result_per_image)
        else:
            halluc_phrases = []
            raw_phrases = []
            corres_boxes = []

            halluc_objs = halluc_anns[iname]["absent_suitable_objs"]
            obj_grounding_score = halluc_anns[iname]["grounding_score"]
            assert len(halluc_objs) == len(obj_grounding_score)
            idxs = np.argsort(obj_grounding_score)
            sorted_objs = np.array(halluc_objs)[idxs]
            bbox_phrases = v["box"] # dict
            for box, phrases in bbox_phrases.items():
                for p in phrases: # set
                    cand_obj = sorted_objs[len(halluc_phrases) % len(sorted_objs)]
                    raw_phrases.append(p)
                    prompt = gen_halluc_phrase_prompt(p, cand_obj)
                    response = llm.get_completion(prompt)
                    response = llm.parse_phrase(response)
                    halluc_phrases.append(response)
                    corres_boxes.append(box)
            result_dict[iname] = {"halluc_phrases": halluc_phrases,
                                "raw_phrase": raw_phrases,
                                "raw_boxes": corres_boxes,
                                "image_size": v["image_size"]}
            torch.save(result_dict[iname], result_per_image)
    out_file = os.path.join(out_dir, f"halluc_phrases_{version}_{split}.json")
    with open(out_file, "w") as fp:
        json.dump(result_dict, fp)
    return result_dict



if __name__ == "__main__":
    chatgpt = Chatgpt()
    # htarget = HallucTarget(chatgpt, "data/refcoco/anns_spatial/captions_with_grounding.json")
    # htarget.gen_absent_object_all(out_dir="output/response", num_vis=0)
    # get_obj_grounding_score_all()
    # result_dict = gen_halluc_phrase(chatgpt, version="refcocog_umd", split="val")
