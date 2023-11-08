import os
import sys
import copy
import json
sys.path.append(os.getcwd())
from llms.chatgpt import Chatgpt
from llms.prompts import gen_absent_object_prompt, gen_absent_unsuitable_object_prompt
from util.plot import visualize_image_caption_abs_obj, read_image_pil
from hallucination.detection import GroundedDetection


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
        img = read_image_pil(os.path.join(im_dir, iname))
        objects = v["absent_suitable_objs"]
        scores = get_obj_grounding_score(img, objects, detector)
        v["grounding_score"] = scores
    with open(file, "w") as fp:
        json.dump(infos, fp)
    return infos



if __name__ == "__main__":
    chatgpt = Chatgpt()
    htarget = HallucTarget(chatgpt, "data/refcoco/anns_spatial/captions_with_grounding.json")
    # htarget.gen_absent_object_all(out_dir="output/response", num_vis=0)
    # get_obj_grounding_score_all()
