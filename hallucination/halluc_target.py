import os
import sys
import copy
import json
sys.path.append(os.getcwd())
from llms.chatgpt import Chatgpt
from llms.prompts import gen_absent_object_prompt, gen_absent_unsuitable_object_prompt
from util.plot import visualize_image_caption_abs_obj
from .detection import GroundedDetection


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
        result_dict = {}
        # for i in range(len(self.images)):
        for i in range(num_vis):
            image = self.images[i]
            caption_info = self.captions[image].copy()
            caption = caption_info["caption"]
            abs_objs = self.gen_absent_object(caption)
            abs_unsuitable_objs = self.gen_absent_unsuitable_object(caption)
            result_dict[image] = {"caption": caption, 
                                  "absent_suitable_objs": abs_objs,
                                  "absent_unsuitable_objs": abs_unsuitable_objs}
            result_dict[image] = {"caption": caption, 
                                  "absent_suitable_objs": abs_objs}

            if i < num_vis:
                visualize_image_caption_abs_obj(image, caption, abs_objs, out_dir=out_dir)
                visualize_image_caption_abs_obj(image, caption, abs_unsuitable_objs, out_dir=out_dir+"_unsuitable")
        out_file = os.path.join(out_dir, "abs_obj_example.json")
        with open(out_file, "w") as fp:
            json.dump(result_dict, fp)
        return result_dict



if __name__ == "__main__":
    chatgpt = Chatgpt()
    htarget = HallucTarget(chatgpt, "data/refcoco/anns_spatial/captions_with_grounding.json")
    htarget.gen_absent_object_all(out_dir="output/val6")

    # detector = GroundedDetection()
    # boxes, pred_phrases = detector.inference(image, obj, box_threshold=0.25, 
    #                                          text_threshold=0.2, iou_threshold=0.5)

