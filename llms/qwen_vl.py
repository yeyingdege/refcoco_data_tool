import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


class Qwen_VL():
    def __init__(self, model_name="Qwen/Qwen-VL", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device, trust_remote_code=True).eval()
        # use bf16
        # model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cpu", trust_remote_code=True).eval()
        # Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
        # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

        self.default_caption_prompt = "Generate the caption in English:"
        self.default_grounding_prompt = "Generate the caption in English with grounding:"


    def get_response(self, image, text=None, opt="caption"):
        """
        image: Either an absolute local path or an url
        text: text prompt
        """
        if text is None or text=="":
            text = self.default_caption_prompt
        query = self.tokenizer.from_list_format([
            {'image': image}, 
            {'text': text},
        ])
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs)
        raw_response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        
        if opt == "caption":
            result_dict = self._process_response(raw_response, txt_pt=text)
        else:
            result_dict = self._process_obj_bbox(raw_response, txt_pt=text)

            image = self.tokenizer.draw_bbox_on_latest_picture(raw_response)
            if image:
                image.save('2.jpg')
            else:
                print("no box")
        return result_dict


    def _process_response(self, response, txt_pt, endoftext="<|endoftext|>"):
        start = response.find(txt_pt)
        out_str = response[start+len(txt_pt):]
        out_str = out_str.replace(endoftext, "")
        out_str, phrases = self._extract_ref(out_str)
        out_str, bbox = self._extract_bbox(out_str)
        out_str = out_str.strip()
        result_dict = {"raw_response": response,
                       "processed_response": out_str,
                       "bboxes": bbox,
                       "phrases": phrases}
        return result_dict
    
    def _process_obj_bbox(self, response, txt_pt, endoftext="<|endoftext|>"):
        start = response.find(txt_pt)
        out_str = response[start+len(txt_pt):]
        out_str = out_str.replace(endoftext, "")
        pattern_obj = r'<ref>(.*?)</ref>'
        objects = re.findall(pattern_obj, out_str)
        
        final_objs = []
        bboxes = []
        for obj in objects:
            prefix = f'{obj}</ref><box>'
            pattern_bbox = rf'{prefix}(.*?)</box>'
            bbox = re.findall(pattern_bbox, out_str)
            final_objs.append(obj.strip())
            bboxes.append(bbox[0])
        result = {"bboxes": bboxes, "phrases": final_objs}
        return result
    
    def _extract_ref(self, s):
        phrases = []
        while "<ref>" in s:
            begin = s.find("<ref>")
            end = s.find("</ref>")
            p = s[begin:end] + "</ref>"
            phrases.append(p[5:-6])
            s = s.replace(p, p[5:-6])
        return s, phrases
    
    def _extract_bbox(self, s):
        bbox = []
        while "<box>" in s:
            begin = s.find("<box>")
            end = s.find("</box>")
            box = s[begin:end] + "</box>"
            bbox.append(box[5:-6])
            s = s.replace(box, "")
        return s, bbox


if __name__=="__main__":
    test_image = 'data/refcoco/images/train2014/COCO_train2014_000000297370.jpg'
    test_image = os.path.join(os.getcwd(), test_image)

    qwen = Qwen_VL()
    response = qwen.get_response(test_image, text=qwen.default_grounding_prompt, opt="det")
    print(response["phrases"], response["bboxes"])

