import os
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


    def get_response(self, image, text=None):
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
        """
        Picture 1:<img>/home/yanghong/source/refcoco_data_tool/data/refcoco/images/train2014/COCO_train2014_000000000154.jpg</img>
        Generate the caption in English: zebras in a field , grazing .<|endoftext|>
        """
        response = self._process_response(raw_response, txt_pt=self.default_caption_prompt)
        # image = self.tokenizer.draw_bbox_on_latest_picture(raw_response)
        # if image:
        #     image.save('2.jpg')
        # else:
        #     print("no box")
        return response


    def _process_response(self, response, txt_pt, endoftext="<|endoftext|>"):
        start = response.find(txt_pt)
        output = response[start+len(txt_pt):]
        output = output.replace(endoftext, "")
        output = output.strip()
        return output


if __name__=="__main__":
    test_image = 'data/refcoco/images/train2014/COCO_train2014_000000000154.jpg'
    test_image = os.path.join(os.getcwd(), test_image)

    qwen = Qwen_VL()
    response = qwen.get_response(test_image)
    print(response)
