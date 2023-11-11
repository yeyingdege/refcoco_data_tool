import os, sys
import re
import copy
import openai
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.getcwd())
from util.logger import setup_logger


class Chatgpt:
    def __init__(self, log_file="output/"):
        _ = load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.logger = setup_logger("chatgpt", log_file, 0)


    def get_completion(self, prompt, model="gpt-3.5-turbo", 
                       max_tokens=1000, temperature=0):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        result = response.choices[0].message["content"]
        # logging
        self.logger.info(f"prompt:{prompt}")
        self.logger.info(f"response:{result}")
        return result


    def parse_str_to_list(self, s):
        try:
            new_s = re.findall(r'\[.*?\]', s)[0]
        except:
            pos = s.find("C:")
            new_s = s[pos+2:]
        if new_s[0] == "[" and new_s[-1] == "]":
            new_s = new_s[1:-1]
        l = new_s.split(",")
        l = [e.strip() for e in l]
        # remove quotes
        ll = [w[1:-1] for w in l if w[0] == "'" and w[-1] == "'"]
        return ll
    

    def parse_phrase(self, s):
        pos = s.find("Output:")
        if pos < 0:
            new_s = s
        else:
            new_s = s[pos+len("Output:"):]
        new_s = new_s.strip()
        return new_s
