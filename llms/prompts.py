from util.word_utils import remove_whitespace



def gen_absent_unsuitable_object_prompt(caption):
    prompt = f"""
    Given a picture of {caption}. \
    List some objects that are definitely not in the image and unsuitable for appearing in this scene. \
    Output only a python list of objects.
    """
    prompt = remove_whitespace(prompt)
    return prompt


def gen_absent_object_prompt(caption):
    prompt = f"""
    You are an expert at inferring the probability of an object appearing in a given scene description. \
    Given a picture of ##{caption}##. You must do the following steps: \
    1. Identify the object set A depicted in the image. \
    2. Identify object set B that are not mentioned in the text description but are likely to be present in the picture. \
    3. Deduce less likely object set C with up to 10 objects that could feasibly be present in this scene, without showing in A and B. \
    4. Print out object set C as a Python list.
    Here is an examples: \
    caption: Two men smiling in front of two giraffes in an enclosure.
    object set A: person, giraffe. \
    object set B: grass, tree, sky. \
    object set C: elephant, wolf, cat, building, ...
    output: ['elephant', 'cat', 'wolf', 'tiger', 'river']
    """
    prompt = remove_whitespace(prompt)
    return prompt


def gen_halluc_phrase_prompt(phrase, object):
    prompt = f"""
    Given a phrase and an object, first, locate the target object in phrase. Second, replace \
    the target object with the given object. Phrase: {phrase}, object: {object}. \
    Only new phrases are output.
    """
    # example: phrase: middle zebra, object: elephant, output: ["middle elephant"]
    prompt = remove_whitespace(prompt)
    return prompt

