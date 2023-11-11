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
    # prompt = f"""
    # You are an expert at inferring the probability of an object appearing in a given scene description. \
    # Given the caption of a picture. You must do the following steps: \
    # 1. Identify the object set A depicted in the image. \
    # 2. Identify object set B that are not mentioned in the text description but are likely to be present in the picture. \
    # 3. Deduce less likely object set C with up to 10 objects that could feasibly be present in this scene, without showing in A and B. \
    # 4. Output object set C as a Python list.
    # Here is an examples: \
    # caption: Two men smiling in front of two giraffes in an enclosure.
    # object set A: person, giraffe. \
    # object set B: grass, tree, sky. \
    # output (set C): [elephant, cat, wolf, tiger, river, ...] \
    # caption: {caption}, ouput:
    # """
    prompt = remove_whitespace(prompt)
    return prompt


def gen_halluc_phrase_prompt(phrase, object):
    # prompt = f"""
    # Given a phrase and an object, replace the subject in the phrase with the object \
    # and make the new phrase conform to objective logic. Make sure the original subject does not exist in the output. \
    # Avoid using verbs in the output phrase. For example: \
    # Phrase: person on left yellow boots, object: flower. Output: flower on left. \
    # Phrase: woman standing inbetween the two guys, object: hot dog. Output: hot dog inbetween the two guys. \
    # Phrase: {phrase}, object: {object}. Output:
    # """
    prompt = f"""
    Given a phrase and an object, replace the subject in the phrase with the object \
    and make the new phrase conform to objective logic. Make sure the original subject does not exist in the output. \
    Avoid using verbs in the output phrase. For example: \
    Phrase: person on left yellow boots, object: flower. Output: flower on left. \
    Phrase: {phrase}, object: {object}. Output:
    """
    prompt = remove_whitespace(prompt)
    return prompt

