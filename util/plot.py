import os
import os.path as osp
from PIL import Image, ImageDraw, ImageFont
from textwrap3 import wrap


def visualize_image_info(image_name, boxes=None, phrase=None, collect_box_text=None, draw_phrase=False, 
                         image_dir="data/refcoco/images/train2014", out_dir='output'):
    '''Visualize image box and phrase annotation, 
    given an index (not COCO image id) or image name.
    '''
    color_set = [(127,255,255), (255,255,127), (255,127,255), (255,0,0), (0,255,0), (0,0,255), \
                    (255,127,127), (127,255,127), (127,127,255), (127,0,0), (0,127,0), (0,0,127)]

    img_file = osp.join(image_dir, image_name)
    print('\nVisualizing image', img_file)
    img = Image.open(img_file)
    if img.mode == "L": # grayscale image
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
    fnt_large = ImageFont.truetype("STXINWEI.TTF", size=30)
    fnt_small = ImageFont.truetype("STXINWEI.TTF", size=15)
    if collect_box_text == {} or collect_box_text is None:
        collect_box_text = {}
        for box, text in zip(boxes, phrase):
            if box not in collect_box_text:
                collect_box_text[box] = set()
            collect_box_text[box].add(text)
    print('image width and height: ({}, {})'.format(img.width, img.height))
    for ii, (box, texts) in enumerate(collect_box_text.items()):
        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        w, h = int(box[2] - box[0]), int(box[3] - box[1])
        print('bbox {}: ({}, {}), ({}, {})'.format(ii+1, cx, cy, w, h))
        draw.rectangle(box, outline=color_set[ii], width=4)
        if not draw_phrase:
            draw.text((box[0]+15, box[1]+15), str(ii+1), fill=color_set[ii], font=fnt_large)
        # draw expression
        for c, text in enumerate(texts):
            print(text)
            if draw_phrase:
                draw.text((box[0]+10, box[1] + c*15), text, fill=color_set[ii], font=fnt_small)
        # draw bbox coordinate
        # box_str = '('+str(int(box[0]))+', '+str(int(box[1]))+'\t'+str(int(box[2]))+', '+str(int(box[3]))+')'
        # draw.text((box[0]+10, box[1] + (c+1)*15), box_str, fill=color_set[ii], font=fnt_small)
    os.makedirs(out_dir, exist_ok=True)
    img.save(f"{out_dir}/{image_name}")


def visualize_image_caption_abs_obj(image_name, caption, objs, out_dir='output',
                                    image_dir="data/refcoco/images/train2014"):
    '''Visualize image caption and absent objects.
    '''
    color_set = [(255,0,0), (0,255,0), (0,0,255)]
    fnt_small = ImageFont.truetype("STXINWEI.TTF", size=20)
    margin = offset = 10

    img_file = osp.join(image_dir, image_name)
    img = Image.open(img_file)
    if img.mode == "L": # grayscale image
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    obj_text = ",".join(objs)
    
    for line in wrap(caption, width=50):
        draw.text((margin, offset), line, font=fnt_small, fill=color_set[0])
        offset += fnt_small.getsize(line)[1]
    for line in wrap(obj_text, width=50):
        draw.text((margin, offset), line, font=fnt_small, fill=color_set[1])
        offset += fnt_small.getsize(line)[1]

    os.makedirs(out_dir, exist_ok=True)
    img.save(f"{out_dir}/caption_{image_name}")


def resize_image_with_aspect_ratio(image, new_width):
    """
    Resize a PIL image while maintaining its aspect ratio.

    :param image: PIL Image object
    :param new_width: New width in pixels
    :return: Resized PIL Image object
    """
    width, height = image.size
    new_height = int(height * (new_width / width))
    
    # Resize the image using the new dimensions
    resized_image = image.resize((new_width, new_height))
    return resized_image
