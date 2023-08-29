import os
import os.path as osp
from PIL import Image, ImageDraw, ImageFont


def visualize_image_info(image_name, boxes, phrase, draw_phrase=False, 
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

