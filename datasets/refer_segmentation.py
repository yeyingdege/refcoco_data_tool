# Builder for visual grouding datasets
import os
import os.path as osp
import numpy as np
from grounding_datasets import ReferSegDataset
from PIL import Image, ImageDraw, ImageFont
import transforms as T
import torch

class RefCOCO(ReferSegDataset):
    def __init__(self, data_root, im_dir, seg_dir, split, transforms, version="refcoco_unc",
                 max_query_len=40, bert_model='bert-base-uncased'):
        super(RefCOCO, self).__init__(
            data_root=data_root,
            im_dir=im_dir,
            seg_dir=seg_dir,
            dataset=version,
            split=split,
            max_query_len=max_query_len,
            bert_model=bert_model
        )
        self._transforms = transforms
        self.img_infos, self.box_phrases_dict = self.summarize_obj_ref_per_image()
    
    def __getitem__(self, idx):
        input_sample, target = super(RefCOCO, self).__getitem__(idx)
        target = {k: torch.as_tensor(v) for k, v in target.items()}
        # target['boxes'] = torch.as_tensor(target['boxes'])
        img = Image.fromarray(input_sample["img"])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        input_sample["img"] = img
        return input_sample, target

    def summarize_obj_ref_per_image(self,):
        img_infos = []
        box_phrases_dict = {}
        for i in range(len(self.images)):
            img_file, seg_file, bbox, phrase = self.images[i]
            if img_file not in box_phrases_dict:
                box_phrases_dict[img_file] = {"phrase":[], "boxes":[]}
            box_phrases_dict[img_file]["phrase"].append(phrase)
            box_phrases_dict[img_file]["boxes"].append(np.array([bbox], dtype=np.float32))
        print('Total {} images'.format(len(box_phrases_dict)))
        for iname in sorted(box_phrases_dict.keys()):
            info = {"img_file": iname,
                    "phrase": box_phrases_dict[iname]["phrase"],
                    "boxes": box_phrases_dict[iname]["boxes"]}
            img_infos.append(info)
        return img_infos, box_phrases_dict

    def visualize_image_info(self, i, out_dir='output'):
        '''Visualize image box and phrase annotation, 
        given an index (not COCO image id) or image name.
        '''
        if isinstance(i, int):
            info = self.img_infos[i]
            img_file = osp.join(self.im_dir, info["img_file"])
            phrase = info["phrase"]
            boxes = info["boxes"]
            suffix = info["img_file"].split('.')[0]
        elif isinstance(i, str):
            info = self.box_phrases_dict[i]
            img_file = osp.join(self.im_dir, i)
            phrase = info["phrase"]
            boxes = info["boxes"]
            suffix = i.split('.')[0]
        else:
            raise TypeError("Only support index or image name as input!")
        print('\nVisualizing image', img_file)
        img = Image.open(img_file)
        draw = ImageDraw.Draw(img)
        # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
        fnt = ImageFont.truetype("STXINWEI.TTF", size=15)
        collect_box_text = {}
        for box, text in zip(boxes, phrase):
            box = tuple(box[0])
            if box not in collect_box_text:
                collect_box_text[box] = set()
            collect_box_text[box].add(text)

        for ii, (box, texts) in enumerate(collect_box_text.items()):
            print('bbox {}: ({:.2f}, {:.2f}), ({:.2f}, {:.2f})'.format(ii, box[0], box[1], box[2], box[3]))
            draw.rectangle(box, outline='red')
            for c, text in enumerate(texts):
                print(text)
                draw.text((box[0], box[1] + c*15), text, fill='yellow', font=fnt)
            # box_str = '('+str(box[0])+', '+str(box[1])+'\t'+str(box[2])+', '+str(box[3])+')'
            # draw.text((box[0], box[1] + (c+1)*15), box_str, fill='yellow', font=fnt)
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        img.save(f"./{out_dir}/refcoco_{suffix}.jpg")


def make_refer_seg_transforms(img_size=224 ,max_img_size=1333 ,test=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not test:
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.RandomIntensitySaturation(),
            T.RandomResize([img_size], max_size=max_img_size),
            # T.RandomAffineTransform(degrees=(-5,5), translate=(0.1, 0.1),
            #                         scale=(0.9, 1.1)),
            normalize
        ])
    else:
        return T.Compose([
            T.RandomResize([img_size], max_size=max_img_size),
            normalize
        ])


def build_refcoco_segmentation(
        split='train', 
        version='refcoco_unc',
        data_root="./data/refcoco/anns",
        im_dir="./data/refcoco/images/train2014",
        seg_dir="./data/refcoco/masks",
        img_size=640, 
        max_img_size=640,
        bert_model='bert-base-uncased'
    ):
    '''
        'refcoco_unc'
        'refcoco+_unc'
        'refcocog_google'
        'refcocog_umd'
    '''
    istest = split != 'train'

    return RefCOCO(
        data_root=data_root,
        im_dir=im_dir,
        seg_dir=seg_dir,
        version=version,
        transforms=make_refer_seg_transforms(img_size, max_img_size, test=istest),
        split=split,
        bert_model=bert_model
    )

if __name__ == "__main__":
    # examples
    testB = build_refcoco_segmentation(split='testB')
    # for i in range(0, 750, 10):
    #     testB.visualize_image_info(i)
    testB.visualize_image_info("COCO_train2014_000000000154.jpg")