# Copyright (c) OpenMMLab. All rights reserved.
# Written by jbwang1997
# Reference: https://github.com/jbwang1997/BboxToolkit

import argparse
import codecs
import datetime
import itertools
import json
import logging
import os
import os.path as osp
import shutil
import time
from functools import partial, reduce
from math import ceil
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
try:
    import shapely.geometry as shgeo
except ImportError:
    shgeo = None
class dota2ship():
    def __init__(self, root, output_path):
        self.img_dirs = [osp.join(root, "images")]
        self.ann_dirs = [osp.join(root, "annfiles")]
        self.sizes = [1024]
        self.gaps = [512]
        self.rates = [1.]  # mutil
        self.img_rate_thr = 0.6
        self.iof_thr = 0.7
        self.no_padding = True
        self.padding_value = [0]
        self.save_dir = output_path
        self.img_ext = ".png"
        self.save_imgs = osp.join(self.save_dir, 'images')
        self.save_files=osp.join(self.save_dir, 'annfiles')
        os.makedirs(self.save_imgs, exist_ok=True)
        os.makedirs(self.save_files, exist_ok=True)
        self.nproc=10

    def get_sliding_window(info, sizes, gaps, img_rate_thr):
        """Get sliding windows.

        Args:
            info (dict): Dict of image's width and height.
            sizes (list): List of window's sizes.
            gaps (list): List of window's gaps.
            img_rate_thr (float): Threshold of window area divided by image area.

        Returns:
            list[np.array]: Information of valid windows.
        """
        eps = 0.01
        windows = []
        width, height = info['width'], info['height']
        for size, gap in zip(sizes, gaps):
            assert size > gap, f'invaild size gap pair [{size} {gap}]'
            step = size - gap  ###ss:1024-200

            x_num = 1 if width <= size else ceil((width - size) / step + 1)
            x_start = [step * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size > width:
                if width - x_start[-1] < 500:  ###
                    x_start = x_start[:-1]
                else:
                    x_start[-1] = width - size

            y_num = 1 if height <= size else ceil((height - size) / step + 1)
            y_start = [step * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size > height:
                if height - y_start[-1] < 500:  ###
                    y_start = y_start[:-1]
                else:
                    y_start[-1] = height - size

            start = np.array(
                list(itertools.product(x_start, y_start)), dtype=np.int64)
            stop = start + size
            windows.append(np.concatenate([start, stop], axis=1))
        windows = np.concatenate(windows, axis=0)

        img_in_wins = windows.copy()
        img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
        img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
        img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                    (img_in_wins[:, 3] - img_in_wins[:, 1])
        win_areas = (windows[:, 2] - windows[:, 0]) * \
                    (windows[:, 3] - windows[:, 1])
        img_rates = img_areas / win_areas
        if not (img_rates > img_rate_thr).any():
            max_rate = img_rates.max()
            img_rates[abs(img_rates - max_rate) < eps] = 1
        return windows[img_rates > img_rate_thr]


    def poly2hbb(self,polys):
        """Convert polygons to horizontal bboxes.

        Args:
            polys (np.array): Polygons with shape (N, 8)

        Returns:
            np.array: Horizontal bboxes.
        """
        shape = polys.shape
        polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
        lt_point = np.min(polys, axis=-2)
        rb_point = np.max(polys, axis=-2)
        return np.concatenate([lt_point, rb_point], axis=-1)


    def bbox_overlaps_iof(self,bboxes1, bboxes2, eps=1e-6):
        """Compute bbox overlaps (iof).

        Args:
            bboxes1 (np.array): Horizontal bboxes1.
            bboxes2 (np.array): Horizontal bboxes2.
            eps (float, optional): Defaults to 1e-6.

        Returns:
            np.array: Overlaps.
        """
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]

        if rows * cols == 0:
            return np.zeros((rows, cols), dtype=np.float32)

        hbboxes1 = self.poly2hbb(bboxes1)
        hbboxes2 = bboxes2
        hbboxes1 = hbboxes1[:, None, :]
        lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
        rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
        wh = np.clip(rb - lt, 0, np.inf)
        h_overlaps = wh[..., 0] * wh[..., 1]

        l, t, r, b = [bboxes2[..., i] for i in range(4)]
        polys2 = np.stack([l, t, r, t, r, b, l, b], axis=-1)
        if shgeo is None:
            raise ImportError('Please run "pip install shapely" '
                              'to install shapely first.')
        sg_polys1 = [shgeo.Polygon(p) for p in bboxes1.reshape(rows, -1, 2)]
        sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]
        overlaps = np.zeros(h_overlaps.shape)
        for p in zip(*np.nonzero(h_overlaps)):
            overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
        unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
        unions = unions[..., None]

        unions = np.clip(unions, eps, np.inf)
        outputs = overlaps / unions
        if outputs.ndim == 1:
            outputs = outputs[..., None]
        return outputs


    def get_window_obj(self,info, windows, iof_thr):
        """

        Args:
            info (dict): Dict of bbox annotations.
            windows (np.array): information of sliding windows.
            iof_thr (float): Threshold of overlaps between bbox and window.

        Returns:
            list[dict]: List of bbox annotations of every window.
        """
        bboxes = info['ann']['bboxes']
        iofs = self.bbox_overlaps_iof(bboxes, windows)

        window_anns = []
        for i in range(windows.shape[0]):
            win_iofs = iofs[:, i]
            pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

            win_ann = dict()
            for k, v in info['ann'].items():
                try:
                    win_ann[k] = v[pos_inds]
                except TypeError:
                    win_ann[k] = [v[i] for i in pos_inds]
            win_ann['trunc'] = win_iofs[pos_inds] < 1
            window_anns.append(win_ann)
        return window_anns


    def crop_and_save_img(self,info, windows, window_anns, img_dir, no_padding,
                          padding_value, save_dir, anno_dir, img_ext):
        """

        Args:
            info (dict): Image's information.
            windows (np.array): information of sliding windows.
            window_anns (list[dict]): List of bbox annotations of every window.
            img_dir (str): Path of images.
            no_padding (bool): If True, no padding.
            padding_value (tuple[int|float]): Padding value.
            save_dir (str): Save filename.
            anno_dir (str): Annotation filename.
            img_ext (str): Picture suffix.

        Returns:
            list[dict]: Information of paths.
        """
        img = cv2.imread(osp.join(img_dir, info['filename']))
        ### pass the tittle picture
        if img.shape[0] * img.shape[1] <= 1500 * 1500:
            # print(osp.join(img_dir, info['filename']),osp.join(save_dir, info['filename']))
            shutil.copy(osp.join(img_dir, info['filename']), osp.join(save_dir, info['filename']))
            shutil.copy(osp.join(img_dir, info['filename']).replace(".png", '.txt').replace("images", "labelTxt"),
                        osp.join(anno_dir, info['filename']).replace(".png", '.txt'))
            return []

        patch_infos = []
        for i in range(windows.shape[0]):
            patch_info = dict()
            for k, v in info.items():
                if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                    patch_info[k] = v

            window = windows[i]
            x_start, y_start, x_stop, y_stop = window.tolist()
            patch_info['x_start'] = x_start
            patch_info['y_start'] = y_start
            patch_info['id'] = \
                info['id'] + '__' + str(x_stop - x_start) + \
                '__' + str(x_start) + '___' + str(y_start)
            patch_info['ori_id'] = info['id']

            ann = window_anns[i]
            ann['bboxes'] = self.translate(ann['bboxes'], -x_start, -y_start)
            patch_info['ann'] = ann

            patch = img[y_start:y_stop, x_start:x_stop]
            if not no_padding:
                height = y_stop - y_start
                width = x_stop - x_start
                if height > patch.shape[0] or width > patch.shape[1]:
                    padding_patch = np.empty((height, width, patch.shape[-1]),
                                             dtype=np.uint8)
                    if not isinstance(padding_value, (int, float)):
                        assert len(padding_value) == patch.shape[-1]
                    padding_patch[...] = padding_value
                    padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                    patch = padding_patch
            patch_info['height'] = patch.shape[0]
            patch_info['width'] = patch.shape[1]

            bboxes_num = patch_info['ann']['bboxes'].shape[0]
            outdir = os.path.join(anno_dir, patch_info['id'] + '.txt')
            patch_info['filename'] = patch_info['id'] + img_ext
            patch_infos.append(patch_info)

            ### pass non-ship picture
            if bboxes_num == 0:
                continue

            cv2.imwrite(osp.join(save_dir, patch_info['id'] + img_ext), patch)

            with codecs.open(outdir, 'w', 'utf-8') as f_out:
                if bboxes_num == 0:
                    pass
                else:
                    for idx in range(bboxes_num):
                        obj = patch_info['ann']
                        outline = ' '.join(list(map(str, obj['bboxes'][idx])))
                        diffs = str(
                            obj['diffs'][idx]) if not obj['trunc'][idx] else '2'
                        outline = outline + ' ' + obj['labels'][idx] + ' ' + diffs
                        f_out.write(outline + '\n')

        return patch_infos


    def single_split(self,info, sizes, gaps, img_rate_thr, iof_thr, no_padding,
                     padding_value, save_dir, anno_dir, img_ext):
        """

        Args:
            arguments (object): Parameters.
            sizes (list): List of window's sizes.
            gaps (list): List of window's gaps.
            img_rate_thr (float): Threshold of window area divided by image area.
            iof_thr (float): Threshold of overlaps between bbox and window.
            no_padding (bool): If True, no padding.
            padding_value (tuple[int|float]): Padding value.
            save_dir (str): Save filename.
            anno_dir (str): Annotation filename.
            img_ext (str): Picture suffix.
            lock (object): Lock of Manager.
            prog (object): Progress of Manager.
            total (object): Length of infos.
            logger (object): Logger.

        Returns:
            list[dict]: Information of paths.
        """
        info, img_dir = info,self.img_dirs
        windows = self.get_sliding_window(info, sizes, gaps, img_rate_thr)
        window_anns = self.get_window_obj(info, windows, iof_thr)
        patch_infos = self.crop_and_save_img(info, windows, window_anns, img_dir,
                                        no_padding, padding_value, save_dir,
                                        anno_dir, img_ext)
        # assert patch_infos

        # lock.acquire()
        # prog.value += 1
        # msg = f'({prog.value / total:3.1%} {prog.value}:{total})'
        # msg += ' - ' + f"Filename: {info['filename']}"
        # msg += ' - ' + f"width: {info['width']:<5d}"
        # msg += ' - ' + f"height: {info['height']:<5d}"
        # msg += ' - ' + f"Objects: {len(info['ann']['bboxes']):<5d}"
        # msg += ' - ' + f'Patches: {len(patch_infos)}'
        # logger.info(msg)
        # lock.release()

        return patch_infos


    def setup_logger(self,log_path):
        """Setup logger.

        Args:
            log_path (str): Path of log.

        Returns:
            object: Logger.
        """
        logger = logging.getLogger('img split')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = osp.join(log_path, now + '.log')
        handlers = [logging.StreamHandler(), logging.FileHandler(log_path, 'w')]

        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


    def translate(self,bboxes, x, y):
        """Map bboxes from window coordinate back to original coordinate.

        Args:
            bboxes (np.array): bboxes with window coordinate.
            x (float): Deviation value of x-axis.
            y (float): Deviation value of y-axis

        Returns:
            np.array: bboxes with original coordinate.
        """
        dim = bboxes.shape[-1]
        translated = bboxes + np.array([x, y] * int(dim / 2), dtype=np.float32)
        return translated


    def load_dota(self,img_dir, ann_dir=None, nproc=10):
        """Load DOTA dataset.

        Args:
            img_dir (str): Path of images.
            ann_dir (str): Path of annotations.
            nproc (int): number of processes.

        Returns:
            list: Dataset's contents.
        """
        assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
        assert ann_dir is None or osp.isdir(
            ann_dir), f'The {ann_dir} is not an existing dir!'

        print('Starting loading DOTA dataset information.')
        start_time = time.time()
        _load_func = partial(self._load_dota_single, img_dir=img_dir, ann_dir=ann_dir)
        if nproc > 1:
            pool = Pool(nproc)
            contents = pool.map(_load_func, os.listdir(img_dir))
            pool.close()
        else:
            contents = list(map(_load_func, os.listdir(img_dir)))
        contents = [c for c in contents if c is not None]
        end_time = time.time()
        print(f'Finishing loading DOTA, get {len(contents)} iamges,',
              f'using {end_time - start_time:.3f}s.')

        return contents


    def _load_dota_single(self,imgfile, img_dir, ann_dir):
        """Load DOTA's single image.

        Args:
            imgfile (str): Filename of single image.
            img_dir (str): Path of images.
            ann_dir (str): Path of annotations.

        Returns:
            dict: Content of single image.
        """
        img_id, ext = osp.splitext(imgfile)
        if ext not in ['.jpg', '.JPG', '.png', '.tif', '.bmp']:
            return None

        imgpath = osp.join(img_dir, imgfile)
        size = Image.open(imgpath).size
        txtfile = None if ann_dir is None else osp.join(ann_dir, img_id + '.txt')
        content = self._load_dota_txt(txtfile)

        content.update(
            dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
        return content


    def _load_dota_txt(self,txtfile):
        """Load DOTA's txt annotation.

        Args:
            txtfile (str): Filename of single txt annotation.

        Returns:
            dict: Annotation of single image.
        """
        gsd, bboxes, labels, diffs = None, [], [], []
        if txtfile is None:
            pass
        elif not osp.isfile(txtfile):
            print(f"Can't find {txtfile}, treated as empty txtfile")
        else:
            with open(txtfile, 'r') as f:
                for line in f:
                    if line.startswith('gsd'):
                        num = line.split(':')[-1]
                        try:
                            gsd = float(num)
                        except ValueError:
                            gsd = None
                        continue

                    items = line.split(' ')
                    if len(items) >= 9:  ###pass the diffs over 0  and int(items[9]) ==0
                        bboxes.append([float(i) for i in items[:8]])
                        labels.append(items[8])
                        diffs.append(int(items[9]) if len(items) == 10 else 0)

        bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
            np.zeros((0, 8), dtype=np.float32)
        diffs = np.array(diffs, dtype=np.int64) if diffs else \
            np.zeros((0,), dtype=np.int64)
        ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
        return dict(gsd=gsd, ann=ann)


    def run(self,):
        """Main function of image split."""

        if self.ann_dirs is None:
            self.ann_dirs = [None for _ in range(len(self.img_dirs))]
        padding_value = self.padding_value[0] \
            if len(self.padding_value) == 1 else self.padding_value
        sizes, gaps = [], []
        for rate in self.rates:
            sizes += [int(size / rate) for size in self.sizes]
            gaps += [int(gap / rate) for gap in self.gaps]
        logger = self.setup_logger(self.save_dir)

        print('Loading original data!!!')
        infos, img_dirs = [], []
        for img_dir, ann_dir in zip(self.img_dirs, self.ann_dirs):
            _infos = self.load_dota(img_dir=img_dir, ann_dir=ann_dir, nproc=self.nproc)
            _img_dirs = [img_dir for _ in range(len(_infos))]
            infos.extend(_infos)
            img_dirs.extend(_img_dirs)

        print('Start splitting images!!!')
        info =


        self.single_split(info=info,sizes=sizes,gaps=gaps,img_rate_thr=self.img_rate_thr,
                          iof_thr=self.iof_thr,no_padding=self.no_padding,
                          padding_value=padding_value, save_dir=self.save_dir,
                          anno_dir=self.ann_dirs, img_ext=self.img_ext
                          )

if __name__ == '__main__':
    raster = r"D:\omqcode\dataprocess\rgb\shp2falseexample"
    output = r"D:\omqcode\dataprocess\rgb\shp2falseexample\output"
    r = dota2ship(raster,output)
    r.run()

"""
  "img_dirs": [
    "G:\\omq\\data\\dota\\trainsplit/images"
  ],
  "ann_dirs": [
    "G:\\omq\\data\\dota\\trainsplit/labelTxt/"
  ],


"img_dirs": [
    "D:\\omqcode\\dataprocess\\test/images"
  ],
  "ann_dirs": [
    "D:\\omqcode\\dataprocess\\test/annfiles/"
  ],
"""


