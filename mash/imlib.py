# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
from PIL import Image
from functools import wraps
import time

class imlib():
    def __init__(self, mode='RGB', fmt='CHW', lib='cv2', force_color=True):
        assert mode.upper() in ('RGB', 'L', 'Y', 'RAW')
        self.mode = mode.upper()

        assert fmt.upper() in ('HWC', 'CHW', 'NHWC', 'NCHW')
        self.fmt = 'CHW' if fmt.upper() in ('CHW', 'NCHW') else 'HWC'

        assert lib.lower() in ('cv2', 'pillow')
        self.lib = lib.lower()

        self.force_color = force_color

        self.dtype = np.uint8

        self._imread = getattr(self, '_imread_%s_%s'%(self.lib, self.mode))
        self._imwrite = getattr(self, '_imwrite_%s_%s'%(self.lib, self.mode))
        self._trans_batch = getattr(self, '_trans_batch_%s_%s'
                                    % (self.mode, self.fmt))
        self._trans_image = getattr(self, '_trans_image_%s_%s'
                                    % (self.mode, self.fmt))
        self._trans_back = getattr(self, '_trans_back_%s_%s'
                                   % (self.mode, self.fmt))

    def _imread_cv2_RGB(self, path):
        return cv2_imread(path, cv2.IMREAD_COLOR)[..., ::-1]
    def _imread_cv2_RAW(self, path):
        return cv2_imread(path, -1)
    def _imread_cv2_Y(self, path):
        if self.force_color:
            img = cv2_imread(path, cv2.IMREAD_COLOR)
        else:
            img = cv2_imread(path, cv2.IMREAD_ANYCOLOR)
        if len(img.shape) == 2:
            return np.expand_dims(img, 3)
        elif len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError('The dimension should be either 2 or 3.')
    def _imread_cv2_L(self, path):
        return cv2_imread(path, cv2.IMREAD_GRAYSCALE)
    def _imread_pillow_RGB(self, path):
        img = Image.open(path)
        im = np.array(img.convert(self.mode))
        img.close()
        return im
    _imread_pillow_L = _imread_pillow_RGB
    # WARNING: the RGB->YCbCr procedure of PIL may be different with OpenCV
    def _imread_pillow_Y(self, path):
        img = Image.open(path)
        if img.mode == 'RGB':
            im = np.array(img.convert('YCbCr'))
        elif img.mode == 'L':
            if self.force_color:
                im = np.array(img.convert('RGB').convert('YCbCr'))
            else:
                im = np.expand_dims(np.array(img), 3)
        else:
            img.close()
            raise NotImplementedError('Only support RGB and gray images now.')
        img.close()
        return im

    def _imwrite_cv2_RGB(self, image, path):
        cv2.imwrite(path, image[..., ::-1])
    def _imwrite_cv2_RAW(self, image, path):
        pass
    def _imwrite_cv2_Y(self, image, path):
        if image.shape[2] == 1:
            cv2.imwrite(path, image[..., 0])
        elif image.shape[2] == 3:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR))
        else:
            raise ValueError('There should be 1 or 3 channels.')
    def _imwrite_cv2_L(self, image, path):
        cv2.imwrite(path, image)
    def _imwrite_pillow_RGB(self, image, path):
        Image.fromarray(image).save(path)
    _imwrite_pillow_L = _imwrite_pillow_RGB
    def _imwrite_pillow_Y(self, image, path):
        if image.shape[2] == 1:
            self._imwrite_pillow_L(np.squeeze(image, 2), path)
        elif image.shape[2] == 3:
            Image.fromarray(image, mode='YCbCr').convert('RGB').save(path)
        else:
            raise ValueError('There should be 1 or 3 channels.')

    def _trans_batch_RGB_HWC(self, images):
        return np.ascontiguousarray(images)
    def _trans_batch_RGB_CHW(self, images):
        return np.ascontiguousarray(np.transpose(images, (0, 3, 1, 2)))
    _trans_batch_RAW_HWC = _trans_batch_RGB_HWC
    _trans_batch_RAW_CHW = _trans_batch_RGB_CHW
    _trans_batch_Y_HWC = _trans_batch_RGB_HWC
    _trans_batch_Y_CHW = _trans_batch_RGB_CHW
    def _trans_batch_L_HWC(self, images):
        return np.ascontiguousarray(np.expand_dims(images, 3))
    def _trans_batch_L_CHW(slef, images):
        return np.ascontiguousarray(np.expand_dims(images, 1))

    def _trans_image_RGB_HWC(self, image):
        return np.ascontiguousarray(image)
    def _trans_image_RGB_CHW(self, image):
        return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    _trans_image_RAW_HWC = _trans_image_RGB_HWC
    _trans_image_RAW_CHW = _trans_image_RGB_CHW
    _trans_image_Y_HWC = _trans_image_RGB_HWC
    _trans_image_Y_CHW = _trans_image_RGB_CHW
    def _trans_image_L_HWC(self, image):
        return np.ascontiguousarray(np.expand_dims(image, 2))
    def _trans_image_L_CHW(self, image):
        return np.ascontiguousarray(np.expand_dims(image, 0))

    def _trans_back_RGB_HWC(self, image):
        return image
    def _trans_back_RGB_CHW(self, image):
        return np.transpose(image, (1, 2, 0))
    _trans_back_RAW_HWC = _trans_back_RGB_HWC
    _trans_back_RAW_CHW = _trans_back_RGB_CHW
    _trans_back_Y_HWC = _trans_back_RGB_HWC
    _trans_back_Y_CHW = _trans_back_RGB_CHW
    def _trans_back_L_HWC(self, image):
        return np.squeeze(image, 2)
    def _trans_back_L_CHW(self, image):
        return np.squeeze(image, 0)

    img_ext = ('png', 'PNG', 'jpg', 'JPG', 'bmp', 'BMP', 'jpeg', 'JPEG')

    def is_image(self, fname):
        return any(fname.endswith(i) for i in self.img_ext)

    def read(self, paths):
        if isinstance(paths, (list, tuple)):
            images = [self._imread(path) for path in paths]
            return self._trans_batch(np.array(images))
        return self._trans_image(self._imread(paths))

    def back(self, image):
        return self._trans_back(image)

    def write(self, image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._imwrite(self.back(image), path)

def read_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(30):
            try:
                ret = func(*args, **kwargs)
                if ret is None:
                    raise OSError()
                else:
                    break
            except OSError:
                print('%s OSError' % str(args))
                time.sleep(1)
        return ret
    return wrapper

@read_until_success
def cv2_imread(*args, **kwargs):
    return cv2.imread(*args, **kwargs)