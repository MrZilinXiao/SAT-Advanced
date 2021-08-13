import os
import struct
import zlib


import cv2
import imageio
import numpy as np

from tqdm import tqdm

# to make sure compatibility with Python 3, replace .join with reduce + add
from operator import add
from functools import reduce

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f' * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(
            4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        # self.color_data = ''.join(struct.unpack('c' * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        # self.depth_data = ''.join(struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))
        self.color_data = reduce(add, struct.unpack('c' * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = reduce(add, struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename, max_frames=int(10e10)):
        self.version = 4
        self.load(filename, max_frames)

    def load(self, filename, max_frames):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            # self.sensor_name = ''.join(struct.unpack('c' * strlen, f.read(strlen)))
            self.sensor_name = reduce(add, struct.unpack('c' * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            num_frames = min(num_frames, max_frames)
            self.frames = []
            for i in tqdm(range(num_frames)):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_color_frame(self, image_size=None, frame_id=0):  # image_size: [w, h]
        color = self.frames[frame_id].decompress_color(self.color_compression_type)
        if image_size is not None:
            color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        return color


if __name__ == '__main__':
    sensor_data = SensorData('/data/ScanNet_copy/scans/scene0679_00/scene0679_00.sens')
    color_img_test = sensor_data.export_color_frame(frame_id=0)