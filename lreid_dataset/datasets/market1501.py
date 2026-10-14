from __future__ import division, print_function, absolute_import
import json
from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples
# from reid.utils.serialization import read_json, write_json
# from lreid_dataset.datasets import ImageDataset
import re
import glob
import os.path as osp

class IncrementalSamples4market(IncrementalPersonReIDSamples):
    '''
    Market 数据集处理类
    '''
    _junk_pids = [0, -1]  # 表示垃圾图像的 ID 列表
    dataset_dir = 'market1501'  # 数据集的目录名称
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'  # 数据集下载 URL

    def __init__(self, datasets_root, cam_filter, relabel=True, combineall=False,
                 initial_class_ratio=1.0, class_split_file=None):
        # 初始化方法，设置根目录、是否重新标记 ID 以及是否合并所有数据
        self.relabel = relabel  # 是否重新标记 ID
        self.combineall = combineall  # 是否合并所有训练数据（包括查询和库）
        self.initial_class_ratio = initial_class_ratio
        self.initial_pids = set()
        self.selected_pids = self.load_class_split(class_split_file)
        root = osp.join(datasets_root, self.dataset_dir, 'Market-1501-v15.09.15')  

        # 构建数据集的根路径
        self.train_dir = osp.join(root, 'bounding_box_train')  # 训练集目录
        self.query_dir = osp.join(root, 'query')  # 查询集目录
        self.gallery_dir = osp.join(root, 'bounding_box_test')  # 库集目录


        self.check_before_run()
        self._select_initial_identities()

        train, cam_num_classes = self.process_cam_dir(self.train_dir, cam_filter)  # 处理训练集，并重新标记 ID
        query = self.process_dir(self.query_dir, cam_filter, relabel=False)  # 处理查询集，不重新标记 ID
        gallery = self.process_dir(self.gallery_dir, relabel=False)  # 处理库集，不重新标记 ID
        self.train, self.query, self.gallery, self.num_classes = train, query, gallery, cam_num_classes  # 将处理后的数据集赋值给类属性
        self._show_info(train, query, gallery)  # 显示数据集的基本信息

    @staticmethod
    def load_class_split(split_file):
        if split_file is None:
            return None
        with open(split_file, 'r') as f:
            split = json.load(f)
        if split.get('order') != [0, 1, 2, 4, 5]:
            raise ValueError('Invalid camera order in {}'.format(split_file))
        return {int(cam): set(item['pids'])
                for cam, item in split['cameras'].items()}

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _select_initial_identities(self):
        """Select camera-0 identities without constructing global labels."""
        if self.selected_pids is not None:
            return

        img_paths = sorted(glob.glob(osp.join(self.train_dir, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        cam0_pids = set()
        for path in img_paths:
            pid, camid = map(int, pattern.search(path).groups())
            if pid > 0 and camid == 1:
                cam0_pids.add(pid)
        cam0_pids = sorted(cam0_pids)
        num_initial_classes = max(1, int(len(cam0_pids) * self.initial_class_ratio))
        self.initial_pids = set(cam0_pids[:num_initial_classes])

    def process_dir(self, dir_path, cam_filter=None, relabel=False):
        # 处理数据集文件夹中的图像，并返回包含图像路径、行人 ID 和摄像头 ID 的列表
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))  # 获取文件夹中所有图片的路径
        pattern = re.compile(r'([-\d]+)_c(\d)')  # 定义用于提取行人 ID 和摄像头 ID 的正则表达式
        data = []  # 用于存储处理后的图像数据
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())  # 提取图片的行人 ID 和摄像头 ID
            if pid <= 0:
                continue  # 忽略垃圾图像
            camid -= 1  # 摄像头 ID 从 0 开始计数
            if cam_filter is not None:
                if camid == cam_filter:
                    data.append((img_path, pid, camid, 0))  # 将图像路径、行人 ID、摄像头 ID 以及第 4 列的数据（0）添加到列表中
            else:
                data.append((img_path,pid,camid,0))

        return data  # 返回处理后的图像数据

    def process_cam_dir(self, dir_path, cam_filter):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        if not isinstance(cam_filter, int):
            raise ValueError('CIOR expects one camera per incremental step')

        raw_samples = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            camid -= 1
            if pid <= 0 or camid != cam_filter:
                continue
            if self.selected_pids is not None:
                if pid not in self.selected_pids.get(cam_filter, set()):
                    continue
            elif cam_filter == 0 and pid not in self.initial_pids:
                continue
            raw_samples.append((img_path, pid, camid, 0))

        # Each camera owns an independent local identity space.  Zero-based
        # labels are used internally because CrossEntropyLoss requires them.
        local_pid = {
            pid: label
            for label, pid in enumerate(sorted({sample[1] for sample in raw_samples}))
        }
        dataset = [(path, local_pid[pid], camid, domain)
                   for path, pid, camid, domain in raw_samples]
        return dataset, len(local_pid)
