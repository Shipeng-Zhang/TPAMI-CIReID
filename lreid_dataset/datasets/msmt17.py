from __future__ import division, print_function, absolute_import
import collections
import os.path as osp
import glob
import re

from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples
# from reid.utils.serialization import read_json, write_json
from lreid_dataset.data.dataset import ImageDataset

# 原始摄像头从1开始编号，1~15
cam_num = 15
order_2 = [0,3,4,5,6,7,8,10,11,12,13,14,1,2,9]
class IncrementalSamples4msmt17(IncrementalPersonReIDSamples):
    '''
    Market Dataset
    '''
    dataset_dir = 'MSMT17'  # 数据集的目录名称

    def __init__(self, datasets_root, cam_filter, relabel=True, combineall=False):
        # 初始化方法，设置根目录、是否重新标记 ID 以及是否合并所有数据
        self.global_pid2lable={}
        self.next_pid_lable = 0
        self.relabel = relabel  # 是否重新标记 ID
        self.combineall = combineall  # 是否合并所有训练数据（包括查询和库）
        root = osp.join(datasets_root, self.dataset_dir)  

        # 构建数据集的根路径
        self.train_dir = osp.join(root, 'bounding_box_train')  # 训练集目录
        self.query_dir = osp.join(root, 'query')  # 查询集目录
        self.gallery_dir = osp.join(root, 'bounding_box_test')  # 库集目录


        self.check_before_run()
        self.get_global_lable(self.train_dir) # 建立全局pid-lable映射

        self.camera_group = ([cam_filter] if isinstance(cam_filter, int)
                             else list(cam_filter))
        train, cam_num_classes = self.process_cam_dir(self.train_dir, self.camera_group)  # 处理训练集，并重新标记 ID
        query = self.process_dir(self.query_dir, self.camera_group, relabel=False)  # 处理查询集，不重新标记 ID
        gallery = self.process_dir(self.gallery_dir, relabel=False)  # 处理库集，不重新标记 ID
        self.train, self.query, self.gallery, self.num_classes = train, query, gallery, cam_num_classes  # 将处理后的数据集赋值给类属性
        self._show_info(train, query, gallery)  # 显示数据集的基本信息
    
    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def get_global_lable(self,train_dir):
        img_paths = sorted(glob.glob(osp.join(train_dir,'*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        for cam_idx in order_2:
            for img_path in img_paths:
                pid,camid = map(int,pattern.search(img_path).groups())
                if pid < 0:
                    continue
                camid -= 1
                if camid != cam_idx:
                    continue
                if pid not in self.global_pid2lable:
                    self.global_pid2lable[pid] = self.next_pid_lable
                    self.next_pid_lable += 1
    
    def process_dir(self, dir_path, cam_filter=None, relabel=False):
        # 处理数据集文件夹中的图像，并返回包含图像路径、行人 ID 和摄像头 ID 的列表
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))  # 获取文件夹中所有图片的路径
        pattern = re.compile(r'([-\d]+)_c(\d+)')  # 定义用于提取行人 ID 和摄像头 ID 的正则表达式
        data = []  # 用于存储处理后的图像数据
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())  # 提取图片的行人 ID 和摄像头 ID
            if pid < 0:
                continue
            camid -= 1
            if cam_filter is not None :
                if camid in cam_filter:
                    data.append((img_path, pid, camid, 0))
            else:
                data.append((img_path, pid, camid, 0))
                
        return data

    def process_cam_dir(self, dir_path, camera_group):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))  # 获取文件夹中所有图片的路径
        pattern = re.compile(r'([-\d]+)_c(\d+)')  # 定义用于提取行人 ID 和摄像头 ID 的正则表达式

        datasets = collections.defaultdict(list)
        cam_num_classes = [] # 统计每个相机的类别数

        for cam_idx in range(cam_num):
            dataset = []
            pids = set()
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid < 0:
                    continue  # junk images are just ignored
                camid -= 1  # index starts from 0
                if camid!=cam_idx:
                    continue
                pid = self.global_pid2lable[pid]

                dataset.append((img_path, pid, camid, 0))
                pids.add(pid)

            datasets[cam_idx] = dataset
            cam_num_classes.append(len(pids))

        train = [sample for camera in camera_group for sample in datasets[camera]]
        num_classes = len({sample[1] for sample in train})
        return train, num_classes

class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'msmt17'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid,3))

        return data
