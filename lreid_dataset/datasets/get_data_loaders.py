from pathlib import Path

import lreid_dataset.datasets as datasets
from reid.utils.data import IterLoader
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _build_train_loaders(train_set, height, width, batch_size, workers,
                         num_instances):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5,
                        mean=[0.485, 0.456, 0.406]),
    ])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    use_identity_sampler = num_instances > 0
    sampler = (RandomMultipleGallerySampler(train_set, num_instances)
               if use_identity_sampler else None)
    sample_count = len(sampler) if sampler is not None else len(train_set)
    iterations = max(1, sample_count // batch_size)
    train_loader = IterLoader(
        DataLoader(
            Preprocessor(train_set, transform=train_transformer),
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=not use_identity_sampler,
            pin_memory=True,
            drop_last=sample_count >= batch_size,
        ),
        length=iterations,
    )
    init_loader = DataLoader(
        Preprocessor(train_set, transform=test_transformer),
        batch_size=128,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, init_loader


def get_data(cam_filter, data_dir, height, width, batch_size, workers,
             num_instances, name, initial_class_ratio=1.0,
             class_split_file=None):
    root = data_dir

    camera_ids = [int(camera) for camera in cam_filter.split("cam")[-1].split('+')]
    cam_filter = camera_ids[0] if len(camera_ids) == 1 else camera_ids
    if name == 'msmt17' and 1 in camera_ids:
        num_instances = 16

    dataset_args = ({'initial_class_ratio': initial_class_ratio,
                     'class_split_file': class_split_file}
                    if name == 'market1501' else {})
    dataset = datasets.create(name, cam_filter, root, **dataset_args)

    train_img_count = len(dataset.train)
    num_classes = dataset.num_classes
    train_set = sorted(dataset.train)
    train_loader, init_loader = _build_train_loaders(
        train_set, height, width, batch_size, workers, num_instances)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     transform=test_transformer),
        batch_size=128, num_workers=workers, shuffle=False, pin_memory=True)
    group_name = '_'.join(map(str, camera_ids))
    return [dataset, num_classes, train_loader, test_loader, init_loader,
            name + 'cam_' + group_name, train_img_count]


def apply_pseudo_labels(data_bundle, labels_by_filename, cfg):
    """Replace local labels with global pseudo labels and rebuild loaders."""
    dataset = data_bundle[0]
    missing = [sample[0] for sample in dataset.train
               if sample[0] not in labels_by_filename]
    if missing:
        raise KeyError('Missing pseudo labels for {} images'.format(len(missing)))

    dataset.train = [
        (fname, int(labels_by_filename[fname]), camid, domain)
        for fname, _, camid, domain in dataset.train
    ]
    dataset.num_classes = len({sample[1] for sample in dataset.train})
    train_set = sorted(dataset.train)
    train_loader, init_loader = _build_train_loaders(
        train_set,
        cfg.height,
        cfg.width,
        cfg.batch_size,
        cfg.workers,
        cfg.num_instances,
    )
    data_bundle[1] = dataset.num_classes
    data_bundle[2] = train_loader
    data_bundle[4] = init_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = cfg.height, cfg.width
    split_file = None
    if cfg.new_class_ratio is not None and training_set[0].startswith('market1501'):
        split_file = PROJECT_ROOT / 'analysis' / 'market1501_new_ratio_{}.json'.format(
            cfg.new_class_ratio)
        if not split_file.is_file():
            raise FileNotFoundError('Class split not found: {}'.format(split_file))
    training_loaders = [
        get_data(
            cam_filter,
            data_dir,
            height,
            width,
            cfg.batch_size,
            cfg.workers,
            cfg.num_instances,
            name='market1501',
            initial_class_ratio=cfg.initial_class_ratio,
            class_split_file=str(split_file) if split_file else None,
        )
        for cam_filter in training_set
    ]
    testing_loaders = [get_data(cam_filter, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances, name='market1501') for cam_filter in testing_only_set]
    return training_loaders, testing_loaders
