import warnings
from functools import partial

import numpy as np
import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n


def my_collate_fn(batch, not_stacked_keys: list):  # specific for dict Dataset return
    # print(colored('seen in collate_fn: {}'.format(batch), 'red'))
    elem = batch[0]
    assert isinstance(elem, dict)  # elem must be a dict
    all_keys = elem.keys()
    # not_stacked_keys = ['data2']
    stacked_keys = list(set(all_keys) - set(not_stacked_keys))
    res = {key: default_collate([d[key] for d in batch]) for key in stacked_keys}

    # not stacked keys will serve as a normal Python List
    res.update({key: [element[key] for element in batch] for key in not_stacked_keys})
    return res


def dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None,
                          not_stacked_keys: list = None):
    """

    """
    batch_size_multiplier = 1 if split == 'train' else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == 'train' and len(dataset) % b_size == 1:
        print('dropping last batch during training')
        drop_last = True

    shuffle = split == 'train'

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    kwargs = {
        'batch_size': b_size,
        'num_workers': n_workers,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'pin_memory': pin_memory,
        'worker_init_fn': worker_init_fn
    }

    if split == 'train' and not_stacked_keys is not None:  # change collate_fn only when training
        # make a partial function
        self_collate = partial(my_collate_fn, not_stacked_keys=not_stacked_keys)
        kwargs.update({'collate_fn': self_collate})

    # data_loader = DataLoader(dataset,
    #                          batch_size=b_size,
    #                          num_workers=n_workers,
    #                          shuffle=shuffle,
    #                          drop_last=drop_last,
    #                          pin_memory=pin_memory,
    #                          worker_init_fn=worker_init_fn)

    data_loader = DataLoader(dataset, **kwargs)

    return data_loader


def extractor_dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    WARNING! DO NOT TOUCH THIS! THIS IS FOR EXTRACTOR EXCLUSIVELY!
    """
    batch_size_multiplier = 64
    b_size = int(batch_size_multiplier)

    drop_last = False
    # if split == 'train' and len(dataset) % b_size == 1:
    #     print('dropping last batch during training')
    #     drop_last = True

    shuffle = False

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    data_loader = DataLoader(dataset,
                             batch_size=b_size,
                             num_workers=n_workers,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn)
    return data_loader


def sample_scan_object(object, n_points):
    sample = object.sample(n_samples=n_points)
    return np.concatenate([sample['xyz'], sample['color']], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.ones(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples


def pad_samples_fusion(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.ones(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples, n_pad


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        instance_labels.extend(['pad'] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in instance_labels])

    return instance_labels


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        # xyz = segmented_objects[:, :, :3]
        # mean_center = xyz.mean(axis=1)
        # xyz -= np.expand_dims(mean_center, 1)
        # max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        # max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        # xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        # segmented_objects[:, :, :3] = xyz
        # return segmented_objects
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        # segmented_objects[:, :, :3] = xyz
        ## if include scale in scale vector
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz
        mean_center = np.concatenate((mean_center, np.expand_dims(max_dist, 1)), axis=1)

    return segmented_objects, mean_center
