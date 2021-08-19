import torch
from loguru import logger


def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def save_state_dicts(checkpoint_file, epoch=None, best_test_acc=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if best_test_acc is not None:
        checkpoint['best_test_acc'] = best_test_acc

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):  # 这个是resume用的
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    ## default load
    # if not is_eval:
    #     for key, value in kwargs.items():  # strict load everything
    #         value.load_state_dict(checkpoint[key])
    # else:
    for key, value in kwargs.items():  # load model only
        # if key != 'model':
        #     continue
        kwargs = {}
        if key == 'model':
            kwargs['strict'] = False
        load_status = value.load_state_dict(checkpoint[key], **kwargs)
        if load_status is not None and str(load_status) != '<All keys matched successfully>':
            logger.warning("Caught some errors when loading state_dict for {}:\n".format(key) +
                           f"missing keys: {load_status.missing_keys}\nunexpected_keys: {load_status.unexpected_keys}")
        # load_status = value.load_state_dict(checkpoint[key], strict=False)
        # if str(load_status) != '<All keys matched successfully>':
        #     logger.warning("Caught some errors when loading state_dict for {}:\n".format(key) +
        #                    f"missing keys: {load_status.missing_keys}\nunexpected_keys: {load_status.unexpected_keys}")

    # ## pretrain mismatch load
    # for key, value in kwargs.items():
    #     pretrained_dict = checkpoint[key]
    #     model_dict = value.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     assert (len([k for k, v in pretrained_dict.items()])!=0)
    #     model_dict.update(pretrained_dict)
    #     value.load_state_dict(model_dict)

    epoch = checkpoint.get('epoch')
    best_test_acc = checkpoint.get('best_test_acc')
    # if epoch:
    return epoch, best_test_acc  # might be None!
