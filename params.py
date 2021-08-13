from yacs.config import CfgNode
import argparse


def get_default_params():
    params = CfgNode()
    params.exp_name = 'default'
    params.exp_note = 'default'
    params.dataset = 'Nr3D'
    params.batch_size = 16

    return params


def get_parser_args():  # will cover other argument, 'python3 main.py batch_size 32'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "opts",
        help="Override config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)

    return parser.parse_args()


class CLIPSAT_Config:
    def __init__(self, yaml_path=None, parser_args=None):
        """
        priority: parser_args > yaml_file > get_default_params
        :param yaml_path:
        :param parser_args:
        """
        self.params = get_default_params()
        if type(yaml_path) == str:
            self.params.merge_from_file(yaml_path)
        if parser_args is not None:
            self.params.merge_from_list(parser_args.opts)

    def __str__(self):
        return self.params.dump()

    def dump(self, target_path):
        with open(target_path, 'w') as f:
            f.write(self.params.dump())


if __name__ == '__main__':
    cfg = CLIPSAT_Config(parser_args=get_parser_args())
    print(str(cfg))
