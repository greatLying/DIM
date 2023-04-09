'''Deep Implicit Infomax

'''

import argparse #解析命令行参数
from argparse import RawTextHelpFormatter
import logging #记录日志
import sys #处理系统参数

from cortex.main import run #cortex

#cortex_DIM
from cortex_DIM.evaluation_models.classification_eval import ClassificationEval
from cortex_DIM.evaluation_models.ndm_eval import NDMEval
from cortex_DIM.evaluation_models.msssim_eval import MSSSIMEval
from cortex_DIM.models.controller import Controller
from cortex_DIM.models.coordinates import CoordinatePredictor
from cortex_DIM.models.dim import GlobalDIM, LocalDIM
from cortex_DIM.models.prior_matching import PriorMatching


logger = logging.getLogger('DIM')


if __name__ == '__main__':

    #定义一个字典 mode_dict，包含了多种 DIM 模型的键值对，键是模型的名称，值是模型的类。
    mode_dict = dict(
        local=LocalDIM,
        glob=GlobalDIM,
        prior=PriorMatching,
        coordinates=CoordinatePredictor,
        classifier=ClassificationEval,
        ndm=NDMEval,
        msssim=MSSSIMEval
    )

    #从 mode_dict 中提取模型的名称（键），存放在 names 元组中。
    names = tuple(mode_dict.keys())

    #从每个模型的文档字符串（docstring）中提取第一行，并将其存放在 infos 列表中。接着将这些信息连接成一个字符串，以便在命令行参数的帮助信息中显示。
    infos = []
    for k in mode_dict.keys():
        mode = mode_dict[k]
        info = mode.__doc__.split('\n', 1)[0]  # Keep only first line of doctstring.
        infos.append('{}: {}'.format(k, info))
    infos = '\n\t'.join(infos)

    #创建一个命令行参数解析器，用于解析用户输入的模型参数
    models = []
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('models', nargs='+', choices=names,
                        help='Models used in Deep InfoMax. Choices are: \n\t{}'.format(infos))
    i = 1
    while True:
        arg = sys.argv[i]
        if arg in ('--help', '-h') and i == 1:
            i += 1
            break

        if arg.startswith('-'):
            break  # argument have begun
        i += 1
    #从命令行参数中解析 'models' 参数的值，从第二个元素（索引为1）开始，直到第 i 个元素（不包含）的命令行参数
    args = parser.parse_args(sys.argv[1:i])
    models = args.models
    #使用 list(set(models)) 对 models 进行去重处理，以防用户在命令行参数中输入了重复的模型名称。
    models = list(set(models))
    models = dict((k, mode_dict[k]) for k in models)

    sys.argv = [sys.argv[0]] + sys.argv[i:]
    controller = Controller(inputs=dict(inputs='data.images'), **models)
    run(controller)#cortex库
