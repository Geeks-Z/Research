import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    logs_name = "logs/{}/{}/{}".format(args["model_name"], args["dataset"],args["backbone_type"])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args,
    )
    # 保存训练集和测试集的数量
    # torch.save(data_manager.train_dataset_num, './res/' + str(args["dataset"]) +'_train_num.pth')
    # torch.save(data_manager.test_dataset_num, './res/' + str(args["dataset"]) + '_test_num.pth')
    model = factory.get_model(args["model_name"], args)
    logging.info("All params: {}".format(count_parameters(model._network)))
    logging.info(
        "Trainable params: {}".format(count_parameters(model._network, True))
    )
    model.train(data_manager)
    # torch.save(model._network.backbone.state_dict(), '/home/team/zhaohongwei/checkpoint/state_dict' + str(args["dataset"]) +'.pth')
    cnn_accy, nme_accy = model.eval_accuracy()
    print('Top1 Average Accuracy :', cnn_accy["top1"])
    print('Top5 Average Accuracy :', cnn_accy["top5"])

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
