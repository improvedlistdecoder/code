import os
import sys
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from util.log_util import get_logger
from util.data_util import SignalDataset
from util.data_util import SignalTestset
from util.conf_util import get_default_conf
from model.cycnet import cycnet

if len(sys.argv) == 2:
    conf_name = sys.argv[1]
    print("test conf_name:", conf_name)
    conf = get_default_conf(f"./config/{conf_name}.yaml")
else:
    print("default")
    conf = get_default_conf()

if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
else:
    device = torch.device("cpu")

logger = get_logger(conf["para"]["logger_name"])

def test(model, device, test_loader, para,Boosting_number):
    model = model.to(device)
    model.eval()
    BER_total = []
    FER_total = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device).to(torch.float)
            output = data
            for i in range(int(Boosting_number)):
                output = model(output, False)
            results = 1 - torch.sigmoid(output * 1000000)
            bool_equal = (results == target).to(torch.float)
            word_target = (conf["data"]["v_size"] + 1)* \
                torch.ones(1, conf["para"]["test_batch_size"])
            word_target = word_target.cuda()
            codeword_equal = (torch.sum(bool_equal, -1).cuda()
                              == word_target).to(torch.float)
            BER = 1 - (torch.sum(bool_equal) /
                       (results.shape[0] * results.shape[1]))
            FER = 1 - torch.sum(codeword_equal) / results.shape[0]
            FER_total.append(FER.cpu().numpy())
            BER_total.append(BER.cpu().numpy())
            print(batch_idx,"BER:",BER,"FER:",FER)
        FER = np.mean(FER_total)
        BER = np.mean(BER_total)

        snr = para["snr"]
        logger.warning(f"num_permu={num_permu},Boosting_number={Boosting_number},SNR={snr},BER={BER:.7f},FER={FER:.7f}")

if __name__ == "__main__":
    para = conf["para"]
    seed = para["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for num_permu in conf["para"]["num_permu_list"].split(","):
        para["num_permu"] = int(num_permu)
        for Boosting_number in conf["para"]["Boosting_number_list"].split(","):
            para["Boosting_number"] = int(Boosting_number)
            for snr in conf["para"]["snr_list"].split(","):
                para["snr"] = int(snr)
                model = cycnet(conf, device).to(device)
                model.load_state_dict(torch.load(conf["para"]["test_model_path"]))
                testset = SignalTestset(conf)
                test_loader = DataLoader(testset, batch_size=conf["para"]["test_batch_size"])
                test(model, device, test_loader, para,Boosting_number)
