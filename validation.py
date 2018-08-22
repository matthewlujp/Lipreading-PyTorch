from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os
from tqdm import trange

class Validator():
    def __init__(self, options):

        self.validationdataset = LipreadingDataset(options["validation"]["dataset"], "val", False)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]

        self.gpuid = options["general"]["gpuid"]

    def epoch(self, model, epoch):
        print("Starting validation...")
        count = 0
        validator_function = model.validator_function()

        for i_batch, sample_batched in enumerate(trange(self.validationdataloader, ncols=80)):
            input = Variable(sample_batched['temporalvolume'])
            labels = sample_batched['label']

            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

            outputs = model(input)

            count += validator_function(outputs, labels)

            print(count)


        accuracy = count / len(self.validationdataset)
        with open("accuracy.txt", "a") as outputfile:
            outputfile.write("\nepoch {} --- correct count: {}, total count: {} accuracy: {}" .format(epoch, count, len(self.validationdataset), accuracy ))
