from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.cuda as cuda
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os
import math
from tqdm import tqdm
from utils import *


class Trainer():
    def __init__(self, options):
        self.trainingdataset = LipreadingDataset(options["training"]["dataset"], "train")
        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]

        self.learningrate = options["training"]["learningrate"]

        self.modelType = options["training"]["learningrate"]

        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    def epoch(self, model, epoch) -> float:
        #set up the loss function.
        criterion = model.loss()
        optimizer = optim.SGD(
                        model.parameters(),
                        lr = self.learningRate(epoch),
                        momentum = self.learningrate,
                        weight_decay = self.weightdecay)
        validator_function = model.validator_function()

        #transfer the model to the GPU.
        if(self.usecudnn):
            criterion = criterion.cuda()

        startTime = datetime.now()

        correct_count = 0
        summed_loss = 0
        total_samples = 0
        
        with tqdm(total=len(self.trainingdataloader), desc="Epoch {:02}".format(epoch), ascii=False, ncols=150) as t:
            for i_batch, sample_batched in enumerate(self.trainingdataloader):
                optimizer.zero_grad()
                input = Variable(sample_batched['temporalvolume'])
                labels = Variable(sample_batched['label'])

                if cuda.is_available() and self.usecudnn:
                    input = input.cuda()
                    labels = labels.cuda()

                outputs = model(input)
                loss = criterion(outputs, labels.squeeze(1))

                loss.backward()
                optimizer.step()

                correct_count += validator_function(outputs, labels)
                summed_loss += loss.data * len(sample_batched)
                total_samples += len(sample_batched['label'])

                estimated_remaining_time = estimate_remaining_time(i_batch, datetime.now() - startTime, len(self.trainingdataloader))
                t.set_postfix(loss=float(summed_loss.data)/total_samples, acc=correct_count/total_samples, rest_time=estimated_remaining_time)
                t.update()

        print("Epoch completed, avg loss {}, avg acc {}, saving state...".format(float(summed_loss.data)/total_samples, correct_count/total_samples))

        return float(summed_loss.data)/total_samples