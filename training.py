from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os
import math
from tqdm import trange

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time, totalitems):
    os.system('clear')

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

def estimate_remaining_time(i, time, totalitems):
    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    return timedelta_string(estTime)

class Trainer():
    def __init__(self, options, model_save_dir):
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

        self.model_save_dir = model_save_dir

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    def epoch(self, model, epoch):
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
        print("Starting training...")

        correct_count = 0
        summed_loss = 0
        total_samples = 0
        
        for i_batch, sample_batched in enumerate(trange(self.trainingdataloader), nclos=80):
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])

            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

            outputs = model(input)
            loss = criterion(outputs, labels.squeeze(1))

            loss.backward()
            optimizer.step()

            correct_count += validator_function(outputs, labels)
            summed_loss += loss.data * len(sample_batched)
            total_samples += len(sample_batched)

            t.set_description("Epoch {:02}".format(epoch))
            estimated_remaining_time = estimate_remaining_time(total_samples, datetime.now() - startTime, len(self.trainingdataset))
            t.set_postfix(loss=summed_loss/total_samples, accuracy=correct_count/total_samples, remaining_time=estimated_remaining_time)

        print("Epoch completed, avg loss {}, avg acc {}, saving state...".format(summed_loss/total_samples, correct_count/total_samples))
        torch.save(model.state_dict(), os.path.join(self.model_save_dir, "epoch{}.pt".format(epoch)))
