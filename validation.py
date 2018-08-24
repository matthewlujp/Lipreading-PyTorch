from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class Validator():
    def __init__(self, options, save_dir):

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

        self.save_dir = save_dir

    def epoch(self, model, epoch) -> float:
        count = 0
        validator_function = model.validator_function()
        
        with tqdm(total=len(self.validationdataloader), desc="validation", ascii=False, ncols=150) as t:
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume'])
                labels = sample_batched['label']

                if(self.usecudnn):
                    input = input.cuda()
                    labels = labels.cuda()

                outputs = model(input)

                count += validator_function(outputs, labels)

                t.update()

        accuracy = count / len(self.validationdataset)
        with open(os.path.join(self.save_dir, "accuracy.txt"), "a") as outputfile:
            outputfile.write("\nepoch {} --- correct count: {}, total count: {} accuracy: {}" .format(epoch, count, len(self.validationdataset), accuracy))

        return accuracy
