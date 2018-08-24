import os
import sys

class CSVSaver:
    def __init__(self, filename: str, *args):
        if os.path.exists(filename):
            raise Exception("specified file already exists")
        if len(args) == 0:
            raise Exception("no items to save in csv")

        self.__items = args
        self.__filename = filename
        self._init_file()

    def _init_file(self):
        with open(self.__filename, 'w') as f:
            f.write("epoch," + ','.join(self.__items) + '\n')

    def add(self, epoch: int, **kwargs):
        if not os.path.exists(self.__filename):
            self._init_file()

        for k in kwargs.keys():
            if k not in self.__items:
                print("WARN: item {} specified in add method is not registered as a csv item".format(k), file=sys.stderr)

        item_vals = [str(epoch)]
        for item  in self.__items:
            v = kwargs.get(item)
            v = str(v) if v else ''
            item_vals.append(v)
        with open(self.__filename, 'a') as f:
            f.write(",".join(item_vals) + '\n')

    
        

