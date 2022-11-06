import os
import shutil
import csv
import random

class mnist_anatomy_config:

    input_rows = 48
    input_cols = 48
    input_deps = 3

    arch = 'Linknet'
    backbone = 'inceptionresnetv2'
    exp_name = arch + '-' + backbone
    patience = 5
    verbose = 1
    lr = 0.1
    nb_epoch = 10000
    batch_size = 128
    nb_class = 10
    use_multiprocessing = False
    workers = 1
    max_queue_size = 1

    def __init__(self, args):
        if args.arch is not None:
            self.arch = args.arch
        if args.backbone is not None:
            self.backbone = args.backbone
        self.exp_name = self.arch + '-' + self.backbone

        self.input_rows = args.input_rows
        self.input_cols = args.input_cols
        self.input_deps = args.input_deps
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.patience = args.patience
        self.init = args.init
        
        self.idx_path = 'idx'
        self.model_path = os.path.join("models", "run_"+str(args.run))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.predict_path = os.path.join(self.model_path, "predicts")
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)
        self.logs_path = os.path.join(self.model_path, "logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_ids' in a:
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
