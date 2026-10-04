from yolox.exp.default.yolox_nano import Exp as _Base


class Exp(_Base):
    def __init__(self):
        super().__init__()
        self.data_dir = "/content/datasets/COCO"
        self.train_ann = "instances_train2017_10k.json"
        self.val_ann = "instances_val2017.json"
        self.max_epoch = 1
        self.no_aug_epochs = -1  # 0 bile mosaic'i kapatır (bkz. hücre başı)
        self.eval_interval = 10_000
        self.data_num_workers = 12
        self.print_interval = 10
        self.save_history_ckpt = False
        self.exp_name = "pilot_nano_10k"
