"""Gerçek NAS'ta seçilen statik alt ağın tam COCO ince ayarı (Karar 6) ve değerlendirmesi.

Seçenekler (`exp.merge`):
    nas_arch    alt ağ kimliği, supernet.config_id biçimi: r416_d33_k<30 hane>
    nas_init    supernet_search.py extract ile kaydedilen alt ağ state_dict'i (.pth, {"model": ...});
                iskelete strict yüklenir. Boşsa iskelet rastgele kalır (yalnız test için).

Tarif nano_macro_long ile aynıdır (tam train2017, 12 epoch, lr 0,001/64, warmup 1, no_aug son 1 epoch):
nas_r2 kontrolü (dokunulmamış Nano, 416'da 25,7) aynı tarifle eğitildiği için doğrudan karşılaştırılır.
Çözünürlük eğitimde değil değerlendirmede (`eval.py --tsize <res>`) uygulanır.
"""

import torch

from nano_macro_long import Exp as _Long
from supernet import DEPTH_STAGES, dw_layer_names, parse_config_id, subnet_skeleton


class Exp(_Long):
    def __init__(self):
        super().__init__()
        self.nas_arch = ""
        self.nas_init = ""
        self.exp_name = "nano_subnet"

    @property
    def arch_id(self):
        return self.nas_arch

    def get_model(self, sublinear=False):
        if "model" in self.__dict__:
            return self.model
        assert self.nas_arch, "nas_arch gerekli"
        cfg = parse_config_id(self.nas_arch)
        from yolox.exp.default.yolox_nano import Exp as Nano
        model = Nano.get_model(self)
        names = dw_layer_names(model)
        assert len(names) == len(cfg["k"]), (len(names), len(cfg["k"]))
        subnet_skeleton(model, dict(zip(names, cfg["k"])), cfg["depth"])
        if self.nas_init:
            state = torch.load(self.nas_init, map_location="cpu")["model"]
            model.load_state_dict(state, strict=True)
            self.nas_summary["init_loaded"] = True
        self.nas_summary.update({"arch": self.nas_arch,
                                 "depth": {s: cfg["depth"][s] for s in DEPTH_STAGES}})
        self.test_size = (cfg["res"], cfg["res"])
        self.model = model
        return model
