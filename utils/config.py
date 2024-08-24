from easydict import EasyDict

cfg = EasyDict()
cfg.epoch = 200
cfg.learning_rate = 1e-4
cfg.weight_decay = 1e-5
cfg.patience = 50

cfg.unetr = EasyDict()
cfg.unetr.img_shape = (96, 96, 96)
cfg.unetr.input_dim = 4
cfg.unetr.output_dim = 3
cfg.unetr.patch_size = 16
cfg.unetr.embed_dim = 768
cfg.unetr.num_layers = 12
cfg.unetr.num_heads = 12
cfg.unetr.mlp_dim = 2048
cfg.unetr.extract_layers = [3, 6, 9, 12]
