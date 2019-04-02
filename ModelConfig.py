class SR_Config():
    block_num=8
    growth_rate=16
    dense_layer_num=8
    bottle_neck_channel=256
    bottle_neck_layer=2
    input_channel=growth_rate*dense_layer_num
    dense_features=growth_rate*dense_layer_num*block_num
class Dense():
    growth_rate=12
    layer_num=15
    dense_channel=growth_rate*(layer_num+1)
    neck_channel=128
class Config():
    patch_size=64
    train_ms_path = '/media/zhou/文档/GF2/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826-MSS2.tiff'
    train_pan_path = '/media/zhou/文档/GF2/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826-PAN2.tiff'
    val_ms_path = '/media/zhou/文档/GF2/GF2_PMS1_E102.6_N24.4_20170119_L1A0002132394/GF2_PMS1_E102.6_N24.4_20170119_L1A0002132394-MSS1.tiff'
    val_pan_path = '/media/zhou/文档/GF2/GF2_PMS1_E102.6_N24.4_20170119_L1A0002132394/GF2_PMS1_E102.6_N24.4_20170119_L1A0002132394-PAN1.tiff'
    batch_size=128
    lr=0.0001
    weight_decay=0.0001