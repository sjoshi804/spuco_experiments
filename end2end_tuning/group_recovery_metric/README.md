guild run waterbirds_spare.py pretrained=True infer_lr='[1e-3, 1e-4, 1e-5]' infer_weight_decay='[1e-2,1e-1,1]' infer_num_epochs='[1,2]' batch_size=64 val-size-pct='[5, 15]' --stage-trials

guild run waterbirds_eiil.py pretrained=True infer_lr='[1e-3, 1e-4, 1e-5]' infer_weight_decay='[1e-2,1e-1,1]' infer_num_epochs='[1,2]' batch_size=64 val-size-pct='[5, 15]' --stage-trials

guild run waterbirds_jtt.py pretrained=True infer_lr='[1e-3, 1e-4, 1e-5]' infer_weight_decay='[1e-2,1e-1,1]' infer_num_epochs='[40,50,60]' batch_size=64 val-size-pct='[5, 15]' --stage-trials