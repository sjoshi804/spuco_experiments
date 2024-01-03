guild run waterbirds_erm.py lr='[1e-5, 1e-4, 1e-3, 1e-2]' weight_decay='[1, 1e-1, 1e-2, 1e-3]' pretrained=True batch_size=64 --stage-trials

guild run waterbirds_gb.py lr='[1e-5, 1e-4, 1e-3, 1e-2]' weight_decay='[1, 1e-1, 1e-2, 1e-3]' pretrained=True batch_size=64 --stage-trials

guild run waterbirds_gdro.py lr='[1e-5, 1e-4, 1e-3, 1e-2]' weight_decay='[1, 1e-1, 1e-2, 1e-3]' pretrained=True batch_size=64  --stage-trials

guild run waterbirds_pde.py lr='[1e-5, 1e-4, 1e-3, 1e-2]' weight_decay='[1, 1e-1, 1e-2, 1e-3]' pretrained=True batch_size=64 --stage-trials

guild run waterbirds_spare_train.py lr='[1e-5, 1e-4, 1e-3, 1e-2]' weight_decay='[1, 1e-1, 1e-2, 1e-3]' pretrained=True batch_size=64 --stage-trials