# Commands

guild run waterbirds_eiil.py infer_lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-2,1e-1,1]' infer_num_epochs='[1,2]' --stage-trials

guild run waterbirds_jtt.py infer_lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-2,1e-1,1]' infer_num_epochs='[40,50,60]' --stage-trials

guild run waterbirds_spare.py infer_lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-2,1e-1,1]' infer_num_epochs='[1,2]' --stage-trials