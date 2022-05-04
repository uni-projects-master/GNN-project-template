import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
from dataReader_utils.dataReader import ogbn_arxiv_reader
from model.network_std import GCNetwork
from conv.hEGConv import hEGConv
from impl.nodeClassificationImpl import modelImplementation_nodeClassificator_ogbn as modelImplementation_nodeClassificator
from utils.utils_method import printParOnFile

if __name__ == '__main__':

    test_type='hEGC'

    # sis setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_list = range(5)
    n_epochs=500
    test_epoch=1
    early_stopping_patience=100



    # test hyper par
    dropout_list = [0.0, 0.2, 0.5, 0.7]
    lr_list = [0.2, 0.1, 0.05, 0.001]
    weight_decay_list = [5e-3, 5e-4, 5e-6,0]
    k_list = [2, 5, 10, 20, 50]
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset
    dataset_name = 'ogbn-arxiv'
    self_loops = True

    g, features, labels, n_classes, train_mask, test_mask, valid_mask, evaluator = ogbn_arxiv_reader(device,"~/Dataset/")

    for lr in lr_list:
        for dropout in dropout_list:
            for weight_decay in weight_decay_list:
                for k in k_list:
                    for run in run_list:
                        test_name = "run_" + str(run) +'_'+ test_type
                        #Env
                        test_name = test_name +\
                                    "_data-" + dataset_name +\
                                    "_lr-" + str(lr) +\
                                    "_dropout-" + str(dropout) +\
                                    "_weight-decay-" + str(weight_decay) +\
                                    "_k-" + str(k)

                        test_type_folder=os.path.join("./test_log/",test_type)
                        if not os.path.exists(test_type_folder):
                            os.makedirs(test_type_folder)
                        training_log_dir = os.path.join(test_type_folder, test_name)
                        print(test_name)
                        if not os.path.exists(training_log_dir):
                            os.makedirs(training_log_dir)

                            printParOnFile(test_name=test_name, log_dir=training_log_dir, par_list={"dataset_name": dataset_name,
                                                                                                    "learning_rate": lr,
                                                                                                    "dropout": dropout,
                                                                                                    "weight_decay": weight_decay,
                                                                                                    "k": k,
                                                                                                    "test_epoch": test_epoch,
                                                                                                    "self_loops": self_loops})



                            model = GCNetwork(g=g,
                                              in_feats=features.shape[1],
                                              n_classes=n_classes,
                                              dropout=dropout,
                                              k=k,
                                              convLayer=hEGConv,
                                              device=device).to(device)

                            model_impl = modelImplementation_nodeClassificator(model=model,
                                                                               criterion=criterion,
                                                                               device=device)
                            model_impl.set_optimizer(lr=lr,
                                                     weight_decay=weight_decay)

                            model_impl.train_test_model(input_features=features,
                                                        labels=labels,
                                                        train_mask=train_mask,
                                                        test_mask=test_mask,
                                                        valid_mask=valid_mask,
                                                        n_epochs=n_epochs,
                                                        test_epoch=test_epoch,
                                                        test_name=test_name,
                                                        log_path=training_log_dir,
                                                        patience=early_stopping_patience,
                                                        evaluator=evaluator)
                            if str(device) == 'cuda':
                                del model
                                del model_impl
                                torch.cuda.empty_cache()
                        else:
                            print("test has been already execute")
