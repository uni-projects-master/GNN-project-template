import sys
import os
import torch
from dataReader_utils.dataReader import DGLDatasetReader
from model.LGNet import LGNetwork
from conv.LSGN import LSGN
from impl.nodeClassificationImpl import modelImplementation_nodeClassificator
from utils.utils_method import printParOnFile
from torch.utils.tensorboard import SummaryWriter
import torch.cuda
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


def run_validation():
    test_type = 'LineSGN'
    # sys setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_list = range(5)
    n_epochs = 150
    test_epoch = 1
    early_stopping_patience = 80

    # test hyper par
    lr_list = [1e-2, 1e-3]
    lambda_list = [1e-3]
    weight_decay_list = [0.0, 5e-3, 5e-4, 5e-6]

    # model settings
    dropout_list = [0.0, 0.2, 0.5]
    num_layers_list = [2, 4, 5]
    hidden_dim_list = [16, 32]
    activation_list = ['ReLU', 'LeakyReLU']
    inc_type_list = ['in', 'out', 'both']

    # line graph hyper par
    lg_level_list = [1, 2, 3]
    backtrack_list = [True, False]

    # Set Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset>
    dataset_name = 'cora'
    problem_type = "semi-supervised"
    self_loops = True
    for lg_level in lg_level_list:
        for backtrack in backtrack_list:
            for lr in lr_list:
                for reg_lambda in lambda_list:
                    for num_layers in num_layers_list:
                        for hidden_dim in hidden_dim_list:
                            for activation in activation_list:
                                for dropout in dropout_list:
                                    for inc_type in inc_type_list:
                                        for weight_decay in weight_decay_list:
                                            for run in run_list:
                                                test_name = "run_" + str(run) + '_' + test_type
                                                # Env
                                                test_name = test_name + \
                                                            "_data-" + dataset_name + \
                                                            "_lr-" + str(lr) + \
                                                            "_dropout-" + str(dropout) + \
                                                            "_lambda-" + str(reg_lambda) + \
                                                            "_weight-decay-" + str(weight_decay) + \
                                                            "_num_layer-" + str(num_layers) + \
                                                            "_hidden-dim-" + str(hidden_dim) + \
                                                            "_activation-" + str(activation) + \
                                                            "_lg-level-" + str(lg_level) + \
                                                            "_inc-type-" + str(inc_type) + \
                                                            "_backtrack-" + str(backtrack)

                                                writer = SummaryWriter("./test_log/" + str(test_type) + '/tensorboard/{}'.format(test_name))
                                                test_type_folder = os.path.join("./test_log/" + test_type + "/data")
                                                if not os.path.exists(test_type_folder):
                                                    os.makedirs(test_type_folder)
                                                training_log_dir = os.path.join(test_type_folder, test_name)
                                                print(test_name)
                                                if not os.path.exists(training_log_dir):
                                                    os.makedirs(training_log_dir)

                                                    printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                                   par_list={"dataset_name": dataset_name,
                                                                             "learning_rate": lr,
                                                                             "dropout": dropout,
                                                                             "lambda": reg_lambda,
                                                                             "weight_decay": weight_decay,
                                                                             "num_layers": num_layers,
                                                                             "hidden_dim": hidden_dim,
                                                                             "activation": activation,
                                                                             "lg_level": lg_level,
                                                                             "inc_type": inc_type,
                                                                             "backtrack": backtrack,
                                                                             "test_epoch": test_epoch,
                                                                             "self_loops": self_loops})

                                                    line_graphs, features, labels, n_classes, n_feats, lg_node_list, train_mask, test_mask, valid_mask = \
                                                        DGLDatasetReader(dataset_name=dataset_name, lg_level=lg_level,
                                                                         backtrack=backtrack,
                                                                         problem_type=problem_type,
                                                                         self_loops=self_loops,
                                                                         device=device)

                                                    model = LGNetwork(lg_list=line_graphs,
                                                                      in_feats=n_feats,
                                                                      n_classes=n_classes,
                                                                      dropout=dropout,
                                                                      num_layers=num_layers,
                                                                      h_feats=hidden_dim,
                                                                      activation=activation,
                                                                      inc_type=inc_type,
                                                                      convLayer=LSGN,
                                                                      lg_node_list=lg_node_list,
                                                                      device=device).to(device)

                                                    model_impl = modelImplementation_nodeClassificator(model=model,
                                                                                                       criterion=criterion,
                                                                                                       device=device)
                                                    model_impl.set_optimizer(lr=lr, weight_decay=weight_decay)

                                                    model_impl.train_test_model(input_features=features,
                                                                                labels=labels,
                                                                                train_mask=train_mask,
                                                                                test_mask=test_mask,
                                                                                valid_mask=valid_mask,
                                                                                n_epochs=n_epochs,
                                                                                reg_lambda=reg_lambda,
                                                                                test_epoch=test_epoch,
                                                                                test_name=test_name,
                                                                                log_path=training_log_dir,
                                                                                writer=writer,
                                                                                patience=early_stopping_patience)
                                            else:
                                                print("test has been already execute")


if __name__ == '__main__':

    #print(torch.cuda.device_count())
    #print(torch.cuda.is_available())
    #print(torch.__version__)
    run_validation()
