import sys
import os
import torch
from dataReader_utils.dataReader import DGLDatasetReader
from model.LGNet import LGNetwork
from conv.LGCN import LGCN
from impl.nodeClassificationImpl import modelImplementation_nodeClassificator
from utils.utils_method import printParOnFile
from torch.utils.tensorboard import SummaryWriter
import torch.cuda
from validation import plots

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)


def run_validation(forceobj=True):
    test_type = 'LGCN-fully-supervised'
    test_folder = os.path.join(parent_dir, "experiments/cora/test_log/LGCN-fully-supervised/data")

    eval_mode = True
    # sys setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_list = range(1)
    n_epochs = 250
    test_epoch = 1
    early_stopping_patience = 100

    # model settings
    lr = 1e-2
    reg_lambda = 0
    lg_level = 1

    # Set Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset>
    dataset_name = 'Cora'

    test_name = 'run_0_LGCN-fully-supervised_data-cora_dropout-0.2_weight-decay-0.0_num_layer-6_hidden-dim-32_backtrack-True_activationReLU_update-SUM_inc-both'

    num_layers = 6
    hidden_dim = 32
    backtrack = True
    update = 'SUM'
    inc_type = 'both'

    test_type_folder = os.path.join("./test_log/" + test_type + "/data")

    training_log_dir = os.path.join(test_type_folder, test_name)

    print(test_name)

    line_graphs, features, labels, num_classes, n_feats, lg_node_list, train_mask, test_mask, valid_mask = \
        DGLDatasetReader(dataset_name=dataset_name, self_loops=False)

    model = LGNetwork(lg_list=line_graphs,
                      in_feats=n_feats,
                      n_classes=num_classes,
                      num_layers=num_layers,
                      h_feats=hidden_dim,
                      update=update,
                      inc_type=inc_type,
                      lg_node_list=lg_node_list,
                      device=device).to(device)
    model_impl = modelImplementation_nodeClassificator(model=model, criterion=criterion, device=device)

    model_impl.load_model(test_name=test_name, log_folder=training_log_dir)

    model_out, y_true, y_pred, y_pred_proba, set_labels = model_impl.model_eval(features=features,
                                                                                labels=labels,
                                                                                mask=test_mask,
                                                                                num_classes=num_classes)
    '''plots.conf_matrix(y_true, y_pred, dataset_name)
    plots.plot_roc_curve(set_labels, y_pred_proba, num_classes, dataset_name)
    plots.plot_auc_roc_micro(set_labels, y_pred_proba, num_classes, dataset_name)
    plots.plot_precision_recall(y_true, y_pred, num_classes)
    plots.plot_macro_micro_metrics(y_true, y_pred, dataset_name)
    plots.vis_embeddings(features, labels, dataset_name)'''
    plots.vis_graphs(line_graphs[0])
    num_par = model_impl.count_parameters()


if __name__ == '__main__':
    # print(torch.cuda.device_count())
    # print(torch.cuda.is_available())
    # print(torch.__version__)
    run_validation()
