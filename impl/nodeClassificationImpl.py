import os
import torch
import numpy as np
from torch.nn import Module
import time
from utils.utils_method import prepare_log_files
import torch.nn.functional as F
import warnings
import sys

import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

warnings.filterwarnings("ignore", category=UserWarning)


class modelImplementation_nodeClassificator(Module):
    def __init__(self, model, criterion, device=None):
        super(modelImplementation_nodeClassificator, self).__init__()
        self.optimizer = None
        self.model = model
        self.criterion = criterion
        self.device = device

    def set_optimizer(self, lr, weight_decay=0):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_test_model(self, input_features, labels, train_mask, test_mask, valid_mask, reg_lambda, n_epochs,
                         test_epoch, test_name="", log_path=".", patience=30):

        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)
        dur = []
        best_val_acc = 0.0
        best_val_loss = 100000.0
        no_improv = 0

        input_features = input_features.to(self.device)
        labels = labels.to(self.device)
        for epoch in range(n_epochs):
            if no_improv > patience:
                break

            self.model.train()
            epoch_start_time = time.time()
            self.optimizer.zero_grad()
            model_out, logits = self.model(input_features)
            loss = self.criterion(logits[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

            cur_epoch_time = time.time() - epoch_start_time
            dur.append(cur_epoch_time)

            train_loss, train_acc = self.evaluate(input_features, labels, train_mask)
            val_loss, val_acc = self.evaluate(input_features, labels, valid_mask)
            test_loss, test_acc = self.evaluate(input_features, labels, test_mask)

            if epoch % 5 == 0:
                '''print("epoch : ", epoch, " -- loss: ", loss.item(), "-- time: ", cur_epoch_time)
                print("training acc : ", train_acc, " -- test_acc : ", test_acc, " -- valid_acc : ", val_acc)
                print("training loss : ", train_loss.item(), " -- test_loss : ", test_loss.item(), " -- valid_loss : ",
                      val_loss.item())
                print("------")'''
                mean_epoch_time = np.mean(np.asarray(dur))
                train_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        train_loss.item(),
                        train_acc,
                        mean_epoch_time,
                        loss))
                train_log.flush()
                test_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        test_loss.item(),
                        test_acc,
                        mean_epoch_time,
                        loss))
                test_log.flush()
                valid_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        val_loss.item(),
                        val_acc,
                        mean_epoch_time,
                        loss))
                valid_log.flush()
            # early stopping
            no_improv += 1
            if val_acc > best_val_acc:
                no_improv = 0
                best_val_loss = val_loss.item()
                best_val_acc = val_acc
                # print("--ES--")
                # print("save_new_best_model, with acc:", val_acc)
                # print("------")
                self.save_model(test_name, log_path)

        # print("Best val acc:", best_val_acc)
        # print("Best val loss:", best_val_loss)
        self.load_model(test_name, log_path)
        # print("-----BEST EPOCH RESULT-----")
        # _, train_acc = self.evaluate(input_features, labels, train_mask)
        # _, val_acc = self.evaluate(input_features, labels, valid_mask)
        # _, test_acc = self.evaluate(input_features, labels, test_mask)
        # print("training acc : ", train_acc, " -- test_acc : ", test_acc, " -- valid_acc : ", val_acc)

    def save_model(self, test_name, log_folder='./'):
        torch.save(self.model.state_dict(), os.path.join(log_folder, test_name + '.pt'))

    def load_model(self, test_name, log_folder):
        self.model.load_state_dict(
            torch.load(os.path.join(log_folder, test_name + '.pt'), map_location=torch.device('cpu')))

    def evaluate(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            model_out, logits = self.model(features)
            set_labels = labels[mask]
            set_logits = logits[mask]
            set_model_out = model_out[mask]
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            loss = self.criterion(set_model_out, set_labels)
            acc = correct.item() * 1.0 / len(set_labels)
            return loss, acc

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def model_eval(self, features, labels, mask, num_classes):
        self.model.eval()
        with torch.no_grad():
            model_out, logits = self.model(features)
            set_labels = labels[mask]
            set_logits = logits[mask]
            set_model_out = model_out[mask]
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            original_acc = correct.item() * 1.0 / len(set_labels)
            y_pred_proba = F.softmax(set_logits, dim=1)

        y_pred = indices.numpy()
        y_true = set_labels.numpy()

        return model_out, y_true, y_pred, y_pred_proba, set_labels

    def robustness_testing(self, X, y, mask, X_different, y_different, mask_different):
        self.model.eval()
        with torch.no_grad():
            model_out, logits = self.model(X)
            set_labels = y[mask]
            set_logits = logits[mask]
            set_model_out = model_out[mask]
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            original_acc = correct.item() * 1.0 / len(set_labels)
            y_pred_proba = F.softmax(set_logits, dim=1)

        print("Original Test Accuracy: ", original_acc)

        X_adversarial = self.fgsm_attack(X, y, epsilon=0.0005)

        with torch.no_grad():
            model_out, logits = self.model(X_adversarial)
            set_labels = y[mask]
            set_logits = logits[mask]
            set_model_out = model_out[mask]
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            adversarial_acc = correct.item() * 1.0 / len(set_labels)

        print("Adversarial Test Accuracy: ", adversarial_acc)

        with torch.no_grad():
            model_out, logits = self.model(X_different)
            set_labels = y_different
            set_logits = logits
            set_model_out = model_out
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            different_dataset_acc = correct.item() * 1.0 / len(set_labels)
        print("Accuracy on Different Dataset:", different_dataset_acc)

    def fgsm_attack(self, features, target, epsilon=0.0001):
        features.requires_grad = True
        model_out, logits = self.model(features)
        loss = self.criterion(model_out, target)
        self.model.zero_grad()
        loss.backward()
        perturbed_features = features + epsilon * features.grad.sign()
        perturbed_features = torch.clamp(perturbed_features, 0, 1)  # Ensure data remains in valid range
        return perturbed_features

    def feat_importance(self, X):
        # Compute SHAP values
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer.shap_values(X)

        # Extract feature importance scores
        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
        feature_names = ['Feature 1', 'Feature 2', 'Feature 3', ...]  # Replace with your actual feature names

        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        sorted_shap_values = mean_abs_shap_values[sorted_indices]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_feature_names, sorted_shap_values)
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title('Feature Importance')
        plt.show()
