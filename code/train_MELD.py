import argparse
import time
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report

from model import BiModel, MaskedNLLLoss
from dataloader import MELDDataset
from enums import FeatureType

# set random seed so that we can reproduce the results
SEED = 1234
np.random.seed(SEED)


def get_train_valid_sampler(trainset, valid=0.1):
    """
    Creates training and validation samplers for a given dataset.
    Args:
        trainset (Dataset): The dataset to be split into training and validation sets.
        valid (float, optional): The proportion of the dataset to include in the validation split. 
                                 Should be between 0 and 1. Default is 0.1 (10%).
    Returns:
        tuple: A tuple containing two SubsetRandomSampler objects, one for the training set and one for the validation set.
    """

    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(path, n_classes, feature_types, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    """
    Creates and returns DataLoader objects for training, validation, and testing datasets for the MELD dataset.
    Args:
        path (str): Path to the dataset.
        n_classes (int): Number of classes in the dataset.
        feature_types (list): List of feature types to be used.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        valid (float, optional): Proportion of the training set to use for validation. Default is 0.1.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Default is False.
    Returns:
        tuple: A tuple containing three DataLoader objects for the training, validation, and testing datasets.
    """

    trainset = MELDDataset(path=path, n_classes=n_classes,
                           feature_types=feature_types)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(path=path, n_classes=n_classes,
                          feature_types=feature_types, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, feature_type, optimizer=None, train=False, cuda=False):
    """
    Train or evaluate a model based on the provided parameters.
    Args:
        model (torch.nn.Module): The model to train or evaluate.
        loss_function (callable): The loss function to use.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset.
        feature_type (FeatureType): The type of features used (e.g., AUDIO, TEXT, VISUAL, etc.).
        optimizer (torch.optim.Optimizer, optional): The optimizer to use for training. Required if train is True.
        train (bool, optional): If True, the model is trained. If False, the model is evaluated. Default is False.
        cuda (bool, optional): If True, use CUDA for computation. Default is False.
    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss.
            - avg_accuracy (float): The average accuracy.
            - labels (list): The true labels.
            - preds (list): The predicted labels.
            - masks (list): The masks used during training/evaluation.
            - avg_fscore (float): The average F1 score.
            - alphas (list): The attention weights (alphas, alphas_f, alphas_b, vids).
            - class_report (str): The classification report.
    """

    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        tensor_data = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if feature_type == FeatureType.AUDIO:
            acouf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                acouf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == FeatureType.TEXT:
            textf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                textf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == FeatureType.VISUAL:
            visuf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                visuf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == FeatureType.TEXT_VISUAL:
            textf, visuf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                torch.cat([textf, visuf], dim=-1), qmask, umask)
        elif feature_type == FeatureType.TEXT_AUDIO:
            textf, acouf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                torch.cat([textf, acouf], dim=-1), qmask, umask)
        elif feature_type == FeatureType.AUDIO_VISUAL:
            acouf, visuf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                torch.cat([acouf, visuf], dim=-1), qmask, umask)
        elif feature_type == FeatureType.TEXT_AUDIO_VISUAL:
            textf, acouf, visuf, qmask, umask, label = tensor_data
            log_prob, alpha, alpha_f, alpha_b = model(
                # seq_len, batch, n_classes
                torch.cat([textf, acouf, visuf], dim=-1), qmask, umask)
        else:
            raise ValueError("Feature type not understood")

        lp_ = log_prob.transpose(0, 1).contiguous(
        ).view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
#             if args.tensorboard:
#                 for param in model.named_parameters():
#                     writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if len(preds) > 0:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(
        labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(
        f1_score(labels, preds, sample_weight=masks, average='weighted', zero_division=0.0)*100, 2)
    class_report = classification_report(
        labels, preds, sample_weight=masks, digits=4, zero_division=0.0)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids], class_report


# choose between 'sentiment' or 'emotion'
# classification_type = 'emotion'
# feature_type = 'multimodal'

DATA_PATH = 'data/MELD_features_raw1.pkl'
# batch_size = 30
# n_classes = 3
# n_epochs = 100
# active_listener = False
# attention = 'general'
# class_weight = False
# dropout = 0.1
# rec_dropout = 0.1
# l2 = 0.00001
# lr = 0.0005

D_TEXT = 600
D_AUDIO = 300
D_VISUAL = 342
N_CLASSES = 7


def get_default_parser() -> argparse.ArgumentParser:
    """
    Creates and returns the argument parser with default arguments for training the MELD model.
    Returns:
        argparse.ArgumentParser: The argument parser with the following arguments:
            --no-cuda (store_true): Does not use GPU (default: False).
            --lr (float): Learning rate (default: 0.001).
            --l2 (float): L2 regularization weight (default: 0.00001).
            --rec-dropout (float): Recurrent dropout rate (default: 0.1).
            --dropout (float): Dropout rate (default: 0.1).
            --batch-size (int): Batch size (default: 30).
            --epochs (int): Number of epochs (default: 15).
            --class-weight (store_true): Use class weight (default: False).
            --active-listener (store_true): Use active listener (default: False).
            --attention (str): Attention type (default: 'general'). Options: ["general", "genreral2", "concat", "dot"]
            --tensorboard (store_true): Enables tensorboard log (default: False).
            --features-type (str): Feature type, required. Options: ["audio", "text", "visual", "text_audio", "text_visual", "audio_visual", "text_audio_visual"].
            --data-path (str): Path to data (default: DATA_PATH).
            --output-dir (str): Output directory (optional).
    """

    # Parse command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=15, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general',
                        help='Attention type. Options: ["general", "genreral2", "concat", "dot"]')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--features-type', type=str,
                        help='Feature type: ["audio", "text", "visual", "text_audio", "text_visual", "audio_visual", "text_audio_visual"]', required=True)
    parser.add_argument('--data-path', type=str,
                        help='path to data', default=DATA_PATH)
    parser.add_argument('--output-dir', type=str,
                        help='output directory', required=False)
    return parser


def load_baseline_hyperparameters(args) -> argparse.Namespace:
    """
    Load baseline hyperparameters for training.
    This function sets the learning rate and the number of epochs
    to predefined baseline values.
    Parameters:
    args (argparse.Namespace): The arguments namespace to update with baseline hyperparameters.
    Returns:
    argparse.Namespace: The updated arguments namespace with baseline hyperparameters.
    """

    args.lr = 0.0005
    args.epochs = 100
    return args


def main():
    """
    Main function to train and evaluate a model on the MELD dataset.
    This function performs the following steps:
    1. Parses command-line arguments and loads baseline hyperparameters.
    2. Sets up feature types and dimensions based on the input arguments.
    3. Initializes the model, loss function, and optimizer.
    4. Loads the MELD dataset and prepares data loaders for training, validation, and testing.
    5. Trains the model for a specified number of epochs, evaluating on validation and test sets.
    6. Tracks the best model performance based on F-score and saves the best model.
    7. Prints the final test performance, including classification report and confusion matrix.
    8. Saves the trained model to the specified output directory.
    Args:
        None
    Returns:
        None
    """

    start = time.time()

    # tensorboard = False
    # if tensorboard:
    #     from tensorboardX import SummaryWriter
    #     writer = SummaryWriter()

    args = get_default_parser().parse_args()
    # we need to use hyperparameters that are used in the baseline model
    # to make the comparison fair
    args = load_baseline_hyperparameters(args)
    assert args.lr == 0.0005
    assert args.epochs == 100
    feature_type = FeatureType.from_str(args.features_type)

    cuda = torch.cuda.is_available()
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # set up dimensions
    if feature_type == FeatureType.TEXT:
        print("Running on the text features........")
        D_m = D_TEXT
    elif feature_type == FeatureType.AUDIO:
        print("Running on the audio features........")
        D_m = D_AUDIO
    elif feature_type == FeatureType.VISUAL:
        print("Running on the visual features........")
        D_m = D_VISUAL
    elif feature_type == FeatureType.TEXT_VISUAL:
        print("Running on the text and visual features........")
        D_m = D_TEXT + D_VISUAL
    elif feature_type == FeatureType.TEXT_AUDIO:
        print("Running on the text and audio features........")
        D_m = D_TEXT + D_AUDIO
    elif feature_type == FeatureType.AUDIO_VISUAL:
        print("Running on the audio and visual features........")
        D_m = D_AUDIO + D_VISUAL
    elif feature_type == FeatureType.TEXT_AUDIO_VISUAL:
        print("Running on the text, audio and visual features........")
        D_m = D_TEXT + D_AUDIO + D_VISUAL
    else:
        raise ValueError("Feature type not understood")

    # set up dimensions of different network components
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100  # concat attention

    # if classification_type.strip().lower() == 'emotion':
    #     n_classes = 7
    #     loss_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # elif classification_type.strip().lower() == 'sentiment':
    #     n_classes = 3
    #     loss_weights = torch.FloatTensor([1.0, 1.0, 1.0])

    loss_weights = torch.FloatTensor([1.0]*N_CLASSES)

    model = BiModel(D_m, D_g, D_p, D_e, D_h,
                    n_classes=N_CLASSES,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    D_a=D_a,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout)

    if cuda:
        model.cuda()
    if args.class_weight:
        loss_function = MaskedNLLLoss(
            loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_MELD_loaders(args.data_path, N_CLASSES,
                                                               feature_types=feature_type,
                                                               valid=0.0,
                                                               batch_size=args.batch_size,
                                                               num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask, best_epoch = None, None, None, None, None, None

    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _, _ = train_or_eval_model(
            model, loss_function, train_loader, feature_type, optimizer, True, cuda=cuda
        )
        valid_loss, valid_acc, _, _, _, val_fscore, _ = train_or_eval_model(
            model, loss_function, valid_loader, feature_type, cuda=cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = train_or_eval_model(
            model, loss_function, test_loader, feature_type, cuda=cuda
        )

        if best_fscore is None or best_fscore < test_fscore:
            best_fscore, best_loss, best_label, best_pred, best_mask, best_attn, best_epoch = test_fscore, test_loss, test_label, test_pred, test_mask, attentions, e

        # if tensorboard:
        #     writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
        #     writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.
              format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,
                     test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        print(test_class_report)

    # end training
    # if tensorboard:
    #     writer.close()

    end = time.time()

    print('Test performance..')
    print('Best Fscore {} accuracy {} at epoch {}'.format(best_fscore,
                                                          round(accuracy_score(best_label, best_pred, sample_weight=best_mask)*100, 2), best_epoch))
    print(classification_report(best_label, best_pred,
                                sample_weight=best_mask, digits=4, zero_division=0.0))
    print('Confusion matrix')
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    # get the timestamp and elapsed time
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    elapsed_time = (end - start) / 3600
    hours = int(elapsed_time)
    minutes = (elapsed_time - hours) * 60
    print()
    print(f'Elapsed time: {hours} hour(s) {minutes:.2f} minute(s)')

    # save the model
    model_file_name = f'{args.features_type}_BiDi_Att_{timestamp}.pth'
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = './'
    try:
        torch.save(model, f'''{output_dir}/{model_file_name}''')
    except FileNotFoundError:
        print(f'The given output directory {output_dir} does not exist')
        print('Model is saved to the current directory instead')
        output_dir = './'
        torch.save(model, f'''{output_dir}/{model_file_name}''')


if __name__ == '__main__':
    main()
