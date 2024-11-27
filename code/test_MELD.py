import torch
import numpy as np

from sklearn.metrics import confusion_matrix, \
    classification_report

from enums import FeatureType
from train_MELD import DATA_PATH, SEED, train_or_eval_model, get_MELD_loaders, \
    get_default_parser, load_baseline_hyperparameters
from model import BiModel, MaskedNLLLoss
# Import below modules to make torch.load could find the class definition
from model import DialogueRNNCell, DialogueRNN, MatchingAttention


MODEL_PATH = 'models/'

if __name__ == '__main__':
    np.random.seed(SEED)

    parser = get_default_parser()
    parser.add_argument('--model-path', type=str,
                        default=MODEL_PATH, help='Path to model file')

    args = parser.parse_args()
    args = load_baseline_hyperparameters(args)
    assert args.model_path is not None or args.model_name is not None, 'Please specify model path or model name'

    feature_type = FeatureType.from_str(args.features_type)

    # load model
    if args.model_path is not None:
        model_file = args.model_path
    else:
        model_file = f'{MODEL_PATH}{args.model_name}.pth'
    model: BiModel = torch.load(model_file)
    assert isinstance(
        model, BiModel), f'Model type is not BiModel, type(model)={type(model)}'

    N_CLASSES = 7
    loss_weights = torch.FloatTensor([1.0]*N_CLASSES)

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    _, _, test_loader = get_MELD_loaders(DATA_PATH, N_CLASSES,
                                         feature_types=feature_type,
                                         valid=0.0,
                                         batch_size=args.batch_size,
                                         num_workers=0)

    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = train_or_eval_model(
        model, loss_function, test_loader, feature_type, cuda=False
    )

    print(classification_report(test_label, test_pred,
          sample_weight=test_mask, digits=4, zero_division=0.0))
    print('Confusion matrix')
    print(confusion_matrix(test_label, test_pred, sample_weight=test_mask))
