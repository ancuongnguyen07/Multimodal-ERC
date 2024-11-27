# Multi-Modal Speech Recognition Code

This folder contains the code for training and testing a multi-modal emotion classifier using the MELD dataset.

Reference:
- https://github.com/declare-lab/conv-emotion/tree/master/DialogueRNN
- Extracted features file: https://github.com/zerohd4869/MM-DFN/blob/main/data/meld/MELD_features_raw1.pkl

## Files

- `train_MELD.py`: Script to train the deep neural network model.
- `test_MELD.py`: Script to test the trained model on the test dataset.

## Training the Model

To train the model using `train_MELD.py`, follow these steps:

1. Ensure you have the required dependencies installed. You can install them using:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the training script with the desired parameters. For example:
    ```sh
    python train_MELD.py --features-type text_audio_visual --data-path data/MELD_features_raw1.pkl --output-dir models/
    ```
3. Sometimes it is useful to save logs from training steps for further investigation. For example, run the following command to save
training logs to `log_xxx.txt` (change the name of log file as you want):
    ```sh
    python train_MELD.py --features-type text_audio_visual --data-path data/MELD_features_raw1.pkl --output-dir models/ > log_xxx.txt
    ```

### Command-Line Arguments for `train_MELD.py`

- `--no-cuda`: Does not use GPU (default: False).
- `--lr`: Learning rate (default: 0.001).
- `--l2`: L2 regularization weight (default: 0.00001).
- `--rec-dropout`: Recurrent dropout rate (default: 0.1).
- `--dropout`: Dropout rate (default: 0.1).
- `--batch-size`: Batch size (default: 30).
- `--epochs`: Number of epochs (default: 15).
- `--class-weight`: Use class weight (default: False).
- `--active-listener`: Use active listener (default: False).
- `--attention`: Attention type (default: 'general'). Options: ["general", "genreral2", "concat", "dot"]
- `--tensorboard`: Enables tensorboard log (default: False).
- `--features-type`: Feature type, required. Options: ["audio", "text", "visual", "text_audio", "text_visual", "audio_visual", "text_audio_visual"].
- `--data-path`: Path to data (default: `data/MELD_features_raw1.pkl`).
- `--output-dir`: Output directory (optional).

To show help message, run `python train_MELD.py --help`

## Testing the Model

To test the model using `test_MELD.py`, follow these steps:

1. Ensure you have the required dependencies installed. You can install them using:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the testing script with the desired parameters. For example:
    ```sh
    python test_MELD.py --model-path ../models/text_audio_visual_BiDi_Att.pth --features-type text_audio_visual
    ```

### Command-Line Arguments for `test_MELD.py`

- `--model-path`: Path to the saved model file.
- `--features-type`: Feature type, required. Options: ["audio", "text", "visual", "text_audio", "text_visual", "audio_visual", "text_audio_visual"].
- `--batch-size`: Batch size (default: 30).

To show help message, run `python test_MELD.py --help`

## Notes
- The final performance on test dataset reported by `test_MELD.py` and `train_MELD.py` could be different since
`test_MELD` reports the **latest** and `train_MELD` reports the **best** performance after `n` training epochs.
- Ensure that the dataset files are correctly placed in the `data/` directory as specified in the command-line arguments.
- The trained model files should be saved in the `models/` directory.