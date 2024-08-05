import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.nlp import optimization
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef
sys.path.append("../")
from neurallog.models import NeuralLog
from neurallog import data_loader
from neurallog.utils import classification_report
from sklearn.metrics import confusion_matrix, auc
import matplotlib.pyplot as plt
from sklearn import metrics
from argparse import ArgumentParser

embed_dim = 256  # Embedding size for each token
max_len = 75


class BatchGenerator(Sequence):
    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(self.batch_size)
        dummy = np.zeros(shape=(embed_dim,))
        x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        X = np.zeros((len(x), max_len, embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            if len(x) > max_len:
                x = x[-max_len:]
            x = np.pad(np.array(x), pad_width=((max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            X[item_count] = np.reshape(x, [max_len, embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
        return X[:], Y[:, 0]


def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                    epoch_num, model_name=None):
    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='lamb')

    loss_object = SparseCategoricalCrossentropy()

    model = NeuralLog(embed_dim, ff_dim=2048, max_len=75, num_heads=12, dropout=0.1)

    # model.load_weights("hdfs_transformer.hdf5")

    model.compile(loss=loss_object, metrics=['accuracy'],
                  optimizer=optimizer)

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        shuffle=True
                        )
    return model


def train(X, Y, epoch_num, batch_size, model_file=None):
    X, Y = shuffle(X, Y)
    n_samples = len(X)

    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    training_generator, num_train_samples = BatchGenerator(train_x, train_y, batch_size), len(train_x)
    validate_generator, num_val_samples = BatchGenerator(val_x, val_y, batch_size), len(val_x)

    print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples,
                                                                                       num_val_samples))

    model = train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                            epoch_num, model_name=model_file)

    return model


def test_model(model, x, y, batch_size):
    x, y = shuffle(x, y)
    x, y = x[: len(x) // batch_size * batch_size], y[: len(y) // batch_size * batch_size]
    test_loader = BatchGenerator(x, y, batch_size)
    prediction = model.predict_generator(test_loader, steps=(len(x) // batch_size), workers=16, max_queue_size=32,
                                         verbose=1)
    probs = prediction[:, 1]
    prediction = np.argmax(prediction, axis=1)
    y = y[:len(prediction)]
    report = classification_report(np.array(y), prediction)
    print(report)

    conf_matrix = confusion_matrix(y_true=np.array(y), y_pred=prediction)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    tn, fp, fn, tp = conf_matrix.ravel()
    print("True Positive (TP): ", tp)
    print("True Negative (TN): ", tn)
    print("False Positive (FP): ", fp)
    print("False Negative (FN): ", fn)

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    spec = tn/(fp+tn)
    f1 = 2*recall*precision/(recall+precision)
    mcc = matthews_corrcoef(y, prediction.tolist())
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(precision, recall, f1, spec))

    FPR, TPR, thresholds = metrics.roc_curve(np.array(y), probs, pos_label=1)
    AUC = auc(FPR, TPR)

    print('AUC: {}'.format(AUC))
    print('MCC:{}'.format(mcc))

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--log_file", default="../../dataset/bgl/BGL.log", help="path to raw logs")
    parser.add_argument('--window_size', default=20, type=int, help='number of logs in a sequence')
    parser.add_argument('--step_size', default=20, type=int, help='number of logs passed between the start '
                                                                   'indexes of two sequences')
    parser.add_argument('--sampling_method', default='SMOTE', type=str, help='selected sampling method')
    parser.add_argument("--sampling_ratio", default=0.25, type=float)
    parser.add_argument("--max_epoch", default=20, type=int, help="epochs")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model_file", default="bgl_transformer.hdf5", type=str, help="path to the trained model")
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    (x_tr, y_tr), (x_te, y_te) = data_loader.load_supercomputers(
        args.log_file, train_ratio=0.8, windows_size=args.window_size, step_size=args.step_size,
        sampling_method =args.sampling_method, sampling_ratio=args.sampling_ratio, e_type='gpt2', mode='balance')

    model = train(x_tr, y_tr, args.max_epoch, args.batch_size, args.model_file)
    test_model(model, x_te, y_te, batch_size=args.batch_size)