"""
Leave-One-Subject-Out cross-validation for the CBM pipeline.
Evaluates LR sequential and LR independent across all 10 subjects.
Run from the explains-cbm/ directory:
    python loso_cv.py
"""

import os
import random

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
# Force CPU — tensorflow-metal deadlocks on LSTM layers
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import LearningRateScheduler

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config (mirrors notebook Cell 3) ─────────────────────────────────────────

CONCEPT_NAMES = os.listdir('concepts')
if '.ipynb_checkpoints' in CONCEPT_NAMES:
    CONCEPT_NAMES.remove('.ipynb_checkpoints')
CONCEPT_NAMES = sorted(CONCEPT_NAMES)

NUMBER_OF_CONCEPTS = len(CONCEPT_NAMES)

LABEL_NAMES = [
    "Standing", "Sitting", "Lying", "Walking", "Stairs",
    "WaistBends", "ArmRaise", "KneeBend", "Cycling",
    "Jogging", "Running", "Jump",
]

SENSOR_GROUPS = {
    "chest_acc":  [0, 1, 2],
    "ankle_acc":  [5, 6, 7],
    "ankle_gyro": [8, 9, 10],
    "ankle_mag":  [11, 12, 13],
    "arm_acc":    [14, 15, 16],
    "arm_gyro":   [17, 18, 19],
    "arm_mag":    [20, 21, 22],
}

CONCEPT_SENSORS = {
    "chest_inclination":           ["chest_acc"],
    "body_elevation":              ["chest_acc", "ankle_acc", "ankle_gyro"],
    "arm_role":                    ["arm_acc", "arm_gyro", "arm_mag"],
    "ground_impact_force":         ["ankle_acc", "ankle_gyro"],
    "locomotion":                  ["ankle_acc", "ankle_gyro"],
    "movement_cycle_frequency":    ["ankle_acc", "ankle_gyro"],
    "movement_intensity":          ["chest_acc", "arm_acc", "arm_gyro", "arm_mag",
                                    "ankle_acc", "ankle_gyro", "ankle_mag"],
    "limb_coordination_imbalance": ["arm_acc", "arm_gyro", "ankle_acc", "ankle_gyro"],
}

# ── Data helpers (mirrors notebook Cells 6-9) ─────────────────────────────────

def get_windows(data, sample_rate=50, window_sec=2, overlap=0.5):
    window_size = window_sec * sample_rate
    stride = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(data) - window_size + 1, stride):
        windows.append(data[start:start + window_size])
    return np.array(windows), window_size


def get_most_frequent(x):
    return stats.mode(x, axis=1, keepdims=False)[0]


def normalize(train, test):
    shape_train = train.shape
    shape_test = test.shape
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train.reshape(-1, shape_train[-1]))
    test_flat  = scaler.transform(test.reshape(-1, shape_test[-1]))
    return train_flat.reshape(shape_train), test_flat.reshape(shape_test), scaler


def sensor_concept_label_split(train_df, test_df):
    train_sensors = train_df.iloc[:, :-NUMBER_OF_CONCEPTS]
    test_sensors  = test_df.iloc[:,  :-NUMBER_OF_CONCEPTS]

    train_sensors, w = get_windows(train_sensors)
    test_sensors,  _ = get_windows(test_sensors)

    train_concepts = np.asarray(train_df.iloc[:, -NUMBER_OF_CONCEPTS:]).astype('float32')
    test_concepts  = np.asarray(test_df.iloc[:,  -NUMBER_OF_CONCEPTS:]).astype('float32')

    train_concepts, _ = get_windows(train_concepts)
    test_concepts,  _ = get_windows(test_concepts)

    train_concepts = get_most_frequent(train_concepts)
    test_concepts  = get_most_frequent(test_concepts)

    train_labels = get_most_frequent(train_sensors[:, :, 23])
    test_labels  = get_most_frequent(test_sensors[:, :, 23])

    train_sensors = np.delete(train_sensors, 23, axis=2)
    test_sensors  = np.delete(test_sensors,  23, axis=2)

    train_sensors, test_sensors, _ = normalize(train_sensors, test_sensors)
    train_concepts, test_concepts, _ = normalize(train_concepts, test_concepts)

    return train_sensors, test_sensors, train_concepts, test_concepts, train_labels, test_labels, w


# ── Model builder (mirrors notebook Cell 11) ─────────────────────────────────

def build_model_mask(w):
    input_list = []
    for group in SENSOR_GROUPS.values():
        input_list.append(Input(shape=(w, len(group))))

    def branch(inp):
        x = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(inp)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False, recurrent_dropout=0.2)(x)
        x = Dropout(0.3)(x)
        return x

    branch_list = [branch(inp) for inp in input_list]
    branch_outputs = {name: branch_list[i] for i, name in enumerate(SENSOR_GROUPS.keys())}

    concept_outputs = []
    for concept_name in CONCEPT_NAMES:
        allowed = [branch_outputs[s] for s in CONCEPT_SENSORS[concept_name]]
        x = allowed[0] if len(allowed) == 1 else Concatenate(axis=-1)(allowed)
        concept_outputs.append(Dense(1, activation='linear')(x))

    output = Concatenate(axis=-1)(concept_outputs)
    return Model(inputs=input_list, outputs=output)

def build_lr():
    model = Sequential()
    model.add(Dense(len(LABEL_NAMES), activation='softmax'))
    return model


def build_model_nomask(w):
    input_list = []
    for group in SENSOR_GROUPS.values():
        input_list.append(Input(shape=(w, len(group))))

    def branch(inp):
        x = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(inp)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False, recurrent_dropout=0.2)(x)
        x = Dropout(0.3)(x)
        return x

    branch_list = []
    for inp in input_list:
        branch_list.append(branch(inp))
    
    merged = Concatenate(axis=-1)(branch_list) 
    output = Dense(NUMBER_OF_CONCEPTS, activation='linear')(merged)
    return Model(inputs=input_list, outputs=output)


def build_model_bb(w):
    input_list = []
    train_sensors_groups = []
    test_sensors_groups = []

    for group in SENSOR_GROUPS.values():
        input_list.append(Input(shape=(w, len(group))))

    def branch(inp):
        x = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(inp)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False, recurrent_dropout=0.2)(x)
        x = Dropout(0.3)(x)
        return x

    branch_list = []

    for inp in input_list:
        branch_list.append(branch(inp))

    merged = Concatenate(axis=-1)(branch_list) 

    # mock the concept bottleneck layer
    bottleneck = Dense(NUMBER_OF_CONCEPTS, activation='linear')(merged)

    # mock logistic regression layer
    output = Dense(12, activation='softmax')(bottleneck)
    black_box = Model(inputs=input_list, outputs=output)
    return black_box


def lr_schedule(epoch, lr):
    if epoch == 10:
        return 1e-4
    return lr


# ── LOSO loop ─────────────────────────────────────────────────────────────────

def load_subject(i):
    subject = pd.read_csv(
        f'dataset/mHealth_subject{i}.log', sep=r'\s+', header=None
    )
    for folder in CONCEPT_NAMES:
        subject[folder] = pd.read_csv(
            f'concepts/{folder}/{folder}_subject{i}.csv'
        )
    subject = subject[subject.iloc[:, 23] != 0].copy()
    subject.iloc[:, 23] = subject.iloc[:, 23] - 1
    return subject


RESULTS_FILE = './results/loso_results.csv'


def run_loso():
    # resume from existing results if present
    if os.path.exists(RESULTS_FILE):
        done = pd.read_csv(RESULTS_FILE)
        completed = set(done['subject'].tolist())
        print(f'Resuming — {len(completed)} fold(s) already complete: {sorted(completed)}')
    else:
        done = pd.DataFrame(columns=['subject', 
                                     'concept_mae_mask', 'seq_acc_mask', 'ind_acc_mask', 
                                     'concept_mae_nomask', 'seq_acc_nomask', 'ind_acc_nomask', 
                                     'bb_acc'])
        completed = set()

    for test_subject in range(1, 11):
        if test_subject in completed:
            print(f'\n── Fold {test_subject}/10: skipping (already done) ──')
            continue

        print(f'\n── Fold {test_subject}/10: test subject {test_subject} ──')

        train_df = pd.concat(
            [load_subject(i) for i in range(1, 11) if i != test_subject],
            axis=0, ignore_index=True
        )
        test_df = load_subject(test_subject)

        (train_sensors, test_sensors,
         train_concepts, test_concepts,
         train_labels, test_labels, w) = sensor_concept_label_split(train_df, test_df)

        train_groups = [train_sensors[:, :, g] for g in SENSOR_GROUPS.values()]
        test_groups  = [test_sensors[:, :, g]  for g in SENSOR_GROUPS.values()]

        tf.random.set_seed(SEED)

        # BLACK BOX RESULTS 
        model_bb = build_model_bb(w)
        model_bb.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_bb.fit(
            train_groups, train_labels,
            validation_data=(test_groups, test_labels),
            epochs=30, batch_size=64,
            callbacks=[LearningRateScheduler(lr_schedule, verbose=0)],
            verbose=1,
        )
        bb_acc = model_bb.evaluate(test_groups, test_labels, verbose=1)[1]

        # CBM RESULTS - SENSOR MASKING
        model = build_model_mask(w)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber(delta=1.0), metrics=['mae'])
        model.fit(
            train_groups, train_concepts,
            validation_data=(test_groups, test_concepts),
            epochs=30, batch_size=64,
            callbacks=[LearningRateScheduler(lr_schedule, verbose=0)],
            verbose=1,
        )

        seq_train = model.predict(train_groups, verbose=0)
        seq_test  = model.predict(test_groups,  verbose=0)

        c_pred = model.predict(test_groups)
        c_mae_mask = np.mean(np.abs(c_pred - test_concepts))

        lr_seq = build_lr()
        lr_seq.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lr_seq.fit(seq_train, train_labels, epochs=30, batch_size=64)

        lr_ind = build_lr()
        lr_ind.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lr_ind.fit(train_concepts, train_labels, epochs=30, batch_size=64)

        seq_acc_mask = lr_seq.evaluate(seq_test, test_labels, verbose=1)[1]
        ind_acc_mask = lr_ind.evaluate(seq_test, test_labels, verbose=1)[1]

        # CBM RESULTS - NO MASKING
        model = build_model_nomask(w)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber(delta=1.0), metrics=['mae'])
        model.fit(
            train_groups, train_concepts,
            validation_data=(test_groups, test_concepts),
            epochs=30, batch_size=64,
            callbacks=[LearningRateScheduler(lr_schedule, verbose=0)],
            verbose=1,
        )

        seq_train = model.predict(train_groups, verbose=0)
        seq_test  = model.predict(test_groups,  verbose=0)

        c_pred = model.predict(test_groups)
        c_mae_nomask = np.mean(np.abs(c_pred - test_concepts))

        lr_seq = build_lr()
        lr_seq.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lr_seq.fit(seq_train, train_labels, epochs=30, batch_size=64)

        lr_ind = build_lr()
        lr_ind.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lr_ind.fit(train_concepts, train_labels, epochs=30, batch_size=64)

        seq_acc_nomask = lr_seq.evaluate(seq_test, test_labels, verbose=1)[1]
        ind_acc_nomask = lr_ind.evaluate(seq_test, test_labels, verbose=1)[1]

        row = pd.DataFrame([{'subject': test_subject, 
                             'concept_mae_mask': c_mae_mask, 'seq_acc_mask': seq_acc_mask, 'ind_acc_mask': ind_acc_mask,
                             'concept_mae_nomask': c_mae_nomask, 'seq_acc_nomask': seq_acc_nomask, 
                             'ind_acc_nomask': ind_acc_nomask,
                             'bb_acc': bb_acc}])
        done = pd.concat([done, row], ignore_index=True)
        done.to_csv(RESULTS_FILE, index=False)
        
        print(f'  Sequential - Masked: {seq_acc_mask:.3f}  |  Independent - Masked: {ind_acc_mask:.3f}  |    Sequential - No mask: {seq_acc_nomask:.3f}  |  Independent - No mask: {ind_acc_nomask:.3f}  |  Black box: {bb_acc:.3f}  [saved]')

    seq_scores_mask = done['seq_acc_mask'].tolist()
    ind_scores_mask = done['ind_acc_mask'].tolist()
    seq_scores_nomask = done['seq_acc_nomask'].tolist()
    ind_scores_nomask = done['ind_acc_nomask'].tolist()
    
    bb_scores = done['bb_acc'].tolist()

    print('\n── LOSO results ─────────────────────────────────────')
    print(f'Sequential - Masked:  {100 * np.mean(seq_scores_mask):.3f} ± {100 * np.std(seq_scores_mask):.3f}')
    print(f'Independent - Masked: {100 * np.mean(ind_scores_mask):.3f} ± {100 * np.std(ind_scores_mask):.3f}')
    print(f'Sequential - No mask:  {100 * np.mean(seq_scores_nomask):.3f} ± {100 * np.std(seq_scores_nomask):.3f}')
    print(f'Independent - No mask: {100 * np.mean(ind_scores_nomask):.3f} ± {100 * np.std(ind_scores_nomask):.3f}')
    print(f'Black box: {100 * np.mean(bb_scores):.3f} ± {100 * np.std(bb_scores):.3f}')
    print(f'\nPer-fold sequential:  {[round(s, 3) for s in seq_scores_mask]}')
    print(f'Per-fold independent: {[round(s, 3) for s in ind_scores_mask]}')
    print(f'Per-fold black box: {[round(s, 3) for s in bb_scores]}')


if __name__ == '__main__':
    run_loso()
