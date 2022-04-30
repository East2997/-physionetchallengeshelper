#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
from helper_code import *
import scipy as sp, scipy.stats, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from util.utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.signal

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

debug_result_root_path = "debug_result"
check_dir(debug_result_root_path)


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)

    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    labels = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i + 1, num_patient_files))

        hs_data = HeartSound(patient_files[i], data_folder)
        hs_data.load_hs_note()
        hs_data.preprocess_recordings()
        current_patient_data = hs_data.get_patient_data()
        current_recordings = hs_data.get_recordings()
        # hs_data.plot_data()

        # plot_data([current_recordings[0]], Fs=Fs, pic_name=get_patient_id(current_patient_data), is_show=True)
        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        labels.append(current_labels)


    features = np.vstack(features)
    labels = np.vstack(labels)

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    n_estimators = 10  # Number of trees in the forest.
    max_leaf_nodes = 100  # Maximum number of leaf nodes in each tree.
    random_state = 123  # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
                                        random_state=random_state).fit(features, labels)

    # Save the model.
    save_challenge_model(model_folder, classes, imputer, classifier)

    if verbose >= 1:
        print('Done.')

    #TODO：利用run_challenge_model()计算准确度、Kappa和损失函数

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes = model['classes']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    features = get_features(data, recordings)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Get classifier probabilities.
    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(probabilities)
    labels[idx] = 1

    return classes, labels, probabilities


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, classes, imputer, classifier):
    d = {'classes': classes, 'imputer': imputer, 'classifier': classifier}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


# Extract features from the data.
def get_features(data, recordings):
    # TODO：改进特征提取
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations == num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i]) > 0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)


class HeartSound():
    def __init__(self, patient_file, data_folder):
        self.reference_recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
        self.patient_file = patient_file
        self.data_folder = data_folder
        self.recording_locations = []
        self.recording_list = []
        self.processed_recording_list = []
        self.hs_note_list = []
        self.Fs = []
        self.read_recordings()

    def read_recordings(self):
        self.current_patient_data = load_patient_data(self.patient_file)
        self.recording_list, self.Fs = load_recordings(self.data_folder, self.current_patient_data,
                                                       get_frequencies=True)
        self.data_number = get_patient_id(self.current_patient_data)
        self.recording_locations = get_locations(self.current_patient_data)

    def get_patient_data(self):
        return self.current_patient_data

    def get_recordings(self):
        return self.recording_list

    def load_hs_note(self):
        num_locations = get_num_locations(self.current_patient_data)
        recording_information = self.current_patient_data.split('\n')[1:num_locations + 1]
        for i in range(num_locations):
            entries = recording_information[i].split(' ')
            tsv_file = entries[3]
            filename = os.path.join(self.data_folder, tsv_file)
            hs_note = pd.read_csv(filename, sep='\t', header=None).to_numpy()
            self.hs_note_list.append(hs_note)
        return self.hs_note_list

    def preprocess_recordings(self):
        # TODO:更多预处理
        for idx, recording in enumerate(self.recording_list):
            processed_recording = StandardScaler().fit_transform(recording.reshape(-1, 1)).reshape(1, -1)[0]
            processed_recording_1 = self.hs_clean(processed_recording, self.Fs[idx])
            # drawPic(np.array(recording).flatten(), is_show=False,
            #         title=f"{self.data_number}  location={self.recording_locations[idx]}",
            #         to_file=os.path.join(debug_result_root_path, f"{self.data_number}_{self.recording_locations[idx]}_raw.png"))
            drawPic(np.array(processed_recording_1).flatten(), Fs=self.Fs[idx], is_show=True,
                    extra_data=(self.hs_note_list[idx][:,0], self.hs_note_list[idx][:,2]),
                    title=f"{self.data_number}  location={self.recording_locations[idx]}",
                    fig_size=(12.4,4.8),
                    to_file=os.path.join(debug_result_root_path,
                                         f"{self.data_number}_{self.recording_locations[idx]}_clean.png"))
            # plot_data([recording, processed_recording, processed_recording_1], pic_name="proccess_test", Fs=self.Fs[idx], is_show=True)
            self.processed_recording_list.append(processed_recording)

    def hs_clean(self, data, Fs):
        order = int(0.3 * Fs)
        if order % 2 == 0:
            order += 1  # Enforce odd number
        # -> filter_signal()
        frequency = [40, 100]
        frequency = (
                2 * np.array(frequency) / Fs
        )  # Normalize frequency to Nyquist Frequency (Fs/2).
        #     -> get coeffs
        a = np.array([1])
        b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)
        # _filter_signal()
        filtered = scipy.signal.filtfilt(b, a, data)
        return filtered

    def plot_data(self, is_show=True):
        data = self.recording_list[0]
        Fs = self.Fs[0]
        plt.figure()
        x = np.linspace(0, data.shape[0] / Fs, len(data))
        plt.plot(x, data)
        if is_show:
            plt.show()
        plt.close()


