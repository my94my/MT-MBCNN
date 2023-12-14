from collections import OrderedDict
from random import gauss

from mne import events_from_annotations
from mne.io import read_raw_gdf, RawArray
from numpy import where, min, nan, isnan, nanmean, zeros, uint8, concatenate, array, linspace
from scipy.interpolate import interp1d
from scipy.io import loadmat

from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply


def extract_data(datafile, correct_nan_values=True):
    raw_gdf = read_raw_gdf(datafile, stim_channel="auto")
    raw_gdf.load_data()
    # Preprocessing
    data = raw_gdf.get_data()
    if correct_nan_values:
        # correct nan values
        for i in range(data.shape[0]):
            # set to nan
            channel = data[i]
            data[i] = where(channel == min(channel), nan, channel)
            mask = isnan(data[i])
            # replace nans by nanmean function
            channel_mean = nanmean(data[i])
            data[i, mask] = channel_mean

    gdf_events = events_from_annotations(raw_gdf)
    raw_gdf = RawArray(data, raw_gdf.info, verbose="WARNING")
    # remember gdf events
    raw_gdf.info["gdf_events"] = gdf_events
    return raw_gdf.drop_channels(['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])


def extract_events(label_file, gdf_events):
    # all events
    events, name_to_code = gdf_events

    if 'cue unknown/undefined (used for BCI competition) ' not in name_to_code:
        # 2 classes
        if 'Feedback (continuous) - onset (BCI experiment)' not in name_to_code:
            trial_codes = [3, 4]
        else:
            trial_codes = [4, 5]
    else:
        # 3 = 'cue unknown/undefined (used for BCI competition) '
        trial_codes = [4]
    print(name_to_code)
    trial_mask = [event_code in trial_codes for event_code in events[:, 2]]
    trial_events = events[trial_mask]
    # assert len(trial_events) == 288, f"Got {len(trial_events)} markers"
    # overwrite with markers from labels file
    classes = loadmat(label_file)["classlabel"].squeeze()
    trial_events[:, 2] = classes

    # now also create 0-1 vector for rejected trials
    if 'cue unknown/undefined (used for BCI competition) ' not in name_to_code:
        if 'Feedback (continuous) - onset (BCI experiment)' not in name_to_code:
            sot = 2
            row = 1
        else:
            sot = 3
            row = 2
    else:
        # 4 = 'cue unknown/undefined (used for BCI competition) '
        sot = 3
        row = 2
    trial_start_events = events[events[:, 2] == sot]  # 2 = 'Start of Trial, Trigger at t=0s'
    assert len(trial_start_events) == len(trial_events)
    artifact_trial_mask = zeros(len(trial_events), dtype=uint8)
    artifact_events = events[events[:, 2] == row]  # 1 = 'Rejection of whole trial'

    for artifact_time in artifact_events[:, 0]:
        i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
        artifact_trial_mask[i_trial] = 1

    return trial_events, artifact_trial_mask


def load_gdf(data_folder, subject_id, start_offset_ms=-500, end_offset_ms=4000, low_cut_hz=0, high_cut_hz=60, noise_low=30, noise_high=40):
    data = []
    label = []
    noise1_data = []
    noise2_data = []
    for et in ('01T', '02T', '03T', '04E', '05E'):
        data_file = f'{data_folder}/B0{subject_id}{et}.gdf'
        label_file = f'{data_folder}/B0{subject_id}{et}.mat'

        raw_gdf = extract_data(data_file)
        events, artifact_trial_mask = extract_events(label_file, raw_gdf.info["gdf_events"])
        raw_gdf.info["events"] = events
        raw_gdf.info["artifact_trial_mask"] = artifact_trial_mask
        # convert from volt to microvolt
        raw_gdf = mne_apply(lambda a: a * 1e6, raw_gdf)
        # bandpass filter
        noise1_gdf = mne_apply(
            lambda a: bandpass_cnt(a, noise_high, high_cut_hz, raw_gdf.info["sfreq"], axis=1), raw_gdf)
        noise2_gdf = mne_apply(
            lambda a: bandpass_cnt(a, noise_low, noise_high, raw_gdf.info["sfreq"], axis=1), raw_gdf)
        raw_gdf = mne_apply(
            lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, raw_gdf.info["sfreq"], axis=1), raw_gdf)
        # standardization
        noise1_gdf = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=0.001,
                                                      init_block_size=1000, eps=1e-4).T, noise1_gdf)
        noise2_gdf = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=0.001,
                                                      init_block_size=1000, eps=1e-4).T, noise2_gdf)
        raw_gdf = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=0.001,
                                                      init_block_size=1000, eps=1e-4).T, raw_gdf)
        marker = OrderedDict([("Left Hand", [1]), ("Right Hand", [2])])
        noise1_and_target = create_signal_target_from_raw_mne(noise1_gdf, marker, (start_offset_ms, end_offset_ms))
        noise2_and_target = create_signal_target_from_raw_mne(noise2_gdf, marker, (start_offset_ms, end_offset_ms))
        signal_and_target = create_signal_target_from_raw_mne(raw_gdf, marker, (start_offset_ms, end_offset_ms))
        data.append(signal_and_target.X)
        label.append(signal_and_target.y)
        noise1_data.append(noise1_and_target.X)
        noise2_data.append(noise2_and_target.X)
    return concatenate(data, axis=0), concatenate(label, axis=0), concatenate(noise2_data, axis=0), concatenate(
        noise1_data, axis=0)
