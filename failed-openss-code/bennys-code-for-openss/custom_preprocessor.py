import pandas as pd
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.preprocess.actions import (
    Action,
    Overlay,
    AudioClipLoader,
    AudioTrim
)
from opensoundscape.spectrogram import MelSpectrogram


class MelSpectrogramPreprocessor(SpectrogramPreprocessor):

    def __init__(self, sample_duration, overlay_df=None, out_shape=[224, 224, 3]):
        super(MelSpectrogramPreprocessor, self).__init__(sample_duration, overlay_df, out_shape)

        # custom preprocessing pipeline
        self.pipeline = pd.Series(
            {
                "load_audio": AudioClipLoader(),
                # if we are augmenting and get a long file, take a random trim from it
                "random_trim_audio": AudioTrim(is_augmentation=True, random_trim=True),
                # otherwise, we expect to get the correct duration. no random trim
                "trim_audio": AudioTrim(),  # trim or extend (w/silence) clips to correct length
                "to_spec": Action(MelSpectrogram.from_audio),
                "bandpass": Action(
                    MelSpectrogram.bandpass, min_f=0, max_f=11025, out_of_bounds_ok=False
                ),
                "to_img": Action(
                    MelSpectrogram.to_image,
                    shape=out_shape[0:2],
                    channels=out_shape[2],
                    return_type="torch",
                ),
                "overlay": Overlay(
                    is_augmentation=True, overlay_df=overlay_df, update_labels=False
                )
                if overlay_df is not None
                else None,
                "time_mask": Action(actions.time_mask, is_augmentation=True),
                "frequency_mask": Action(actions.frequency_mask, is_augmentation=True),
                "add_noise": Action(
                    actions.tensor_add_noise, is_augmentation=True, std=0.005
                ),
                "rescale": Action(actions.scale_tensor),
                "random_affine": Action(
                    actions.torch_random_affine, is_augmentation=True
                ),
            }
        )

        # remove overlay if overlay_df was not specified
        if overlay_df is None:
            self.pipeline.drop("overlay", inplace=True)
