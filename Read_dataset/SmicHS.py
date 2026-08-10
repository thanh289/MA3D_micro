from Read_dataset.FourDME import FourDME_Dataset


class SmicHS_Dataset(FourDME_Dataset):

    EMOTION2IDX = {
        "Negative": 0,
        "Positive": 1,
        "Surprise": 2,
    }
    IDX2EMOTION = {v: k for k, v in EMOTION2IDX.items()}

    def _parse_subject_id(self, folder_name):
        # folder_name = Filename, e.g. "s2_sur_01" -> subject "s2".
        return folder_name.split("_")[0]