from Read_dataset.FourDME import FourDME_Dataset, EMOTION2IDX, IDX2EMOTION  # noqa: F401  (re-exported for convenience)


class CASME2_Dataset(FourDME_Dataset):
    """Same constructor signature and __getitem__ behavior as
    FourDME_Dataset -- only subject-id parsing differs
    """

    def _parse_subject_id(self, folder_name):
        # folder_name = f"sub{Subject:02d}_{Filename}", e.g. "sub01_EP02_01f".
        # Filename itself may contain underscores (it does: "EP02_01f"),
        # so only the FIRST "_"-separated token is taken -- that's always
        # exactly the "sub{XX}" piece, since run_inference_flow_casme2.py
        # never puts an underscore inside the "sub{XX}" prefix itself.
        return folder_name.split("_")[0]