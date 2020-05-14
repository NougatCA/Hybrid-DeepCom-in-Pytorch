from torch.utils.data import Dataset

import utils


class CodePtrDataset(Dataset):

    def __init__(self, code_path, ast_path, nl_path):
        # get lines
        codes = utils.load_dataset(code_path)
        asts = utils.load_dataset(ast_path)
        nls = utils.load_dataset(nl_path)

        if len(codes) != len(asts) or len(codes) != len(nls) or len(asts) != len(nls):
            raise Exception('The lengths of three dataset do not match.')

        self.codes, self.asts, self.nls = utils.filter_data(codes, asts, nls)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.nls[index], self.nls[index], self.nls[index]

    def get_dataset(self):
        return self.nls, self.nls, self.nls
