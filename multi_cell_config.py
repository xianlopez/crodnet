from config.MultiCellConfigBase import MultiCellConfiguration
import CrodnetOptions


class UpdateMultiCellConfiguration(MultiCellConfiguration):

    dataset_name = 'VOC0712_filtered'

    weights_file = r'C:\development\crodnet\experiments\2019\2019_02_18\model-3'

    multi_cell_opts = CrodnetOptions.MultiCellOptions()
    multi_cell_opts.debug = True

    write_results = True

    percent_of_data = 2
    # buffer_size = 300
    # num_workers = 1
    # root_of_datasets = '/home/xian/datasets'


