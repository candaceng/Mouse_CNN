from typing import Dict


class FilterParams:
    required_keys = [
        'model_id', 'weight_dom_0', 'weight_dom_1', 'weight_non_dom_0', 'weight_non_dom_1',
        'kpeaks_dom_0', 'kpeaks_dom_1', 'kpeaks_non_dom_0', 'kpeaks_non_dom_1',
        'delay_dom_0', 'delay_dom_1', 'delay_non_dom_0', 'delay_non_dom_1',
        'spatial_size', 'sf_sep', 'tuning_angle'
    ]

    def __init__(self, param_table: Dict[str, float]):
        self._validate_table_structure(param_table)
        self.model_id = param_table['model_id']
        self.is_dominant = self.model_id not in ['sONsOFF_001', 'sONtOFF_001']

        self.weight_dom = (param_table['weight_dom_0'], param_table['weight_dom_1'])
        self.weight_non_dom = (param_table['weight_non_dom_0'], param_table['weight_non_dom_1'])
        self.kpeaks_dom = (param_table['kpeaks_dom_0'], param_table['kpeaks_dom_1'])
        self.kpeaks_non_dom = (param_table['kpeaks_non_dom_0'], param_table['kpeaks_non_dom_1'])
        self.delay_dom = (param_table['delay_dom_0'], param_table['delay_dom_1'])
        self.delay_non_dom = (param_table['delay_non_dom_0'], param_table['delay_non_dom_1'])

        self.spatial_size = param_table['spatial_size']
        self.sf_sep = param_table['sf_sep']
        self.tuning_angle = param_table['tuning_angle']

    def _validate_table_structure(self, param_table: Dict[str, float]):
        missing_keys = [key for key in self.required_keys if key not in param_table]
        if missing_keys:
            raise ValueError(f"Missing keys in parameter table: {missing_keys}")

    def get_filter_properties(self, is_dominant: bool):
        if is_dominant:
            return self.weight_dom, self.kpeaks_dom, self.delay_dom
        else:
            return self.weight_non_dom, self.kpeaks_non_dom, self.delay_non_dom
