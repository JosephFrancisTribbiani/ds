import torch
import numpy as np
from torch.utils.data import Dataset


class NBMEDataset(Dataset):
    def __init__(self, tokenizer, feature_texts: np.ndarray, pn_histories: np.ndarray, 
                 locations: np.ndarray, max_length: int = 512):
        self.tokenizer = tokenizer
        self.feature_texts = feature_texts
        self.pn_histories = pn_histories
        self.locations = locations
        self.max_length = max_length

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, idx: int):
        def prepare_input(curr_pn_history: str, curr_feature_text: str):
            """
            Функция предназначена для кодирования пары curr_pn_history, curr_feature_text
            :param curr_pn_history: текущая pn_history
            :param curr_feature_text: текущий feature_text
            :return: encoded - словарь {'input_ids': torch.tensor, 'token_type_ids': torch.tensor,
            'attention_mask': torch.tensor}
            """
            encoded = self.tokenizer(curr_pn_history, curr_feature_text, add_special_tokens=True,
                                     max_length=self.max_length, padding="max_length", return_offsets_mapping=False)
            return {k: torch.tensor(v, dtype=torch.long) for k, v in encoded.items()}

        def prepare_label(curr_pn_history: str, curr_location: list):
            """
            Функция предназначена для получения labels из pn_history и аннотаций.
            :param curr_pn_history: текущая pn_history
            :param curr_location: текущая разметка аннотаций
            :return: labels - разметка токенов, где значения
              -1 - не относящиеся к pn_history токены ([CLS], [SEP], [PAD])
              0 - относящиеся к pn_history токены, но не относящиеся к аннотациям
              1 - токены, относящиеся как и к pn_history, так и к аннотациям
            """
            # кодируем текущий pn_history
            encoded = self.tokenizer(curr_pn_history, add_special_tokens=True, max_length=self.max_length, 
                                     padding='max_length', return_offsets_mapping=True)
            # порядковые номера кодированной последовательности, относящиеся к исходной последовательности
            # т.е порядковые номера токенов [CLS], [SEP] и нулей, добавленных через padding не присутствуют в
            # последовательности - [PAD]
            seq_idxes = np.where(np.array(encoded.sequence_ids()) == 0)[0]

            # из offset_mapping (список первого и последнего индексов строки, относящихся к каждому токену, полученному
            # с помощью tokenizer)
            # фильтруем только относящиеся к последовательности через seq_idxes
            offset_mapping = encoded.get('offset_mapping')
            offset_mapping = np.array(offset_mapping)
            offset_mapping_filtered = offset_mapping[seq_idxes]

            # формируем шаблон labels, где
            # -1 - не относящиеся к последовательности токены
            #  0 - относящиеся к последовательности токены
            label = np.full(len(offset_mapping), -1)
            label[seq_idxes] = 0

            # предобрабатываем значения с аннотациями
            locations = [tuple(map(int, loc.split())) for locs in curr_location for loc in locs.split(';')]
            locations = sorted(locations, key=lambda loc: loc[0])

            # итерируемся методом двух указателей и отмечаем токены, относящиеся к аннотацияем единицами
            annotated_idxes = list()
            pointer_mapping = 0
            for _, (start_loc, end_loc) in enumerate(locations):
                while pointer_mapping < offset_mapping_filtered.shape[0] and \
                        offset_mapping_filtered[pointer_mapping][0] < end_loc:
                    if offset_mapping_filtered[pointer_mapping][1] > start_loc:
                        annotated_idxes.append(pointer_mapping)
                    pointer_mapping += 1
            annotated_seq_idxes = seq_idxes[annotated_idxes]
            label[annotated_seq_idxes] = 1
            label = torch.tensor(label, dtype=torch.float)
            return label

        curr_pn_history = self.pn_histories[idx]
        curr_feature_text = self.feature_texts[idx]
        curr_location = self.locations[idx]
        input = prepare_input(curr_pn_history, curr_feature_text)
        label = prepare_label(curr_pn_history, curr_location)
        return input, label
