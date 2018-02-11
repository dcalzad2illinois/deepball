from typing import Dict
from urllib.error import HTTPError
from pymlb.data import DeepBallConnector
import numpy as np


class VectorSlots(DeepBallConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_vector_values(self, vector_id: int, parameters: Dict[str, object]):
        return self._get("vectors/" + str(vector_id) + "/normalizedvalues", parameters, cache=True)['resource']

    def get_vector_fields(self, vector_id: int):
        return self._get("vectors/" + str(vector_id) + "/fields", {}, cache=True)['resource']

    def get_vector_slots(self, vector_id: int, slot_name: str):
        return self._get("vectors/" + str(vector_id) + "/normalizedslots/" + slot_name, {}, cache=True)['resource']['values']

    def put_vector_slots(self, vector_id: int, slot_name: str, data: Dict[str, Dict[str, np.ndarray]],
                         pre_normalized: bool = False):
        has_base_table = slot_name.startswith("in_") or slot_name.startswith("out_")

        # try to create the slot. if it fails, that's fine since it must already be there
        try:
            self._post("vectors/" + str(vector_id) + "/normalizedslots", {
                "slotname": slot_name,
                "table_alias": (
                    (slot_name if "." not in slot_name else slot_name.split('.')[0]) if has_base_table else None),
                "length": min(min(len(entry) for entry in entries.values()) for entries in data.values())
            })
        except HTTPError:
            # perhaps this was caused by a 409 error. if that's the case, just keep going
            pass

        # now add the values
        update_result = self._put(
            "vectors/" + str(vector_id) + "/" + (
            "unnormalizedslots" if pre_normalized else "normalizedslots") + "/" + slot_name, {
                "values": {chainid: {entryid: entry if isinstance(entry, list) else entry.tolist() for entryid, entry in
                                     entries.items()} for chainid, entries in data.items()}
            })

        return update_result
