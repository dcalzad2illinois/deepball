from typing import Dict, List
from pymlb.data import DeepBallConnector, AQStatType, AQTimeDuration, AQGroupType
import numpy as np


class AggregateQueries(DeepBallConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def new_aggregate_query(self, time_frame: AQTimeDuration, group: AQGroupType, stat_type: AQStatType,
                            parameters: Dict[str, object] = None) -> List[object]:
        get_results = self._get(resource="/".join(["aggregatequeries", time_frame.value, group.value, stat_type.value]),
                                parameters=parameters, cache=True)
        return list(get_results["resource"])

    @staticmethod
    def query_to_matrices(results, key_retriever, field_list: List[str], additional_fields: List = None):
        # split the rows up by key and filter  the columns
        output_dictionary = {}
        for result in results:
            # create the key for this row
            key = key_retriever(result)
            if key not in output_dictionary:
                output_dictionary[key] = []

            # filter the right columns
            this_row = []
            for field in field_list:
                this_row.append(result[field])

            if additional_fields is not None:
                for f in additional_fields:
                    this_row.append(f(result))

            output_dictionary[key].append(this_row)

        # now that we have all the rows for each key, convert each set of rows to a matrix
        for key, rows in output_dictionary.items():
            output_dictionary[key] = np.array(rows)

        return output_dictionary
