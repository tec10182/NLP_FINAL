# import unittest
# from multivalue import BaseDialect, Dialects
# import pandas as pd
# import json


# class TestStringMethods(unittest.TestCase):
#     def test_all_methods(self):
#         D = Dialects.DialectFromVector(dialect_name="all")
#         feature_id_to_function_name = D.load_dict("resources/feature_id_to_function_name.json")
#         uts = D.load_dict("resources/unittests.json")

#         failure_cases = 0
#         for feature_id in uts:
#             for function_name in feature_id_to_function_name[feature_id]:
#                 for ut in uts[feature_id]:
#                     sae = ut["gloss"]
#                     dialect = ut["dialect"]
#                     D.clear()
#                     D.update(sae)
#                     method = getattr(D, function_name)
#                     method()
#                     synth_dialect = D.surface_fix_spacing(D.compile_from_rules())
#                     if synth_dialect.lower() not in {x.lower() for x in dialect}:
#                         failure_cases += 1
#                         print(feature_id, function_name, dialect, synth_dialect)
#         assert failure_cases == 0
import torch
import numpy as np

# Stanza가 사용하는 pickle 내부 객체 허용
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])

import unittest
from multivalue import BaseDialect, Dialects
import pandas as pd
import json
import torch
import numpy as np
from stanza.models.coref.config import Config  # ? 핵심 클래스 import

# ? PyTorch에 필요한 pickle 대상 객체 허용 등록
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    Config  # ?? 이 줄이 새롭게 추가됨
])


class TestStringMethods(unittest.TestCase):
    def test_all_methods(self):
        D = Dialects.DialectFromVector(dialect_name="all")
        feature_id_to_function_name = D.load_dict("resources/feature_id_to_function_name.json")
        uts = D.load_dict("resources/unittests.json")

        failure_cases = 0

        for feature_id in uts:
            for function_name in feature_id_to_function_name.get(feature_id, []):
                for ut in uts[feature_id]:
                    sae = ut["gloss"]
                    dialects = ut["dialect"]

                    with self.subTest(feature_id=feature_id, function=function_name, gloss=sae):
                        D.clear()
                        D.update(sae)

                        try:
                            method = getattr(D, function_name)
                            method()
                        except Exception as e:
                            self.fail(f"[{feature_id}] {function_name} raised exception: {e}")

                        synth_dialect = D.surface_fix_spacing(D.compile_from_rules())
                        synth_dialect = synth_dialect.lower()
                        expected_dialects = {x.lower() for x in dialects}

                        if synth_dialect not in expected_dialects:
                            failure_cases += 1
                            print(f"[FAIL] {feature_id} | {function_name} | Expected: {expected_dialects}, Got: {synth_dialect}")
                            self.fail(f"Generated dialect '{synth_dialect}' not in expected {expected_dialects}")

        print(f"Total failure cases: {failure_cases}")
