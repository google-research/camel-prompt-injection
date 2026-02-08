# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from typing import ClassVar

from camel.capabilities import Capabilities
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.security_policy import NoSecurityPolicyEngine


def pytest_generate_tests(metafunc):
    # We are not using pytest.parametrize directly to make it easier to move
    # to GoogleTest later if needed.
    idlist = []
    argvalues = []
    argnames = []
    for scenario in metafunc.cls.test_data:
        idlist.append(scenario["testcase_name"])
        test_params = {k: v for k, v in scenario.items() if k != "testcase_name"}
        argnames = list(test_params.keys())
        argvalues.append(list(test_params.values()))
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


_a_0 = value.CaMeLInt(0, Capabilities.default(), ())
_a_1 = value.CaMeLInt(1, Capabilities.default(), ())
_a_2 = value.CaMeLInt(2, Capabilities.default(), ())
_a_list = value.CaMeLList([_a_0, _a_1, _a_2], Capabilities.default(), ())
_a_true = value.CaMeLTrue(Capabilities.default(), ())
_a_false = value.CaMeLFalse(Capabilities.default(), ())
_a_none = value.CaMeLNone(Capabilities.default(), ())


class TestIf:
    test_data: ClassVar = [
        dict(
            testcase_name="if_true",
            code="""\
if True:
  a = 1
else:
  a = 0
""",
            expected=_a_none,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_1.new_with_dependencies((_a_true,))}),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
        dict(
            testcase_name="if_false",
            code="""\
if False:
  a = 1
else:
  a = 0
""",
            expected=_a_none,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace({"a": _a_0.new_with_dependencies((_a_false,))}),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
        dict(
            testcase_name="if_false_no_else",
            code="""\
if False:
  a = 1
""",
            expected=_a_none,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
        dict(
            testcase_name="if_dependency_strict",
            code="""\
if a:
  b = 1
""",
            expected=_a_none,
            namespace=ns.Namespace({"a": _a_true}),
            expected_namespace=ns.Namespace(
                {
                    "a": _a_true,
                    "b": _a_1.new_with_dependencies((_a_true,)),
                }
            ),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
        dict(
            testcase_name="if_dependency_not_strict",
            code="""\
if a:
  b = 1
""",
            expected=_a_none,
            namespace=ns.Namespace({"a": _a_true}),
            expected_namespace=ns.Namespace({"a": _a_true, "b": _a_1}),
            eval_mode=interpreter.MetadataEvalMode.NORMAL,
        ),
    ]

    def test_if(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
        eval_mode: interpreter.MetadataEvalMode,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), eval_mode),
        )
        assert isinstance(got_code, result.Ok)
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace


class TestFor:
    test_data: ClassVar = [
        dict(
            testcase_name="for",
            code="""\
for i in [0, 1, 2]:
  a = i
""",
            expected=_a_none,
            namespace=ns.Namespace(),
            expected_namespace=ns.Namespace(
                {
                    # TODO: Unclear why the double dependency. Leaving for now as it does not change anything in practice
                    "a": _a_2.new_with_dependencies((_a_list,)),
                    "i": _a_2.new_with_dependencies((_a_list,)),
                }
            ),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
        dict(
            testcase_name="not_iterated_for",
            code="""\
for i in []:
  a = i
""",
            expected=_a_none,
            namespace=ns.Namespace({"a": _a_1}),
            expected_namespace=ns.Namespace({"a": _a_1}),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
        dict(
            testcase_name="for_dependency_non_strict",
            code="""\
for i in l:
  b = 5
  a = i
""",
            expected=_a_none,
            namespace=ns.Namespace({"l": _a_list}),
            expected_namespace=ns.Namespace(
                {
                    "l": _a_list,
                    "a": _a_2,
                    "i": _a_2,
                    "b": value.CaMeLInt(5, Capabilities.default(), ()),
                }
            ),
            eval_mode=interpreter.MetadataEvalMode.NORMAL,
        ),
        dict(
            testcase_name="for_dependency_strict",
            code="""\
for i in l:
  b = 5
  a = i
""",
            expected=_a_none,
            namespace=ns.Namespace({"l": _a_list}),
            expected_namespace=ns.Namespace(
                {
                    "l": _a_list,
                    # TODO: Unclear why the double dependency. Leaving for now as it does not change anything in practice
                    "a": _a_2.new_with_dependencies((_a_list, _a_list)),
                    "i": _a_2.new_with_dependencies((_a_list,)),
                    "b": value.CaMeLInt(5, Capabilities.default(), ()).new_with_dependencies((_a_list,)),
                }
            ),
            eval_mode=interpreter.MetadataEvalMode.STRICT,
        ),
    ]

    def test_for(
        self,
        code: str,
        expected: value.CaMeLValue,
        namespace: ns.Namespace,
        expected_namespace: ns.Namespace | None,
        eval_mode: interpreter.MetadataEvalMode,
    ):
        got_code, got_namespace, _, _ = interpreter.camel_eval(
            ast.parse(code),
            namespace,
            [],
            [],
            interpreter.EvalArgs(NoSecurityPolicyEngine(), eval_mode),
        )
        assert isinstance(got_code, result.Ok)
        assert expected == got_code.value
        if expected_namespace is not None:
            assert got_namespace == expected_namespace
        else:
            assert namespace == got_namespace
