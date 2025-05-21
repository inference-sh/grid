from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from enum import Enum
from typing import Any, List, Union, TypeVar

T = TypeVar('T')

class ArrayOperator(str, Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    SUBSET = "subset"
    SUPERSET = "superset"
    INTERSECTS = "intersects"
    DISJOINT = "disjoint"
    LENGTH_EQUAL = "length_equal"
    LENGTH_GREATER = "length_greater"
    LENGTH_LESS = "length_less"
    EMPTY = "empty"
    NOT_EMPTY = "not_empty"

class AppInput(BaseAppInput):
    left_operand: List[Any] = Field(description="The left side array of the comparison (condition value)")
    operator: ArrayOperator = Field(description="The array comparison operator to use")
    right_operand: Union[List[Any], Any] = Field(description="The right side array or value of the comparison (value to check against)")
    true_value: Any = Field(description="The value to return if the condition is true")
    false_value: Any = Field(description="The value to return if the condition is false")

class AppOutput(BaseAppOutput):
    result: Any = Field(description="The selected value based on the array condition evaluation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize any resources if needed."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Evaluate the array condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input_data.operator == ArrayOperator.EQUAL:
            condition_met = input_data.left_operand == input_data.right_operand
        elif input_data.operator == ArrayOperator.NOT_EQUAL:
            condition_met = input_data.left_operand != input_data.right_operand
        elif input_data.operator == ArrayOperator.CONTAINS:
            condition_met = input_data.right_operand in input_data.left_operand
        elif input_data.operator == ArrayOperator.NOT_CONTAINS:
            condition_met = input_data.right_operand not in input_data.left_operand
        elif input_data.operator == ArrayOperator.SUBSET:
            condition_met = all(item in input_data.left_operand for item in input_data.right_operand)
        elif input_data.operator == ArrayOperator.SUPERSET:
            condition_met = all(item in input_data.right_operand for item in input_data.left_operand)
        elif input_data.operator == ArrayOperator.INTERSECTS:
            condition_met = any(item in input_data.right_operand for item in input_data.left_operand)
        elif input_data.operator == ArrayOperator.DISJOINT:
            condition_met = not any(item in input_data.right_operand for item in input_data.left_operand)
        elif input_data.operator == ArrayOperator.LENGTH_EQUAL:
            condition_met = len(input_data.left_operand) == len(input_data.right_operand)
        elif input_data.operator == ArrayOperator.LENGTH_GREATER:
            condition_met = len(input_data.left_operand) > len(input_data.right_operand)
        elif input_data.operator == ArrayOperator.LENGTH_LESS:
            condition_met = len(input_data.left_operand) < len(input_data.right_operand)
        elif input_data.operator == ArrayOperator.EMPTY:
            condition_met = len(input_data.left_operand) == 0
        elif input_data.operator == ArrayOperator.NOT_EMPTY:
            condition_met = len(input_data.left_operand) > 0

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input_data.true_value if condition_met else input_data.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass