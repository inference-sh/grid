from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from enum import Enum
from typing import Any, List, TypeVar

T = TypeVar('T')

class ArrayElementOperator(str, Enum):
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    CONTAINS_ALL = "contains_all"
    CONTAINS_ANY = "contains_any"
    CONTAINS_NONE = "contains_none"
    COUNT_EQUAL = "count_equal"
    COUNT_GREATER = "count_greater"
    COUNT_LESS = "count_less"
    FIRST_EQUAL = "first_equal"
    LAST_EQUAL = "last_equal"
    INDEX_EQUAL = "index_equal"
    ALL_EQUAL = "all_equal"
    ANY_EQUAL = "any_equal"
    NONE_EQUAL = "none_equal"

class AppInput(BaseAppInput):
    array: List[Any] = Field(description="The array to check against")
    operator: ArrayElementOperator = Field(description="The array-element comparison operator to use")
    element: Any = Field(description="The element to compare with the array")
    index: int = Field(default=None, description="Optional index for index-based comparisons")
    true_value: Any = Field(description="The value to return if the condition is true")
    false_value: Any = Field(description="The value to return if the condition is false")

class AppOutput(BaseAppOutput):
    result: Any = Field(description="The selected value based on the array-element condition evaluation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize any resources if needed."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Evaluate the array-element condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input_data.operator == ArrayElementOperator.CONTAINS:
            condition_met = input_data.element in input_data.array
        elif input_data.operator == ArrayElementOperator.NOT_CONTAINS:
            condition_met = input_data.element not in input_data.array
        elif input_data.operator == ArrayElementOperator.CONTAINS_ALL:
            condition_met = all(x == input_data.element for x in input_data.array)
        elif input_data.operator == ArrayElementOperator.CONTAINS_ANY:
            condition_met = any(x == input_data.element for x in input_data.array)
        elif input_data.operator == ArrayElementOperator.CONTAINS_NONE:
            condition_met = not any(x == input_data.element for x in input_data.array)
        elif input_data.operator == ArrayElementOperator.COUNT_EQUAL:
            condition_met = input_data.array.count(input_data.element) == input_data.index
        elif input_data.operator == ArrayElementOperator.COUNT_GREATER:
            condition_met = input_data.array.count(input_data.element) > input_data.index
        elif input_data.operator == ArrayElementOperator.COUNT_LESS:
            condition_met = input_data.array.count(input_data.element) < input_data.index
        elif input_data.operator == ArrayElementOperator.FIRST_EQUAL:
            condition_met = input_data.array and input_data.array[0] == input_data.element
        elif input_data.operator == ArrayElementOperator.LAST_EQUAL:
            condition_met = input_data.array and input_data.array[-1] == input_data.element
        elif input_data.operator == ArrayElementOperator.INDEX_EQUAL:
            if input_data.index is None:
                raise ValueError("Index is required for index-based comparison")
            condition_met = (0 <= input_data.index < len(input_data.array) and 
                           input_data.array[input_data.index] == input_data.element)
        elif input_data.operator == ArrayElementOperator.ALL_EQUAL:
            condition_met = all(x == input_data.element for x in input_data.array)
        elif input_data.operator == ArrayElementOperator.ANY_EQUAL:
            condition_met = any(x == input_data.element for x in input_data.array)
        elif input_data.operator == ArrayElementOperator.NONE_EQUAL:
            condition_met = not any(x == input_data.element for x in input_data.array)

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input_data.true_value if condition_met else input_data.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass