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

    async def run(self, input: AppInput, metadata) -> AppOutput:
        """Evaluate the array-element condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input.operator == ArrayElementOperator.CONTAINS:
            condition_met = input.element in input.array
        elif input.operator == ArrayElementOperator.NOT_CONTAINS:
            condition_met = input.element not in input.array
        elif input.operator == ArrayElementOperator.CONTAINS_ALL:
            condition_met = all(x == input.element for x in input.array)
        elif input.operator == ArrayElementOperator.CONTAINS_ANY:
            condition_met = any(x == input.element for x in input.array)
        elif input.operator == ArrayElementOperator.CONTAINS_NONE:
            condition_met = not any(x == input.element for x in input.array)
        elif input.operator == ArrayElementOperator.COUNT_EQUAL:
            condition_met = input.array.count(input.element) == input.index
        elif input.operator == ArrayElementOperator.COUNT_GREATER:
            condition_met = input.array.count(input.element) > input.index
        elif input.operator == ArrayElementOperator.COUNT_LESS:
            condition_met = input.array.count(input.element) < input.index
        elif input.operator == ArrayElementOperator.FIRST_EQUAL:
            condition_met = input.array and input.array[0] == input.element
        elif input.operator == ArrayElementOperator.LAST_EQUAL:
            condition_met = input.array and input.array[-1] == input.element
        elif input.operator == ArrayElementOperator.INDEX_EQUAL:
            if input.index is None:
                raise ValueError("Index is required for index-based comparison")
            condition_met = (0 <= input.index < len(input.array) and 
                           input.array[input.index] == input.element)
        elif input.operator == ArrayElementOperator.ALL_EQUAL:
            condition_met = all(x == input.element for x in input.array)
        elif input.operator == ArrayElementOperator.ANY_EQUAL:
            condition_met = any(x == input.element for x in input.array)
        elif input.operator == ArrayElementOperator.NONE_EQUAL:
            condition_met = not any(x == input.element for x in input.array)

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input.true_value if condition_met else input.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass