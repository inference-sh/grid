from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from enum import Enum
from typing import Any

class StringOperator(str, Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    MATCHES_REGEX = "matches_regex"

class AppInput(BaseAppInput):
    left_operand: str = Field(description="The left side string of the comparison (condition value)")
    operator: StringOperator = Field(description="The string comparison operator to use")
    right_operand: str = Field(description="The right side string of the comparison (value to check against)")
    true_value: Any = Field(description="The value to return if the condition is true")
    false_value: Any = Field(description="The value to return if the condition is false")

class AppOutput(BaseAppOutput):
    result: Any = Field(description="The selected value based on the string condition evaluation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize any resources if needed."""
        pass

    async def run(self, input: AppInput, metadata) -> AppOutput:
        """Evaluate the string condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input.operator == StringOperator.EQUAL:
            condition_met = input.left_operand == input.right_operand
        elif input.operator == StringOperator.NOT_EQUAL:
            condition_met = input.left_operand != input.right_operand
        elif input.operator == StringOperator.CONTAINS:
            condition_met = input.right_operand in input.left_operand
        elif input.operator == StringOperator.STARTS_WITH:
            condition_met = input.left_operand.startswith(input.right_operand)
        elif input.operator == StringOperator.ENDS_WITH:
            condition_met = input.left_operand.endswith(input.right_operand)
        elif input.operator == StringOperator.IS_EMPTY:
            condition_met = not input.left_operand.strip()
        elif input.operator == StringOperator.IS_NOT_EMPTY:
            condition_met = bool(input.left_operand.strip())
        elif input.operator == StringOperator.MATCHES_REGEX:
            import re
            try:
                condition_met = bool(re.search(input.right_operand, input.left_operand))
            except re.error:
                raise ValueError(f"Invalid regular expression pattern: {input.right_operand}")

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input.true_value if condition_met else input.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass