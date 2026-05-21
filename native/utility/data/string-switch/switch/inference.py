from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from enum import Enum
from typing import Any

class NumericalOperator(str, Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    LESS_THAN = "<"

class AppInput(BaseAppInput):
    left_operand: float = Field(description="The left side value of the comparison (condition value)")
    operator: NumericalOperator = Field(description="The numerical comparison operator to use")
    right_operand: float = Field(description="The right side value of the comparison (value to check against)")
    true_value: Any = Field(description="The value to return if the condition is true")
    false_value: Any = Field(description="The value to return if the condition is false")

class AppOutput(BaseAppOutput):
    result: Any = Field(description="The selected value based on the numerical condition evaluation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize any resources if needed."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Evaluate the numerical condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input_data.operator == NumericalOperator.EQUAL:
            condition_met = input_data.left_operand == input_data.right_operand
        elif input_data.operator == NumericalOperator.NOT_EQUAL:
            condition_met = input_data.left_operand != input_data.right_operand
        elif input_data.operator == NumericalOperator.GREATER_THAN:
            condition_met = input_data.left_operand > input_data.right_operand
        elif input_data.operator == NumericalOperator.GREATER_EQUAL:
            condition_met = input_data.left_operand >= input_data.right_operand
        elif input_data.operator == NumericalOperator.LESS_EQUAL:
            condition_met = input_data.left_operand <= input_data.right_operand
        elif input_data.operator == NumericalOperator.LESS_THAN:
            condition_met = input_data.left_operand < input_data.right_operand

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input_data.true_value if condition_met else input_data.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass 