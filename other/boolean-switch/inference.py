from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from enum import Enum
from typing import Any

class BooleanOperator(str, Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    AND = "and"
    OR = "or"
    XOR = "xor"
    NAND = "nand"
    NOR = "nor"
    XNOR = "xnor"

class AppInput(BaseAppInput):
    left_operand: bool = Field(description="The left side boolean of the comparison (condition value)")
    operator: BooleanOperator = Field(description="The boolean comparison operator to use")
    right_operand: bool = Field(description="The right side boolean of the comparison (value to check against)")
    true_value: Any = Field(description="The value to return if the condition is true")
    false_value: Any = Field(description="The value to return if the condition is false")

class AppOutput(BaseAppOutput):
    result: Any = Field(description="The selected value based on the boolean condition evaluation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize any resources if needed."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Evaluate the boolean condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input_data.operator == BooleanOperator.EQUAL:
            condition_met = input_data.left_operand == input_data.right_operand
        elif input_data.operator == BooleanOperator.NOT_EQUAL:
            condition_met = input_data.left_operand != input_data.right_operand
        elif input_data.operator == BooleanOperator.AND:
            condition_met = input_data.left_operand and input_data.right_operand
        elif input_data.operator == BooleanOperator.OR:
            condition_met = input_data.left_operand or input_data.right_operand
        elif input_data.operator == BooleanOperator.XOR:
            condition_met = input_data.left_operand != input_data.right_operand
        elif input_data.operator == BooleanOperator.NAND:
            condition_met = not (input_data.left_operand and input_data.right_operand)
        elif input_data.operator == BooleanOperator.NOR:
            condition_met = not (input_data.left_operand or input_data.right_operand)
        elif input_data.operator == BooleanOperator.XNOR:
            condition_met = input_data.left_operand == input_data.right_operand

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input_data.true_value if condition_met else input_data.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass