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

    async def run(self, input: AppInput, metadata) -> AppOutput:
        """Evaluate the boolean condition and return the appropriate value."""
        condition_met = False
        
        # Evaluate the condition based on the operator
        if input.operator == BooleanOperator.EQUAL:
            condition_met = input.left_operand == input.right_operand
        elif input.operator == BooleanOperator.NOT_EQUAL:
            condition_met = input.left_operand != input.right_operand
        elif input.operator == BooleanOperator.AND:
            condition_met = input.left_operand and input.right_operand
        elif input.operator == BooleanOperator.OR:
            condition_met = input.left_operand or input.right_operand
        elif input.operator == BooleanOperator.XOR:
            condition_met = input.left_operand != input.right_operand
        elif input.operator == BooleanOperator.NAND:
            condition_met = not (input.left_operand and input.right_operand)
        elif input.operator == BooleanOperator.NOR:
            condition_met = not (input.left_operand or input.right_operand)
        elif input.operator == BooleanOperator.XNOR:
            condition_met = input.left_operand == input.right_operand

        # Return the appropriate value based on the condition
        return AppOutput(
            result=input.true_value if condition_met else input.false_value
        )

    async def unload(self):
        """Clean up resources if needed."""
        pass