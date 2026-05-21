"""
Session Test App

A simple multi-function app to verify sessions work correctly.
State persists in memory across function calls within a session.

Functions:
- set_value: Store a key-value pair in session state
- get_value: Retrieve a value by key
- increment: Increment a counter (demonstrates state mutation)
- get_all: Return all stored state
- clear: Clear all state

Usage:
1. Start session with set_value or increment
2. Call get_value/get_all to verify state persists
3. End session and verify state is gone on new session
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional


# --- Input Schemas ---

class SetValueInput(BaseAppInput):
    key: str = Field(description="Key to store")
    value: str = Field(description="Value to store")


class GetValueInput(BaseAppInput):
    key: str = Field(description="Key to retrieve")


class IncrementInput(BaseAppInput):
    key: str = Field(default="counter", description="Counter key to increment")
    amount: int = Field(default=1, description="Amount to increment by")


class GetAllInput(BaseAppInput):
    pass


class ClearInput(BaseAppInput):
    pass


# --- Output Schemas ---

class SetValueOutput(BaseAppOutput):
    success: bool = Field(description="Whether the value was stored")
    key: str = Field(description="Key that was stored")
    value: str = Field(description="Value that was stored")


class GetValueOutput(BaseAppOutput):
    found: bool = Field(description="Whether the key was found")
    key: str = Field(description="Key that was requested")
    value: Optional[str] = Field(default=None, description="Value if found")


class IncrementOutput(BaseAppOutput):
    key: str = Field(description="Counter key")
    previous: int = Field(description="Previous value")
    current: int = Field(description="Current value after increment")


class GetAllOutput(BaseAppOutput):
    state: dict = Field(description="All stored key-value pairs")
    call_count: int = Field(description="Number of calls in this session")


class ClearOutput(BaseAppOutput):
    cleared_keys: list[str] = Field(description="Keys that were cleared")


class App(BaseApp):
    """
    Session test app with in-memory state.

    State persists across function calls WITHIN a session.
    State is lost when session ends or worker restarts.
    """

    async def setup(self):
        """Initialize session state storage."""
        self.state: dict[str, str] = {}
        self.counters: dict[str, int] = {}
        self.call_count: int = 0
        print("[session-test] Setup complete, state initialized")

    async def set_value(self, input_data: SetValueInput) -> SetValueOutput:
        """Store a key-value pair in session state."""
        self.call_count += 1
        self.state[input_data.key] = input_data.value
        print(f"[session-test] set_value: {input_data.key}={input_data.value} (call #{self.call_count})")
        return SetValueOutput(
            success=True,
            key=input_data.key,
            value=input_data.value
        )

    async def get_value(self, input_data: GetValueInput) -> GetValueOutput:
        """Retrieve a value by key."""
        self.call_count += 1
        value = self.state.get(input_data.key)
        found = value is not None
        print(f"[session-test] get_value: {input_data.key}={'found: ' + value if found else 'not found'} (call #{self.call_count})")
        return GetValueOutput(
            found=found,
            key=input_data.key,
            value=value
        )

    async def increment(self, input_data: IncrementInput) -> IncrementOutput:
        """Increment a counter."""
        self.call_count += 1
        previous = self.counters.get(input_data.key, 0)
        current = previous + input_data.amount
        self.counters[input_data.key] = current
        print(f"[session-test] increment: {input_data.key} {previous} -> {current} (call #{self.call_count})")
        return IncrementOutput(
            key=input_data.key,
            previous=previous,
            current=current
        )

    async def get_all(self, input_data: GetAllInput) -> GetAllOutput:
        """Return all stored state."""
        self.call_count += 1
        all_state = {**self.state, **{f"counter:{k}": str(v) for k, v in self.counters.items()}}
        print(f"[session-test] get_all: {len(all_state)} items (call #{self.call_count})")
        return GetAllOutput(
            state=all_state,
            call_count=self.call_count
        )

    async def clear(self, input_data: ClearInput) -> ClearOutput:
        """Clear all state."""
        self.call_count += 1
        cleared = list(self.state.keys()) + [f"counter:{k}" for k in self.counters.keys()]
        self.state.clear()
        self.counters.clear()
        print(f"[session-test] clear: {len(cleared)} items cleared (call #{self.call_count})")
        return ClearOutput(cleared_keys=cleared)

    async def run(self, input_data: SetValueInput) -> SetValueOutput:
        """Default run function (same as set_value)."""
        return await self.set_value(input_data)

    async def unload(self):
        """Clean up on session end."""
        print(f"[session-test] Unload: clearing {len(self.state)} items, {self.call_count} total calls")
        self.state.clear()
        self.counters.clear()
