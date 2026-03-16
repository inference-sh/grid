---
name: handling-cancellation
description: Handle graceful cancellation in inference.sh apps. Use when implementing long-running tasks that users might cancel.
---

# Handling Cancellation

Handle user cancellation for long-running tasks.

## The on_cancel Hook

Define an async `on_cancel` method in your `App` class. This method is called when the supervisor receives a cancellation signal.

```python
class App(BaseApp):
    async def setup(self, config):
        self.cancel_flag = False

    async def on_cancel(self):
        """Called when user cancels the task"""
        print("Cancellation requested...")
        self.cancel_flag = True

        # Return True to confirm you received the signal
        return True

    async def run(self, input_data):
        self.cancel_flag = False

        # Long running loop
        for i in range(100):
            if self.cancel_flag:
                print("Stopping work...")
                # Clean up resources if needed
                break

            await self.heavy_computation(i)
```

## Best Practices

1. **Check Frequently**: In loops, check your cancellation flag at the start of every iteration.
2. **Clean Up**: Close database connections, delete temporary files, or free GPU memory before exiting.
3. **Return Quickly**: The `on_cancel` handler should be fast. Do not block execution there; just set a flag or signal an event.
4. **Force Kill**: If an app does not respond to `on_cancel` within a timeout period (default 30s), it will be forcefully terminated (SIGKILL).
