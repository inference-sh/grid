---
name: handling-cancellation
description: Handle graceful cancellation in inference.sh apps. Use when implementing long-running tasks that users might cancel.
---

# Handling Cancellation

Handle user cancellation for long-running tasks.

## The on_cancel Hook

```python
class App(BaseApp):
    async def setup(self, config):
        self.cancel_flag = False

    async def on_cancel(self):
        """Called when user cancels - must return quickly"""
        self.cancel_flag = True
        return True

    async def run(self, input):
        self.cancel_flag = False
        
        for i in range(100):
            if self.cancel_flag:
                print("Stopping work...")
                break
            await self.heavy_computation(i)
```

## Tips

- Check flag at start of every loop iteration
- `on_cancel` must be fast (just set flag)
- Clean up resources before exiting
- Force kill after 30s timeout if no response

📖 **Full docs**: [inference.sh/docs/extend/cancellation](https://inference.sh/docs/extend/cancellation)
