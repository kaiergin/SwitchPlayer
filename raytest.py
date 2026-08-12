import ray
import asyncio

@ray.remote(max_concurrency=2)
class MyActor:
    def func_a(self, x):
        # Simulate work
        import time
        time.sleep(1)
        print("func_a")
        return f"func_a: {x}"
    
    async def func_b(self, y):
        # Simulate asynchronous work
        await asyncio.sleep(1)
        print("func_b")
        return f"func_b: {y}"

ray.init()

# Create an actor that can run up to 2 methods concurrently.
actor = MyActor.remote()

# Now you can call methods concurrently:
# Call func_a and func_b concurrently
future_b = actor.func_b.remote(20)
future_a = actor.func_a.remote(10)


# Retrieve results (both methods may run in parallel)
print(ray.get([future_a, future_b]))
