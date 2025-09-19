"""
Utilities for Streamlit app to handle async issues
"""
import asyncio
import os
import functools
from typing import Any, Callable

def sync_wrapper(async_func: Callable) -> Callable:
    """Wrapper to run async functions in sync context"""
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            # Try to get existing loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(async_func(*args, **kwargs))
    return wrapper

def fix_event_loop():
    """Fix event loop issues common in Streamlit"""
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    
    # Set appropriate event loop policy
    if os.name == 'nt':  # Windows
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except:
            pass
    
    # Ensure we have an event loop
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def safe_async_call(func, *args, **kwargs):
    """Safely call a function that might have async dependencies"""
    fix_event_loop()
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            # Retry with fresh event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return func(*args, **kwargs)
            finally:
                loop.close()
        else:
            raise e