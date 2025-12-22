import asyncio
from collections import deque
from time import monotonic
import logging


logger = logging.getLogger(__name__)

class AsyncRateLimiter:
    """
    Async-native rate limiter with sliding window.
    """
    
    def __init__(self, rpm: int, window: float = 60.0):
        self.rpm = rpm
        self.window = window
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        if self.rpm <= 0:
            return
        
        async with self._lock:
            now = monotonic()
            
            while self._timestamps and now - self._timestamps[0] >= self.window:
                self._timestamps.popleft()
            
            if len(self._timestamps) >= self.rpm:
                wait_time = self.window - (now - self._timestamps[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    now = monotonic()
                    while self._timestamps and now - self._timestamps[0] >= self.window:
                        self._timestamps.popleft()
            
            self._timestamps.append(monotonic())
    
    @property
    def current_usage(self) -> int:
        """Current number of requests in the window."""
        now = monotonic()
        while self._timestamps and now - self._timestamps[0] >= self.window:
            self._timestamps.popleft()
        return len(self._timestamps)