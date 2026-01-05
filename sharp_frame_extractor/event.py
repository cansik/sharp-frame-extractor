import threading
from typing import TypeVar, Generic, Callable, List, Optional, Iterator

T = TypeVar('T')
H = Callable[[T], None]


class Event(Generic[T]):
    """
    A generic event class that allows you to register and trigger event handlers,
    and also provides a way to wait for the next event to be fired.

    Attributes:
        _handlers (List[H]): A list to store event handlers.
    """

    def __init__(self):
        """
        Initialize the Event instance with an empty list of handlers
        and a threading event to allow waiting for events.
        """
        self._handlers: List[H] = []
        self._latest_value: Optional[T] = None
        self._event_trigger = threading.Event()

    def append(self, handler: H) -> None:
        """
        Append an event handler to the list of handlers.

        Args:
            handler (H): The event handler function to add.
        """
        self._handlers.append(handler)

    def remove(self, handler: H) -> None:
        """
        Remove an event handler from the list of handlers.

        Args:
            handler (H): The event handler function to remove.
        """
        self._handlers.remove(handler)

    def contains(self, handler: H) -> bool:
        """
        Check if a specific event handler is already registered.

        Args:
            handler (H): The event handler function to check for.

        Returns:
            bool: True if the handler is in the list, False otherwise.
        """
        return handler in self._handlers

    def invoke(self, value: T) -> None:
        """
        Invoke all registered event handlers with the provided value.
        Also set the threading event to allow waiting mechanisms to proceed.

        Args:
            value (T): The value to pass to the event handlers.
        """
        self._latest_value = value
        for handler in self._handlers:
            handler(value)

        # Trigger the event for waiting threads
        self._event_trigger.set()

    def invoke_latest(self, value: T) -> None:
        """
        Invoke the most recently added event handler with the provided value.

        If no event handlers are registered, this method does nothing.

        Args:
            value (T): The value to pass to the latest event handler.
        """
        if len(self._handlers) == 0:
            return
        self._handlers[-1](value)

    def clear(self) -> None:
        """
        Clear all registered event handlers, removing them from the list.
        """
        self._handlers.clear()

    def register(self, handler: H) -> H:
        """
        Append an event handler to the list of handlers and return it.
        This method should be used as decorator.

        Args:
            handler (H): The event handler function to add.
        Returns:
            H: Returns the handler given as argument.
        """
        self.append(handler)
        return handler

    @property
    def handler_size(self) -> int:
        """
        Get the number of registered event handlers.

        Returns:
            int: The number of event handlers currently registered.
        """
        return len(self._handlers)

    def __iadd__(self, other):
        """
        Allow the use of '+=' to add an event handler.

        Args:
            other (H): The event handler function to add.

        Returns:
            Event[T]: The updated Event instance.
        """
        self.append(other)
        return self

    def __isub__(self, other):
        """
        Allow the use of '-=' to remove an event handler.

        Args:
            other (H): The event handler function to remove.

        Returns:
            Event[T]: The updated Event instance.
        """
        self.remove(other)
        return self

    def __contains__(self, item) -> bool:
        """
        Check if a specific event handler is already registered using 'in' operator.

        Args:
            item (H): The event handler function to check for.

        Returns:
            bool: True if the handler is in the list, False otherwise.
        """
        return self.contains(item)

    def __call__(self, value: T):
        """
        Allow the instance to be called as a function, invoking all event handlers.

        Args:
            value (T): The value to pass to the event handlers.
        """
        self.invoke(value)

    def wait(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Wait for the next event to be fired, with an optional timeout.

        Args:
            timeout (Optional[float]): The maximum time (in seconds) to wait.
                                        If None, wait indefinitely.

        Returns:
            Optional[T]: The value passed when the event was triggered,
                         or None if the timeout was reached.
        """
        event_occurred = self._event_trigger.wait(timeout)

        # If the event occurred, clear the event and return the latest value
        if event_occurred:
            self._event_trigger.clear()
            return self._latest_value
        else:
            # Return None if the timeout is reached
            return None

    def stream(self, timeout: Optional[float] = None) -> Iterator[Optional[T]]:
        """
        Continuously yield the value whenever the event is triggered, with an optional timeout.

        Args:
            timeout (Optional[float]): The maximum time (in seconds) to wait
                                       between yielding values. If None, wait indefinitely.

        Yields:
            Optional[T]: The value passed each time the event is triggered,
                         or None if the timeout was reached.
        """
        while True:
            yield self.wait(timeout)

    def __getstate__(self):
        """
        Custom method to remove the _event_trigger from the state when pickling.
        """
        state = self.__dict__.copy()
        state['_event_trigger'] = None  # Exclude the event trigger from pickling
        return state

    def __setstate__(self, state):
        """
        Custom method to restore the _event_trigger after unpickling.
        """
        self.__dict__.update(state)
        self._event_trigger = threading.Event()  # Reinitialize the event
