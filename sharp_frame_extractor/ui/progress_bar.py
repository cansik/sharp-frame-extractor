from rich.progress import BarColumn, Task
from rich.progress_bar import ProgressBar
from rich.style import StyleType


class StatefulBarColumn(BarColumn):
    """
    Read bar styles from task.fields so they can be changed at runtime via Progress.update(...).

    Supported task fields:
      - bar_style
      - bar_complete_style
      - bar_finished_style
      - bar_pulse_style
    """

    def render(self, task: Task) -> ProgressBar:
        style: StyleType = task.fields.get("bar_style", self.style)
        complete_style: StyleType = task.fields.get("bar_complete_style", self.complete_style)
        finished_style: StyleType = task.fields.get("bar_finished_style", self.finished_style)
        pulse_style: StyleType = task.fields.get("bar_pulse_style", self.pulse_style)

        return ProgressBar(
            total=max(0, int(task.total)) if task.total is not None else None,
            completed=max(0, int(task.completed)),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started,
            animation_time=task.get_time(),
            style=style,
            complete_style=complete_style,
            finished_style=finished_style,
            pulse_style=pulse_style,
        )
