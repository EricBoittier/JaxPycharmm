import time
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
import asciichartpy as acp
import numpy as np
from rich.columns import Columns

def simple_sine(s):
    return np.sin(2 * np.pi * (0.1 * s))

def get_panel(data):
    return Panel(acp.plot(data), expand=False, title="~~ [bold][yellow]waves[/bold][/yellow] ~~")


table = Table(title="PhysNetJax Training Progress")
table.add_column("Epoch")
panel = get_panel([0])
columns = Columns([table, panel])


with Live(refresh_per_second=4) as live:
    d = []
    for i in range(1000):
        time.sleep(0.1)
        d.append(simple_sine(i))
        # pass only X latest 
        panel = get_panel(d[-50:])
        columns = Columns([table, panel])
        live.update(columns)
