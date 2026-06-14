# graph-agent-viewer

Viewer and visualization helpers for `graph-agent` graphs.

```python
from graph_agent_viewer.visualization import to_mermaid

print(to_mermaid(graph))
```

To inspect a graph in the React viewer and watch runtime events:

```python
from graph_agent_viewer import GraphView

GraphView.run(graph)
```

Use `GraphView.run(graph, step_mode=True)` to pause before each next activation
round and advance with the `Next Step` button.

For a local dependency-free stepper demo, run:

```powershell
python example\mock-react\server.py
```

To exercise the same mock graph through `GraphView` itself, install the viewer
dependencies and run:

```powershell
python example\mock-react\viewer.py
```
