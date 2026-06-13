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
