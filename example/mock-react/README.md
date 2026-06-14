# Mock React Viewer

This example runs a local mock ReAct-style graph for debugging the React viewer
step controls.

For the fastest dependency-free UI mock, run:

```powershell
python example\mock-react\server.py
```

It serves the real viewer static React app and mock `/api/*` endpoints from the
standard library.

To run the same linear mock graph through `GraphView` itself, install the viewer
dependencies and run:

```powershell
python example\mock-react\viewer.py
```

Both paths use one activation per step. Each click on `Next Step` releases one
node, runs it, and then waits for the following click.
