    const h = React.createElement;
    const {useEffect, useMemo, useState} = React;

    function textOf(value) {
      if (value == null) return "";
      if (typeof value === "string") return value;
      if (value.text) return value.text;
      return JSON.stringify(value);
    }

    function statusByNode(graph, events) {
      const status = Object.fromEntries((graph.nodes || []).map((node) => [node.id, "pending"]));
      for (const event of events) {
        const node = event.payload && event.payload.node;
        if (!node) continue;
        if (event.name === "node_started") status[node] = "running";
        if (event.name === "node_finished") status[node] = "finished";
        if (event.name === "viewer_error") status[node] = "error";
      }
      return status;
    }

    function layoutGraph(graph) {
      const nodes = graph.nodes || [];
      const edges = graph.edges || [];
      const byId = Object.fromEntries(nodes.map((node) => [node.id, node]));
      const levels = {};
      const queue = [];
      if (graph.start_node && byId[graph.start_node]) {
        levels[graph.start_node] = 0;
        queue.push(graph.start_node);
      }
      for (let index = 0; index < queue.length; index += 1) {
        const sourceId = queue[index];
        for (const edge of edges.filter((item) => item.source === sourceId)) {
          if (!byId[edge.target] || levels[edge.target] != null) {
            continue;
          }
          levels[edge.target] = levels[sourceId] + 1;
          queue.push(edge.target);
        }
      }
      for (const node of nodes) {
        if (levels[node.id] == null) levels[node.id] = 0;
      }
      const grouped = {};
      for (const node of nodes) {
        const level = levels[node.id];
        grouped[level] = grouped[level] || [];
        grouped[level].push(node);
      }
      const positions = {};
      const columnWidth = 198;
      const rowHeight = 64;
      for (const [levelText, group] of Object.entries(grouped)) {
        const level = Number(levelText);
        group.sort((a, b) => a.id.localeCompare(b.id));
        group.forEach((node, index) => {
          positions[node.id] = {
            x: 48 + level * columnWidth,
            y: 58 + index * rowHeight,
            width: 68,
            height: 28,
          };
        });
      }
      const maxLevel = Math.max(0, ...Object.values(levels));
      const maxRows = Math.max(1, ...Object.values(grouped).map((group) => group.length));
      return {
        nodes,
        edges: edges.filter((edge) => byId[edge.source] && byId[edge.target]),
        positions,
        width: 96 + (maxLevel + 1) * columnWidth,
        height: 96 + maxRows * rowHeight,
      };
    }

    function GraphDiagram({graph, title, edges, statuses, selectedEdgeId, onEdgeClick, emptyText}) {
      const layout = useMemo(() => layoutGraph(graph), [graph]);
      const visibleEdges = edges || layout.edges;
      const [manualPositions, setManualPositions] = useState({});
      const [drag, setDrag] = useState(null);
      const positions = useMemo(() => {
        const merged = {};
        for (const [nodeId, pos] of Object.entries(layout.positions)) {
          const manual = manualPositions[nodeId];
          merged[nodeId] = manual ? {...pos, x: manual.x, y: manual.y} : pos;
        }
        return merged;
      }, [layout.positions, manualPositions]);
      function svgPointFor(svg, event) {
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(svg.getScreenCTM().inverse());
      }
      function onNodePointerDown(event, nodeId) {
        const pos = positions[nodeId];
        if (!pos) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.pointerId != null && event.currentTarget.setPointerCapture) {
          event.currentTarget.setPointerCapture(event.pointerId);
        }
        const svg = event.currentTarget.ownerSVGElement;
        const point = svgPointFor(svg, event);
        setDrag({
          nodeId,
          pointerId: event.pointerId == null ? "mouse" : event.pointerId,
          svg,
          offsetX: point.x - pos.x,
          offsetY: point.y - pos.y,
        });
      }
      function updateDraggedNode(event, currentDrag) {
        const pointerId = event.pointerId == null ? "mouse" : event.pointerId;
        if (!currentDrag || currentDrag.pointerId !== pointerId) return;
        const point = svgPointFor(currentDrag.svg, event);
        const base = layout.positions[currentDrag.nodeId];
        if (!base) return;
        const nextX = Math.max(8, Math.min(layout.width - base.width - 8, point.x - currentDrag.offsetX));
        const nextY = Math.max(8, Math.min(layout.height - base.height - 8, point.y - currentDrag.offsetY));
        setManualPositions((current) => ({
          ...current,
          [currentDrag.nodeId]: {x: nextX, y: nextY},
        }));
      }
      function onNodePointerMove(event) {
        updateDraggedNode(event, drag);
      }
      function onNodePointerUp(event) {
        const pointerId = event.pointerId == null ? "mouse" : event.pointerId;
        if (drag && drag.pointerId === pointerId) {
          setDrag(null);
        }
      }
      useEffect(() => {
        if (!drag) return undefined;
        function move(event) {
          event.preventDefault();
          updateDraggedNode(event, drag);
        }
        function stop() {
          setDrag(null);
        }
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", stop);
        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", stop);
        return () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", stop);
          window.removeEventListener("mousemove", move);
          window.removeEventListener("mouseup", stop);
        };
      }, [drag, layout.positions, layout.width, layout.height]);
      return h("div", {className: "panel"},
        h("div", {className: "panel-title"}, h("span", null, title), h("span", null, graph.start_node || "")),
        h("div", {className: "graph-canvas"},
          h("svg", {
            viewBox: `0 0 ${layout.width} ${layout.height}`,
            width: layout.width,
            height: layout.height,
            role: "img",
            onPointerMove: onNodePointerMove,
            onPointerUp: onNodePointerUp,
            onPointerCancel: onNodePointerUp,
            onMouseMove: onNodePointerMove,
            onMouseUp: onNodePointerUp,
            onMouseLeave: onNodePointerUp,
          },
            h("defs", null,
              h("marker", {id: "arrow", markerWidth: "10", markerHeight: "10", refX: "8", refY: "3", orient: "auto", markerUnits: "strokeWidth"},
                h("path", {d: "M0,0 L0,6 L9,3 z", fill: "#94a3b8"})
              )
            ),
            visibleEdges.length === 0 ? h("text", {x: 24, y: 32, className: "edge-label"}, emptyText || "") : null,
            visibleEdges.map((edge) => {
              const source = positions[edge.source];
              const target = positions[edge.target];
              if (!source || !target) return null;
              const x1 = source.x + source.width;
              const y1 = source.y + source.height / 2;
              const x2 = target.x;
              const y2 = target.y + target.height / 2;
              const midX = (x1 + x2) / 2;
              const midY = (y1 + y2) / 2;
              const selected = selectedEdgeId === edge.id;
              const clickable = Boolean(onEdgeClick);
              const label = edge.name;
              const labelGap = Math.max(46, Math.abs(x2 - x1) - 16);
              const labelWidth = Math.min(160, labelGap, Math.max(46, label.length * 7 + 18));
              const labelX = midX - labelWidth / 2;
              const labelY = Math.max(10, midY - 26);
              return h("g", {
                key: edge.id,
                className: `${clickable ? "edge-clickable" : ""} ${selected ? "edge-selected" : ""}`,
                onClick: clickable ? () => onEdgeClick(edge) : undefined,
              },
                clickable ? h("path", {className: "edge-hit", d: `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`}) : null,
                h("path", {className: "edge-path", markerEnd: "url(#arrow)", d: `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`}),
                h("rect", {className: "edge-label-bg", x: labelX, y: labelY, width: labelWidth, height: 20}),
                h("text", {className: "edge-label", x: midX, y: labelY + 14, textAnchor: "middle"}, label)
              );
            }),
            layout.nodes.map((node) => {
              const pos = positions[node.id];
              const status = (statuses && statuses[node.id]) || "pending";
              return h("g", {
                key: node.id,
                className: `node ${status} ${drag && drag.nodeId === node.id ? "dragging" : ""}`,
                transform: `translate(${pos.x},${pos.y})`,
                onPointerDown: (event) => onNodePointerDown(event, node.id),
                onMouseDown: (event) => onNodePointerDown(event, node.id),
              },
                h("rect", {width: pos.width, height: pos.height}),
                h("text", {className: "label", x: 6, y: 12}, node.label),
                h("text", {className: "kind", x: 6, y: 22}, `${node.kind} / ${status}`),
                node.is_start ? h("text", {className: "badge", x: pos.width - 24, y: 11}, "start") : null
              );
            })
          )
        )
      );
    }

    function StaticGraph({graph}) {
      return h(GraphDiagram, {
        graph,
        title: "Static Graph",
        edges: graph.edges || [],
        statuses: {},
      });
    }

    function DynamicGraph({graph, events, selectedEdgeId, onEdgeClick}) {
      const statuses = useMemo(() => statusByNode(graph, events), [graph, events]);
      const edges = useMemo(() => activationEdges(events), [events]);
      return h(GraphDiagram, {
        graph,
        title: "Runtime Graph",
        edges,
        statuses,
        selectedEdgeId,
        onEdgeClick,
        emptyText: "Waiting for activations",
      });
    }

    function activationEdges(events) {
      const activations = [];
      events.forEach((event, eventIndex) => {
        if (event.name !== "node_activated") return;
        const payload = event.payload || {};
        (payload.edges || []).forEach((edge, edgeIndex) => {
          activations.push({
            id: `${eventIndex}-${edgeIndex}-${edge.name}`,
            name: `${edge.name} #${activations.length + 1}`,
            edge_name: edge.name,
            source: edge.source,
            target: edge.target,
            history: payload.history || [],
            upstream_outputs: payload.upstream_outputs || {},
            node: payload.node,
            event_index: eventIndex,
          });
        });
      });
      return activations;
    }

    function Timeline({events}) {
      return h("div", {className: "panel"},
        h("div", {className: "panel-title"}, h("span", null, "Runtime"), h("span", null, `${events.length} events`)),
        h("div", {className: "timeline"},
          events.length === 0 ? h("div", {className: "empty"}, "Waiting") : events.map((event, index) =>
            h("div", {className: "event-row", key: `${event.name}-${index}`},
              h("div", {className: "event-name"}, event.name),
              h("div", {className: "event-body"},
                event.payload.node ? h("div", {className: "kv"}, h("span", null, "node"), h("div", {className: "value"}, event.payload.node)) : null,
                event.payload.graph ? h("div", {className: "kv"}, h("span", null, "graph"), h("div", {className: "value"}, event.payload.graph)) : null,
                event.payload.history ? h("div", {className: "kv"}, h("span", null, "history"), h("div", {className: "value"}, event.payload.history.map(textOf).join("\n"))) : null,
                event.payload.output ? h("div", {className: "kv"}, h("span", null, "output"), h("div", {className: "value"}, textOf(event.payload.output))) : null,
                event.payload.message ? h("div", {className: "kv"}, h("span", null, "message"), h("div", {className: "value"}, event.payload.message)) : null
              )
            )
          )
        )
      );
    }

    function Structure({graph}) {
      return h("div", {className: "panel"},
        h("div", {className: "panel-title"}, h("span", null, "Structure"), h("span", null, graph.name || "")),
        h("div", {className: "structure"},
          h("div", null,
            h("div", {className: "section-title"}, "Nodes"),
            h("div", {className: "list"}, (graph.nodes || []).map((node) =>
              h("div", {className: "list-item", key: node.id},
                h("span", null, node.label),
                h("span", {className: "chip"}, node.kind)
              )
            ))
          ),
          h("div", null,
            h("div", {className: "section-title"}, "Edges"),
            h("div", {className: "list"}, (graph.edges || []).map((edge) =>
              h("div", {className: "list-item", key: edge.id},
                h("span", null, `${edge.source} -> ${edge.target}`),
                h("span", {className: "chip"}, edge.name)
              )
            ))
          )
        )
      );
    }

    function Outputs({events}) {
      const finished = events.filter((event) => event.name === "node_finished");
      return h("div", {className: "panel"},
        h("div", {className: "panel-title"}, h("span", null, "Outputs"), h("span", null, `${finished.length}`)),
        h("div", {className: "output"},
          finished.length === 0 ? h("div", {className: "empty"}, "No output") : finished.map((event, index) =>
            h(OutputItem, {event, index, key: `${event.payload.node}-${index}`})
          )
        )
      );
    }

    function ActivationDetails({edge}) {
      return h("div", {className: "panel"},
        h("div", {className: "panel-title"}, h("span", null, "Activation"), h("span", null, edge ? `#${edge.event_index + 1}` : "")),
        h("div", {className: "details"},
          !edge ? h("div", {className: "empty"}, "Select a runtime edge") : h("div", null,
            h("div", {className: "output-meta"},
              h("span", {className: "chip"}, edge.edge_name),
              h("span", {className: "chip"}, `${edge.source} -> ${edge.target}`)
            ),
            h("div", {className: "details-section"},
              h("div", {className: "section-title"}, "History"),
              h(MessageList, {messages: edge.history})
            ),
            h("div", {className: "details-section"},
              h("div", {className: "section-title"}, "Upstream Output"),
              h(MessageList, {messages: edge.upstream_outputs && edge.upstream_outputs[edge.edge_name] ? [edge.upstream_outputs[edge.edge_name]] : []})
            )
          )
        )
      );
    }

    function MessageList({messages}) {
      return h("div", {className: "message-list"},
        !messages || messages.length === 0 ? h("div", {className: "empty"}, "Empty") : messages.map((message, index) =>
          h(MessageCard, {message, index, key: index})
        )
      );
    }

    function MessageCard({message, index}) {
      const blocks = message.blocks || [];
      return h("div", {className: "message-card"},
        h("div", {className: "output-header"},
          h("div", {className: "output-node"}, `message #${index + 1}`),
          h("span", {className: "chip"}, message.role || "message")
        ),
        message.text ? h("div", {className: "output-text"}, message.text) : null,
        h("div", {className: "output-blocks"},
          blocks.map((block, blockIndex) => h(MessageBlock, {block, blockIndex, key: blockIndex}))
        )
      );
    }

    function OutputItem({event, index}) {
      const output = event.payload.output || {};
      const blocks = output.blocks || [];
      return h("div", {className: "output-item"},
        h("div", {className: "output-header"},
          h("div", {className: "output-node"}, event.payload.node || "node"),
          h("span", {className: "chip"}, `#${index + 1}`)
        ),
        h("div", {className: "output-meta"},
          h("span", {className: "chip"}, output.role || "message"),
          blocks.length ? h("span", {className: "chip"}, `${blocks.length} block${blocks.length === 1 ? "" : "s"}`) : null
        ),
        output.text ? h("div", {className: "output-text"}, output.text) : null,
        h("div", {className: "output-blocks"},
          blocks.map((block, blockIndex) => h(MessageBlock, {block, blockIndex, key: blockIndex}))
        )
      );
    }

    function MessageBlock({block, blockIndex}) {
      const rows = blockRows(block);
      return h("div", {className: "block-detail"},
        h("div", {className: "output-meta"},
          h("span", {className: "chip"}, block.kind || "block"),
          h("span", {className: "chip"}, `block ${blockIndex + 1}`)
        ),
        rows.map(([label, value]) =>
          h("div", {className: "block-row", key: label},
            h("span", null, label),
            h("div", {className: "mono"}, value)
          )
        )
      );
    }

    function blockRows(block) {
      if (!block) return [];
      if (block.kind === "text" || block.kind === "reasoning") {
        return [["text", block.text || ""]];
      }
      if (block.kind === "tool_call") {
        return [
          ["call_id", block.call_id || ""],
          ["tool", block.tool_name || ""],
          ["args", JSON.stringify(block.arguments || {}, null, 2)],
        ];
      }
      if (block.kind === "tool_result") {
        return [
          ["call_id", block.call_id || ""],
          ["tool", block.tool_name || ""],
          ["error", String(Boolean(block.is_error))],
          ["content", block.content || ""],
        ];
      }
      if (block.kind === "file") {
        return [
          ["name", block.name || ""],
          ["path", block.path || ""],
          ["mime", block.mime_type || ""],
          ["file_id", block.file_id || ""],
        ].filter(([, value]) => value);
      }
      return Object.entries(block)
        .filter(([key]) => key !== "kind")
        .map(([key, value]) => [key, typeof value === "string" ? value : JSON.stringify(value, null, 2)]);
    }

    function App() {
      const [graph, setGraph] = useState(null);
      const [events, setEvents] = useState([]);
      const [error, setError] = useState(null);
      const [selectedEdge, setSelectedEdge] = useState(null);
      useEffect(() => {
        fetch("/api/graph")
          .then((response) => {
            if (!response.ok) throw new Error(`Graph request failed: ${response.status}`);
            return response.json();
          })
          .then(setGraph)
          .catch((error) => setError(error.message));
        const source = new EventSource("/api/events");
        source.addEventListener("graph-event", (event) => {
          setEvents((current) => current.concat(JSON.parse(event.data)));
        });
        source.onerror = () => {
          setError((current) => current || "Runtime event stream disconnected");
        };
        return () => source.close();
      }, []);
      const stats = useMemo(() => {
        if (!graph) return {nodes: 0, edges: 0, running: 0, finished: 0};
        const statuses = statusByNode(graph, events);
        return {
          nodes: graph.nodes.length,
          edges: graph.edges.length,
          running: Object.values(statuses).filter((status) => status === "running").length,
          finished: Object.values(statuses).filter((status) => status === "finished").length,
        };
      }, [graph, events]);
      if (error) return h("div", {className: "empty"}, error);
      if (!graph) return h("div", {className: "empty"}, "Loading");
      return h("div", {className: "shell"},
        h("header", null,
          h("div", null, h("h1", null, graph.name), h("div", {className: "subtitle"}, `start: ${graph.start_node || "none"}`)),
          h("div", {className: "stats"},
            h("div", {className: "stat"}, h("strong", null, stats.nodes), "nodes"),
            h("div", {className: "stat"}, h("strong", null, stats.edges), "edges"),
            h("div", {className: "stat"}, h("strong", null, stats.running), "running"),
            h("div", {className: "stat"}, h("strong", null, stats.finished), "finished")
          )
        ),
        h("main", null,
          h("section", {className: "workspace"},
            h(StaticGraph, {graph}),
            h(DynamicGraph, {
              graph,
              events,
              selectedEdgeId: selectedEdge && selectedEdge.id,
              onEdgeClick: setSelectedEdge,
            })
          ),
          h("aside", null,
            h(ActivationDetails, {edge: selectedEdge}),
            h(Outputs, {events}),
            h(Timeline, {events})
          )
        )
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(h(App));
