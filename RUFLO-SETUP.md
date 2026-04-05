# RuFlo V3 Setup for CT-MUSIQ Project

RuFlo is now configured to automatically run in this project. This document explains the setup and how to manage it.

## ✅ What's Configured

### 1. **Auto-Start Daemon**
- **Setting**: `daemon.autoStart` is now `true` in `.claude/settings.json`
- **Behavior**: The daemon automatically starts when Claude Code connects to this project
- **SessionStart Hook**: Added automatic daemon startup on session initialization

### 2. **Background Workers Enabled**
The daemon runs these workers continuously:
- **map** (5 min interval) — Codebase mapping
- **audit** (10 min interval) — Security analysis
- **optimize** (15 min interval) — Performance optimization
- **consolidate** (30 min interval) — Memory consolidation
- **testgaps** (20 min interval) — Test coverage analysis

### 3. **Memory System**
- **Backend**: Hybrid (vector embeddings + temporal decay)
- **Location**: `.swarm/memory.db`
- **Features**: HNSW indexing, pattern learning, temporal decay, migration tracking

### 4. **Swarm Topology**
- **Topology**: Hierarchical-mesh
- **Max Agents**: 15
- **Auto-scaling**: Enabled
- **Protocol**: Message-bus

---

## 📋 Manual Commands

### Start/Stop the Daemon

```bash
# Start daemon manually
claude-flow daemon start

# Stop daemon
claude-flow daemon stop

# Restart daemon
claude-flow daemon restart

# Check daemon status
claude-flow daemon status

# View daemon logs
cat .claude-flow/daemon.log
```

### Manage Workers

```bash
# Enable a specific worker
claude-flow daemon enable map

# Trigger a worker immediately
claude-flow daemon trigger optimize

# View all available workers
claude-flow daemon --help
```

### Memory Operations

```bash
# Store data in memory
claude-flow memory store -k "key" --value "data"

# Search memory
claude-flow memory search -q "query"

# View memory statistics
claude-flow memory stats

# Train patterns
claude-flow neural train -p coordination
```

### Quick Start

For convenience, a batch file is provided:

```bash
# Run this to ensure daemon is started (Windows)
./start-ruflo.bat
```

---

## 🔧 Configuration

All RuFlo settings are in: `.claude/settings.json`

Key sections:
- **claudeFlow.daemon**: Auto-start and worker configuration
- **claudeFlow.memory**: Memory backend settings
- **claudeFlow.swarm**: Agent topology
- **claudeFlow.learning**: Pattern learning and neural network training

---

## 📊 Project Integration

RuFlo is now integrated into your CT-MUSIQ workflow:

1. **Code Mapping**: Automatically maps your codebase structure
2. **Security Audits**: Periodic security analysis of code changes
3. **Performance Optimization**: Suggests performance improvements
4. **Memory Management**: Consolidates knowledge from experiments
5. **Test Coverage**: Identifies gaps in test coverage

---

## 🚀 Next Steps

You can now use RuFlo to:

1. **Store experiment data**: `claude-flow memory store -k "ct-musiq-exp-1" --value "run metadata"`
2. **Query historical runs**: `claude-flow memory search -q "best validation aggregate"`
3. **Train coordination patterns**: `claude-flow neural train -p coordination`
4. **Run deep analysis**: `claude-flow daemon trigger deepdive`

---

## ⚠️ Troubleshooting

**Daemon not starting?**
```bash
# Check if Node.js is available
node --version

# Check npm installation
which claude-flow
```

**Memory database issues?**
```bash
# Reinitialize memory
claude-flow memory init --force
```

**Swarm not connecting?**
```bash
# Check swarm status
claude-flow swarm status

# Reinitialize swarm
claude-flow swarm init --force
```

---

## 📝 Notes

- RuFlo runs as a background Node.js process (daemon)
- It does not interfere with your training scripts or experiments
- Memory is persisted across sessions
- Workers respect your system's CPU/memory constraints
- Logs are available at `.claude-flow/daemon.log`

---

**Configured on**: 2026-04-05  
**Project**: CT-MUSIQ (Low-Dose CT Image Quality Assessment)  
**User**: M. Samiul Hasnat
