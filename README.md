# KiCad Auto Designer

An intelligent PCB layout automation plugin for KiCad 9.0 that automatically places components using connectivity-aware algorithms, creating professional, compact layouts optimized for signal integrity and manufacturability.

## Features

- **🔌 Connectivity-Aware Placement**: Groups components by electrical function and signal flow
  - Power management subsystem
  - Wireless/RF components isolation
  - Processor and support circuitry
  - Motor control circuits
  - Interface connectors

- **📐 Professional Layout Algorithms**:
  - Square-aspect-ratio shelf packing for efficient space utilization
  - 2D local-search compaction for optimal density
  - Force-directed relaxation to eliminate overlaps
  - Intelligent edge placement for connectors and antennas

- **🎯 Smart Component Recognition**:
  - Automatic connector detection (USB, UART, JTAG, power jacks)
  - Antenna component identification
  - Reference designator pattern matching
  - Component grouping by electrical relationships

- **⚡ Optimization Features**:
  - Minimizes trace lengths between connected components
  - Maintains proper clearances and spacing
  - Automatically creates board edge cuts
  - Centers layout on board for symmetry

## Installation

### Method 1: Direct Installation

1. Locate your KiCad scripting plugins directory:
   - **macOS**: `~/Documents/KiCad/9.0/scripting/plugins/`
   - **Linux**: `~/.config/kicad/9.0/scripting/plugins/`
   - **Windows**: `%APPDATA%\kicad\9.0\scripting\plugins\`

2. Copy `kicad_auto_designer.py` to the plugins directory

3. Restart KiCad PCB Editor or refresh plugins

### Method 2: Symbolic Link (Development)

```bash
# macOS/Linux
ln -s /path/to/kicad_auto_designer.py ~/Documents/KiCad/9.0/scripting/plugins/

# Windows (Command Prompt as Administrator)
mklink "%APPDATA%\kicad\9.0\scripting\plugins\kicad_auto_designer.py" "C:\path\to\kicad_auto_designer.py"
```

## Usage

1. Open your PCB file in KiCad PCB Editor
2. Navigate to **Tools → External Plugins → KiCad Auto Designer**
3. Click **Run** to execute automatic layout
4. The plugin will:
   - Group components by connectivity and function
   - Pack them into a compact rectangular layout
   - Place connectors and antennas at board edges
   - Create edge cuts boundary
   - Center the entire layout

### Before Running

- Ensure all components are imported from schematic
- Set approximate board size (can be adjusted after)
- Save your work (plugin creates a backup state)

### After Running

- Review component placement
- Fine-tune specific positions if needed
- Run DRC (Design Rules Check) to verify clearances
- Proceed with routing

## How It Works

### Layout Algorithm

1. **Component Categorization**:

   ```
   Power Management → Wireless → Processor → Motor Control → Interfaces → Passives
   ```

2. **Packing Strategy**:
   - Calculates target width for square-aspect layout
   - Uses Next-Fit Decreasing Height (NFDH) shelf packing
   - Tests component rotations for better fit
   - Compacts rows and columns

3. **Optimization Passes**:
   - **2D Compaction**: 80 iterations of local moves and swaps
   - **Force-Directed Relaxation**: 20 iterations with 0.05mm minimum spacing
   - **Overlap Resolution**: Greedy pairwise collision detection

4. **Edge Placement**:
   - Antenna → Bottom-left corner (isolated)
   - Power connectors → Bottom edge (0° orientation)
   - Data connectors → Right edge (90° orientation)
   - 4mm board margin, 2mm component spacing

5. **Finalization**:
   - Center layout on board
   - Create rectangular edge cuts with 2mm margin
   - Refresh display

## Configuration

Edit these parameters in `kicad_auto_designer.py`:

```python
# Spacing and margins
min_spacing_nm = int(0.01 * 1_000_000)  # 0.01mm minimum spacing
margin_nm = int(1 * 1_000_000)          # 1mm board margin

# Compaction iterations
max_iterations = 80                      # 2D compaction passes

# Relaxation parameters
relax_iterations = 20                    # Force-directed iterations
min_spacing_relax = 0.05                 # 0.05mm spacing target

# Edge placement
board_margin_mm = 4.0                    # 4mm from edge
edge_spacing_mm = 2.0                    # 2mm between edge components
```

## Component Recognition Patterns

### Connectors

- USB: `USB`, `J_USB`
- UART: `UART`, `SERIAL`, `TX`, `RX`
- JTAG: `JTAG`, `SWD`, `DEBUG`
- Power: `BARREL`, `JACK`, `PWR`, `VIN`
- GPIO: `GPIO`, `HEADER`, `PIN`

### Special Components

- Antenna: `ANTENNA`, `ANT`, `U2`
- Wireless: `U1`, `ESP`, `BLE`, `WIFI`
- Processor: `U4`, `MCU`, `CPU`
- Motor: `DRV`, `MOTOR`, `U3`

## Requirements

- **KiCad Version**: 9.0 or later
- **Python**: 3.8+ (bundled with KiCad)
- **Dependencies**:
  - `pcbnew` (KiCad Python API)
  - `wx` (GUI dialogs)
  - Standard library: `math`, `random`

## Known Limitations

- Does not consider existing traces (place components before routing)
- Manual DRC verification recommended after placement
- Complex boards (>100 components) may require multiple runs
- Net-aware optimization is connectivity-based, not electrical rules-aware

## Roadmap

- [ ] Net-length optimization for matched pairs
- [ ] Layer-aware component placement
- [ ] Thermal via placement for power components
- [ ] Interactive placement with user hints
- [ ] Undo/redo support
- [ ] Batch processing for multiple boards
- [ ] Custom component grouping rules

## Contributing

Contributions are welcome! Areas for improvement:

1. **Algorithm Enhancements**: Better packing, net-aware placement
2. **Component Recognition**: More patterns, ML-based classification
3. **Optimization**: Performance improvements for large boards
4. **Testing**: Test cases for various board types
5. **Documentation**: Usage examples, video tutorials

## License

MIT License - feel free to use, modify, and distribute.

## Author

Created for professional PCB design automation. For issues, feature requests, or contributions, please use the GitHub issue tracker.

## Acknowledgments

- KiCad development team for the excellent Python API
- Shelf packing algorithms from computational geometry literature
- Force-directed layout concepts from graph theory

---

**⚡ Happy Auto-Designing!** 🚀
