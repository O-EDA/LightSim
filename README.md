# LightSim — A Flexible Photonic Computing System Modeling Demo
LightSim is a flexible, fast modeling framework for holistic photonic computing systems.
This demo showcases MZI-based photonic computing with hardware performance modeling. 
Expanded features, including results profiling and more, are forthcoming.

This demo is built upon the [MICSim](https://github.com/MICSim-official/MICSim_V1.0) framework. MICSim is a unified simulation framework designed to enable rapid design space exploration for heterogeneous computing systems.

## Project Structure

```
LightSim/
├── README.md                        
├── requirements.txt  
├── environment.yml                 
├── NeuroSim/                        # NeuroSim v1.3 backend
└── src/
    └── MZI_ADEPT/
        ├── MZI.py                   # main simulation script
        ├── config.yml               # configuration
        └── metric_meter.py          # performance calculation tool
```

### Key Files

| File | Description |
|------|-------------|
| `MZI.py` | Main simulation script. Defines the hierarchical structure from **PTC** → **PE** → **Tile** → **Layer** → **Network**, and performs hardware performance calculations based on `config.yml` |
| `config.yml` | YAML configuration file containing device parameters (MZI, modulator, detector, DAC/ADC, etc.), architecture parameters (PTC/PE/Tile/System dimensions), precision settings, and optical loss parameters. **All simulation parameters can be modified here** |
| `metric_meter.py` | Performance metric collection utility. Supports hierarchical statistics and component-level enable/disable control via the `enable_components` configuration |

## Quick Start

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/O-EDA/LightSim.git
cd Lightsim
```


### 2. Environment Setup
You can set up the environment using either **Conda (Recommended)** or **Pip**.

#### Option A: Using `environment.yml` (Fastest for Conda users)
This method creates the environment and installs all dependencies in one go:
```bash
conda env create -f environment.yml
conda activate lightsim
```

#### Option B: Using `requirements.txt` (Manual setup)
Choose this option if you prefer manual environment management:
```bash
# Create and activate a new environment
conda create -n lightsim python=3.8 -y
conda activate lightsim

# Install dependencies
pip install -r requirements.txt
```

### 3. Compile the NeuroSim v1.3 Backend

LightSim depends on NeuroSim as the underlying C++ backend for electronic device/circuit modeling.

> **Reference**: [MICSim V1.0 — 3. Compile the NeuroSim v1.3 Backend](https://github.com/MICSim-official/MICSim_V1.0#3-compile-the-neurosim-v13-backend)


### 4. Run Simulation

```bash
cd LightSim/src/MZI_ADEPT
python MZI.py
```

## Configuration

All simulation parameters are configured in `config.yml`. Key sections include:

### Device Parameters
- **MZI**: Phase shifter area, programming latency, power
- **modulator**: Modulator energy, area, dimensions
- **detector**: Detector energy, efficiency, area
- **ADC/DAC**: Precision, sampling rate, power, area
- **laser**: Laser efficiency, peak power

### Architecture Parameters
- **PTC**: `num_row_sa`, `num_col_sa` (default 16×16)
- **PE**: `num_row_pe`, `num_col_pe` (default 2×2)
- **Tile**: `num_row_tile`, `num_col_tile` (default 1×1)
- **System**: `num_row_sys`, `num_col_sys` (default 2×2)

### Precision Settings
- `num_bit_input`: Input data bit-width 
- `num_bit_weight`: Weight bit-width 
- `num_bit_output`: Output data bit-width

### Component Enable Control
The `enable_components` section enables granular control over which components contribute to area/energy/latency calculations, which is useful for debugging or selectively masking components.

## Running Different Networks
The default simulation uses LeNet-5. To switch to other networks (AlexNet, ResNet-18, etc.), modify the `layer_list` in the `test_whole()` function of `MZI.py`:

## Sample Output

```
=== Network ===
latency:  0.0499 ns
energy:   471983.8246 pJ
area:     0.0030 mm²
TOPS/W:   0.0109
FPS:      20032.44 FPS
```

