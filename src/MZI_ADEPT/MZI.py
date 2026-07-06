import sys
import os

# 获取当前脚本所在目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（LightSim 的上级目录）
_project_root = os.path.dirname(os.path.dirname(_current_dir))
# 添加 NeuroSim 搜索路径
_neurosim_dir = os.path.join(_project_root, 'NeuroSim')
sys.path.insert(0, _neurosim_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, '.')

import neurosim
import numpy as np
import torch
import yaml
from easydict import EasyDict
from metric_meter import MetricMeter


"""
graph-based PTC modeling
no thermal modeling
""" 

import networkx as nx
def _build_naive_clements_UV_coordinates(block_dim: int, size: float = 1.0) -> np.ndarray:
    """Clements interferometer phi coordinates."""
    phi_theta_gap = 400 * size
    col_gap = 400 * size
    row_gap = 70 * size

    N = block_dim
    total_mzi_num = N * (N - 1) // 2
    total_mzi_even = (N // 2) * ((N + 1) // 2)
    total_mzi_odd = total_mzi_num - total_mzi_even
    even_col_mzi_num = N // 2
    odd_col_mzi_num = (N - 1) // 2
    U_phi_pos = np.zeros((total_mzi_num, 2))
    U_theta_pos = np.zeros((total_mzi_num, 2))
    for idx in range(total_mzi_even):
        row, col = idx % even_col_mzi_num, idx // even_col_mzi_num
        x_base = 2 * col * (phi_theta_gap + col_gap)
        y_base = 2 * row * row_gap
        U_phi_pos[idx] = [x_base, y_base]
        U_theta_pos[idx] = [x_base + phi_theta_gap, y_base]
    for idx in range(total_mzi_odd):
        mzi_idx = idx + total_mzi_even
        row, col = idx % odd_col_mzi_num, idx // odd_col_mzi_num
        x_base = (2 * col + 1) * (phi_theta_gap + col_gap)
        y_base = (2 * row + 1) * row_gap
        U_phi_pos[mzi_idx] = [x_base, y_base]
        U_theta_pos[mzi_idx] = [x_base + phi_theta_gap, y_base]
    U_phi_pos = U_phi_pos[np.lexsort((U_phi_pos[:, 1], U_phi_pos[:, 0]))]
    V_phi_pos = U_phi_pos.copy()
    V_phi_pos[:, 0] += U_phi_pos[-1, 0]
    return np.vstack([U_phi_pos, V_phi_pos])

def _split_and_stack_phi_positions(phi_pos: np.ndarray, num_splits: int, vertical_gap: float = 600.0) -> np.ndarray:
    """Split phi coordinates and stack them vertically."""
    split_positions = np.array_split(phi_pos, num_splits)
    stacked_positions = []
    for i, part in enumerate(reversed(split_positions)):
        part = part.copy()
        part[:, 1] -= i * vertical_gap
        if i % 2 == 0:
            part[:, 0] = part[:, 0] - part[0, 0]
        else:
            part[:, 0] = -part[:, 0] + part[-1, 0]
            part = part[np.lexsort((-part[:, 1], -part[:, 0]))]
        stacked_positions.append(part)
    return np.vstack(stacked_positions)

def get_stacked_coordinates(block_dim=None, num_splits=1, size=1.0, vertical_gap=300.0):
        """
        Generate stacked coordinates for MZI Clements structure (for layout-aware placement).

        Parameters:
        - block_dim (int): MZI block dimension, default uses arch.num_row_pe
        - num_splits (int): number of splits, default 1
        - size (float): coordinate scaling factor, default 1.0
        - vertical_gap (float): vertical stacking gap, default 300.0

        Returns:
        - stacked_coordinates (np.ndarray): (x, y) coordinate array of shape (N, 2)
        """
        if block_dim is None:
            block_dim = int(self.conf.arch.num_row_pe)
        coordinates = _build_naive_clements_UV_coordinates(block_dim=block_dim, size=size)
        stacked_coordinates = _split_and_stack_phi_positions(
            coordinates,
            num_splits=num_splits,
            vertical_gap=vertical_gap,
        )
        return stacked_coordinates



class configuration():
    def __init__(self, config_path):
        self._load_config(config_path)
        self.settings_generation()
    
    def __getitem__(self, key):
        """Support config['key'] syntax to access config.conf"""
        return self.conf[key]
    
    def __setitem__(self, key, value):
        """Support config['key'] = value syntax to set config.conf"""
        self.conf[key] = value
    
    def __getattr__(self, name):
        """Support config.key syntax to access config.conf"""
        if hasattr(self, 'conf') and name in self.conf:
            return self.conf[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.conf = EasyDict(config)



    def settings_generation(self):
        # Model settings
        model_config = self.conf['model']
        self.modeltype = model_config['type']
        self.batchsize = model_config['batch_size']
        
        # Precision settings
        precision_config = self.conf['precision']
        self.numBitInput = precision_config['num_bit_input']
        self.numBitWeight = precision_config['num_bit_weight']
        
        # Mapping settings
        mapping_config = self.conf['mapping']
        self.weightmapping = mapping_config['weight_mapping']
        self.signmapping = mapping_config['sign_mapping']
        self.MappingSetting = mapping_config['mapping_setting']
        
        # Architecture settings
        arch_config = self.conf['arch']
        self.numRowSubArray = arch_config['num_row_sa']
        self.numColSubArray = arch_config['num_col_sa']
        self.numRowCMPE = arch_config['num_row_pe']
        self.numColCMPE = arch_config['num_col_pe']
        self.numRowCMTile = arch_config['num_row_tile']
        self.numColCMTile = arch_config['num_col_tile']
        self.numRowSystem = arch_config['num_row_sys'] 
        self.numColSystem = arch_config['num_col_sys'] 
        
        # Hardware settings
        hw_config = self.conf['hardware']
        self.clkFreq = float(hw_config['clk_freq'])
        self.temp = hw_config['temp']
        self.technode = hw_config['tech_node']
        self.numColMuxed = hw_config['num_col_muxed']
        self.levelOutput = hw_config['level_output']
        
        # Photonic settings
        photonic_config = self.conf['photonic']
        self.WeighDACNum = photonic_config['weigh_dac_num'] 
        
        # Options
        options_config = self.conf['options']
        self.printLayer2CSV = options_config['print_layer_to_csv']
        self.printNetwork2CSV = options_config['print_network_to_csv']
        self.printareaPE = options_config['print_area_pe']
        self.printareaTile = options_config['print_area_tile']
        self.printareaLayer = options_config['print_area_layer']
        
        # Enable components configuration
        if 'enable_components' in self.conf:
            self.enable_components = self.conf['enable_components']
        else:
            self.enable_components = None
        
        
        # set parameter
        input_param = neurosim.InputParameter()
        input_param.temperature =  self.temp
        input_param.transistorType = neurosim.TransistorType.conventional
        input_param.deviceRoadmap = neurosim.DeviceRoadmap.LSTP
        input_param.processNode = self.technode
        tech = neurosim.Technology()
        tech.Configure(self.technode, neurosim.DeviceRoadmap.LSTP, neurosim.TransistorType.conventional)
        

        StaticMVMCell = neurosim.MemCell()
        self.input_param = input_param
        self.tech = tech
        self.StaticMVMCell = StaticMVMCell
        self.cell = StaticMVMCell 
        self.conf.synchronous = False



class Wire():
    def __init__(self, conf):
        self.conf = conf
        self.technode = self.conf.hardware.tech_node # nm
        
        technode_to_wire_width = { # nm to nm
            130: 175,
            90:  110,
            65:  105,
            45:  80,
            32:  56,
            22:  40,
            14:  25,
            10:  18,
            7:   18,
        }
        #Initialize interconnect wires
        wiretable = {
        175: [1.60, 2.20e-8], # for technode: 130
        110: [1.60, 2.52e-8], # for technode: 90
        105: [1.70, 2.68e-8], # for technode: 65
        80:  [1.70, 3.31e-8], # for technode: 45
        56:  [1.80, 3.70e-8], # for technode: 32
        40:  [1.90, 4.03e-8], # for technode: 22
        25:  [2.00, 5.08e-8], # for technode: 14
        18:  [2.00, 6.35e-8], # for technode: 7, 10
        }

        if self.technode in technode_to_wire_width:
            self.wireWidth = technode_to_wire_width[self.technode]
        else:
            available_technodes = sorted(technode_to_wire_width.keys(), reverse=True)
            for tech in available_technodes:
                if self.technode >= tech:
                    self.wireWidth = technode_to_wire_width[tech]
                    break
            else:
                self.wireWidth = 18

        self.AR = wiretable[self.wireWidth][0]
        self.Rho = wiretable[self.wireWidth][1]
        self.Rho *= (1 + 0.00451 * abs(self.conf.hardware.temp - 300))

        if self.conf.wire.prevent_overflow:
            self.unitLengthWireResistance = 1.0 # Use a small number to prevent numerical error for NeuroSim
        else:
            self.unitLengthWireResistance = self.Rho / (self.wireWidth * 1e-9 * self.wireWidth * 1e-9 * self.AR)


class NetworkOnChip():
    def __init__(self, conf):
        self.link_model = neurosim.Bus(conf.input_param, conf.tech, conf.StaticMVMCell)
        self.conf = conf
        self.clk_freq = self.conf.hardware.clk_freq

    def Configure(self, num_noc_row, num_noc_col, grid_width, grid_len):
        """
        grid_width: the width of the grid between routers
        grid_len: the length of the grid  between routers
        """
        self.link_len = (grid_width + grid_len) / 2
        self.num_noc_row = num_noc_row
        self.num_noc_col = num_noc_col
        self.avg_hop_count = (self.num_noc_row + self.num_noc_col) / 3
        self.flit_bits = self.conf.noc.flit_width_bits
        self.busWidth = self.flit_bits
        self.area_per_router = self.conf.noc.area_router
        router_wid = np.sqrt(self.area_per_router)
        router_len = router_wid
        self.wire = Wire(self.conf)
        self.link_model.Configure(neurosim.BusMode.HORIZONTAL, 1, 2, 0, self.flit_bits, 0, self.link_len, self.conf.clkFreq,self.wire.wireWidth, self.wire.unitLengthWireResistance,self.conf.synchronous)

        
        self.cycle_time = 1 / self.clk_freq

    def CalculateArea(self):
        self.area = self.area_per_router * self.num_noc_row * self.num_noc_col

    def CalculateLatency(self, total_bits):
        num_flits = np.ceil(total_bits / self.flit_bits)
        self.link_model.CalculateArea(0, True) 
        
        latency_per_hop_s = (self.conf.noc.router_pipeline_cycles + self.conf.noc.link_delay_cycles) * self.cycle_time
        
        first_flit_latency = self.avg_hop_count * latency_per_hop_s

        last_flits_latency = (num_flits - 1) * self.cycle_time
        
        self.readLatency = first_flit_latency + last_flits_latency

    def CalculatePower(self, total_bits):
        num_flits = np.ceil(total_bits / self.flit_bits)
        E_hop_per_flit_router = self.conf.noc.power_router * self.cycle_time * self.conf.noc.router_pipeline_cycles
        self.link_model.CalculateArea(0, True) 
        self.link_model.CalculatePower(self.flit_bits, 1)
        E_hop_per_flit_link  = self.link_model.readDynamicEnergy
        self.readDynamicEnergy = (E_hop_per_flit_router + E_hop_per_flit_link) * self.avg_hop_count * num_flits



class IOinterface():
    def __init__(self, conf):
        self.area = conf.IOinterface.area
        self.power = conf.IOinterface.power
        self.bandwidth_bps = conf.IOinterface.bit_per_s
        self.num_links = conf.IOinterface.links

    def CalculateArea(self):
        return self.area

    def CalculatePerformance(self, total_bits):
        self.readLatency = total_bits / self.bandwidth_bps
        self.readDynamicEnergy = self.power * self.readLatency
        


class Network():
    def __init__(self, layer_list, config):
        self.conf = config
        self.wire = Wire(self.conf)
        self.input_param = self.conf.input_param
        self.tech = self.conf.tech
        self.StaticMVMCell = self.conf.StaticMVMCell

        self.globalBufferCore = neurosim.Buffer(self.input_param, self.tech, self.StaticMVMCell)
        self.GhTree =  neurosim.HTree(self.input_param, self.tech, self.StaticMVMCell) 
        self.GlobalNoC = NetworkOnChip(self.conf)        
        
        self.Layers = []
        self.layer_list = layer_list
        self.AdderArray =  neurosim.Adder(self.input_param, self.tech, self.StaticMVMCell) 

        self.globalunitnum = 512


    def Map(self):
        #map the netwrok to the hardware chip. configure the floorplan
        print("\nUser-defined Conventional Mapped Tile Storage Size: {}x{}".format(self.conf.numRowCMTile*self.conf.numRowCMPE*self.conf.numRowSubArray, self.conf.numColCMTile*self.conf.numColCMPE*self.conf.numColSubArray))
        print("User-defined Conventional PE Storage Size: {}x{}".format(self.conf.numRowCMPE*self.conf.numRowSubArray,self.conf.numColCMPE*self.conf.numColSubArray))
        print("User-defined SubArray Size: {}x{}".format(self.conf.numRowSubArray, self.conf.numColSubArray))

        for layer_i, layer_structure in enumerate(self.layer_list):
            if layer_structure[-1] == 'Conv' or layer_structure[-1] == 'FC':
                layer = PhotoLayer(self.input_param, self.tech, self.StaticMVMCell, self.conf)
                layer.Map(layer_structure)
                self.Layers.append(layer)
            else:
                raise ValueError("Unsupported layer type!")


        self.layer_type = []
        for l in self.layer_list:
            if  l[-1] not in self.layer_type:
                self.layer_type.append(l[-1])

    def Configure(self):
        self.TotalCMConvTiles = 0
        self.TotalConvTiles = 0


        for layer in self.Layers:


            if self.conf.MappingSetting == "fixed":
                layer.Configure()
            else:
                raise ValueError


            if self.conf.MappingSetting == "fixed":
                self.TotalCMConvTiles = layer.numTiles
            else:
                raise ValueError

        self.TotalConvTiles = self.TotalCMConvTiles 

        self.NumRowOfTile = self.conf.arch.num_row_sys
        self.NumColOfTile = self.conf.arch.num_col_sys

        num_routers = np.ceil(self.TotalConvTiles / 4) 
        num_row_router = int(np.ceil(np.sqrt(num_routers)))
        num_col_router = int(np.ceil(num_routers / num_row_router))


        if self.conf.MappingSetting == "fixed":
            self.Layers[-1].CalculateArea()
            tile_width = self.Layers[-1].PhotoTile.width
            tile_height = self.Layers[-1].PhotoTile.height
            self.GlobalNoC.Configure(num_row_router, num_col_router, tile_width * 2, tile_height * 2)
        else:
            raise ValueError
        
        self.GhTree.Configure(self.NumRowOfTile, self.NumColOfTile, 0.1, self.conf.GHtree.bitwidth, self.conf.clkFreq) 

        self.globalBufferCore.Configure(128*128, 128, 1, 1e7, self.conf.clkFreq, self.conf.arch.gobal_buffer_sram)


        self.number_of_globalBufferCore = np.ceil(self.conf.arch.global_buffer_bit/(128*128))


    def CalculateArea(self):
        enable_config = self.conf.enable_components.get('area') if self.conf.enable_components else None
        self.area_meter = MetricMeter(meter_name="Network", metric_name="area", enable_components_config=enable_config)

        self.layer_groups = []
        total_cm_height = 0
        single_cm_width = 0
        for j, layer in enumerate(self.Layers):
            layer.CalculateArea()


            if self.conf.MappingSetting == "fixed":
                    total_cm_height = layer.height
                    single_cm_width = layer.width
            else:
                raise ValueError

        self.area_meter.add_meter(layer.area_meter, val_mul=1) 


        self.Tile_H = total_cm_height / self.TotalCMConvTiles 
        self.Tile_W = single_cm_width 

        self.TileArrayHeight = self.Tile_H * self.NumRowOfTile
        self.TileArrayWidth  = self.Tile_W * self.NumColOfTile 

        # Top level global buffer
        self.globalBufferCore.CalculateArea(self.TileArrayHeight, -1, neurosim.AreaModify.NONE)
        self.area_meter.update(self.globalBufferCore.area, "global_buffer", self.number_of_globalBufferCore)

        self.GhTree.CalculateArea(self.Tile_H, self.Tile_W, 4) # TODO 4? 
        self.area_meter.update(self.GhTree.area, "network_ic", val_mul=1)




    def CalculatePerformance(self):
        enable_latency_config = self.conf.enable_components.get('latency') if self.conf.enable_components else None
        enable_energy_config = self.conf.enable_components.get('energy') if self.conf.enable_components else None
        self.latency_meter = MetricMeter(meter_name="Network", metric_name="latency", enable_components_config=enable_latency_config)
        self.energy_meter = MetricMeter(meter_name="Network", metric_name="energy", enable_components_config=enable_energy_config)

        end_tile_order = 0
        self.totalOP = 0

        for layer in self.Layers:
            buswidth_for_layer = int(self.GhTree.busWidth) 


            ################### load input feature map from global buffer
            self.num_buswidth_parallel = np.ceil(buswidth_for_layer / self.globalBufferCore.interface_width)



            numBitToLoadOut = layer.InputFeatureMapSize * layer.resend_rate * self.conf.precision.num_bit_input  # NOTE 千万不要加batch，因为里面layer.inputfeaturemapsize已经加上了batch
            

            # Buffer
            self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width)
            self.latency_meter.update(self.globalBufferCore.readLatency, "global_buffer_input", 1 / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel))
            self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width)
            self.energy_meter.update(self.globalBufferCore.readDynamicEnergy, "global_buffer_input")
            



            if self.conf.MappingSetting == "all":
                end_tile_order += layer.numTiles
                x_end = int(end_tile_order // self.NumRowOfTile) # BUG
                y_end = int(end_tile_order % self.NumRowOfTile) # BUG

            elif self.conf.MappingSetting == "fixed":
                end_tile_order = layer.numTiles
                x_end = int(end_tile_order // self.NumRowOfTile) # BUG
                y_end = int(end_tile_order % self.NumRowOfTile) # BUG

            # # GHTREE
            self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil( numBitToLoadOut / buswidth_for_layer))
            self.latency_meter.update(self.GhTree.readLatency, "network_ic_input") 

            self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile,buswidth_for_layer, 
                                         np.ceil( numBitToLoadOut/ buswidth_for_layer))
            self.energy_meter.update(self.GhTree.readDynamicEnergy, "network_ic_input")





            ############################ Layer performance
            layer.CalculatePerformance()
            outputprecision = np.minimum(32,layer.OPoutputprecision) 


            self.latency_meter.add_meter(layer.latency_meter, val_mul=1)
            self.energy_meter.add_meter(layer.energy_meter, val_mul=1)

            
            self.energy_meter.update(
                    layer.laser_power * layer.latency_meter.certain_val("MVM"), 
                    "laser", 
                    val_mul=self.conf.arch.num_col_sys * self.conf.arch.num_row_sys * self.conf.arch.num_col_tile * self.conf.arch.num_row_tile * self.conf.arch.num_col_pe * self.conf.arch.num_row_pe 
                )

            ###################### load output feature data from output buffer.
            numBitToLoadIn =  layer.OutputFeatureMapSize*outputprecision  



            # GHTRee
            self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil(numBitToLoadIn / buswidth_for_layer))
            self.latency_meter.update(self.GhTree.readLatency, "network_ic_output")
            self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                       self.TileArrayWidth / self.NumColOfTile, buswidth_for_layer,
                                       np.ceil(numBitToLoadIn/ buswidth_for_layer))
            self.energy_meter.update(self.GhTree.readDynamicEnergy, "network_ic_output")


            # Buffer
            self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width,
                                                   numBitToLoadIn / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadIn / self.globalBufferCore.interface_width)
            self.latency_meter.update(self.globalBufferCore.writeLatency, "global_buffer_output", 1 / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel))


            self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width,
                                                 numBitToLoadIn / self.globalBufferCore.interface_width,
                                                 self.globalBufferCore.interface_width,
                                                 numBitToLoadIn / self.globalBufferCore.interface_width)
            self.energy_meter.update(self.globalBufferCore.writeDynamicEnergy, "global_buffer_output")


            self.totalOP += layer.OP



        # print('-------------------- Summary --------------------')
        # print('self.totalOP',self.totalOP)
        # print('TOPS/W',float(self.totalOP)/1e12/self.readDynamicEnergy)
        # print('TOPS/W/mm^2',float(self.totalOP)/1e12/self.readDynamicEnergy/(self.area*1e6))
        # print('Throughput TOPS:',float(self.totalOP)/1e12/self.readLatency)
        # print('Compute efficiency TOPS/mm^2:',float(self.totalOP)/(1e12*self.readLatency)/(self.area*1e6))
        # print(f"FPS: {self.conf.model.batch_size / self.readLatency :.2f} FPS")
        # print(f"FPS/W: {self.conf.model.batch_size / self.readLatency / (self.readDynamicEnergy / self.readLatency) :.2f} FPS/W")
        # print('Total Latency', self.readLatency/1e-9, 'ns')
        # print('Total Energy ', (self.readDynamicEnergy)/1e-12, 'pJ')



class PhotoLayer():
    def __init__(self,input_param,tech,cell,conf):
        self.PhotoTile = None
        self.ReLu =  neurosim.BitShifter(input_param,tech,cell)
        self.debug_mode = 1
        self.conf = conf  
        self.input_param = input_param
        self.tech = tech
        self.cell = cell

    def Map(self, layer_config):
        # mapping config
        self.k1 = layer_config[0]
        self.k2 = layer_config[1]
        self.cin = layer_config[2]
        self.cout = layer_config[3]
        self.H = layer_config[4]
        self.W = layer_config[5]
        self.s1 = layer_config[6]
        self.s2 = layer_config[7]
        self.pad  =  layer_config[8]
        self.name =  layer_config[9]
        self.type =  layer_config[10]

        self.OP = 2*self.k1*self.k2*self.cin*self.cout
        self.OP = self.OP * (np.floor((self.H+2*self.pad-self.k1+1)/self.s1)*np.floor((self.W+2*self.pad-self.k2+1)/self.s2))
        self.OP = self.OP*self.conf.batchsize
        self.InputFeatureMapSize = self.k1 * self.k2 * self.cin * (np.floor((self.H+2*self.pad-self.k1+1)/self.s1)*np.floor((self.W+2*self.pad-self.k2+1)/self.s2)) * self.conf.batchsize
        self.OutputFeatureMapSize = self.cout * (np.floor((self.H+2*self.pad-self.k1)/self.s1 + 1) * np.floor((self.W+2*self.pad-self.k2)/self.s2 + 1)) * self.conf.batchsize


        self.TileDigitPerWeight = int(np.ceil(self.conf.precision.num_bit_weight / self.conf.weight_bit_each_device)) 


        self.ArrayPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / self.conf.numColSubArray), self.conf.numColCMPE) # Kernel总共需要多少个SA；如果超出PE的大小限制，则设为一个PE的最大SA数
        self.ArrayPerFanin =  np.minimum(np.ceil(self.cin * self.k1 * self.k2 / self.conf.numRowSubArray), self.conf.numRowCMPE)
        self.DupArrayRow = np.floor(self.conf.numRowCMPE/self.ArrayPerFanin) 
        self.DupArrayCol = np.floor(self.conf.numColCMPE/self.ArrayPerFanout)

        self.PEPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColSubArray * self.conf.numColCMPE)),self.conf.numColCMTile) 
        self.PEPerFanin =  np.minimum(np.ceil(self.cin * self.k1 * self.k2 / (self.conf.numRowSubArray * self.conf.numRowCMPE)),self.conf.numRowCMTile)

        self.DupPERow = np.floor(self.conf.numRowCMTile/self.PEPerFanin) 
        self.DupPECol = np.floor(self.conf.numColCMTile/self.PEPerFanout)

        self.TilePerFanoutNeed = np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColSubArray * self.conf.numColCMPE * self.conf.numColCMTile))
        self.TilePerFaninNeed  = np.ceil(self.cin * self.k1 * self.k2 / (self.conf.numRowSubArray * self.conf.numRowCMPE * self.conf.numRowCMTile))
        self.numTilesNeed = self.TilePerFanoutNeed * self.TilePerFaninNeed


        self.TileFanoutWidth = self.conf.numColSubArray * self.conf.numColCMPE * self.conf.numColCMTile 
        self.TileFaninWidth  = self.conf.numRowSubArray * self.conf.numRowCMPE * self.conf.numRowCMTile


        if self.conf.MappingSetting == "fixed":
            self.TilePerFanout = self.conf.numColSystem 
            self.TilePerFanin  = self.conf.numRowSystem 
            self.numTiles =  self.TilePerFanout * self.TilePerFanin 

            if self.numTiles < self.numTilesNeed: 
                self.DupTile = self.numTiles / self.numTilesNeed 
                self.ReuseTime = np.ceil( self.numTilesNeed / self.numTiles )
                self.Dup = self.DupArrayRow * self.DupArrayCol * self.DupPERow * self.DupPECol * self.DupTile 
            elif self.numTiles >= self.numTilesNeed: 
                self.DupTile = np.floor(self.numTiles / self.numTilesNeed) 
                self.Dup = self.DupArrayRow * self.DupArrayCol * self.DupPERow * self.DupPECol * self.DupTile 

            else:
                raise ValueError
            
        else:
            raise ValueError

        self.totaldigitonchip = self.cout * self.cin * self.k1 * self.k2 * self.Dup * self.TileDigitPerWeight 
        self.totalmemcap =  self.numTiles * self.conf.numColCMPE * self.conf.numRowCMPE * self.conf.numRowSubArray * self.conf.numColSubArray * self.conf.numRowCMTile * self.conf.numColCMTile 
        self.MemEfficiency = self.totaldigitonchip/self.totalmemcap
        self.resend_rate = 1

        self.PhotoTile = PhotoTile(self.input_param, self.tech, self.cell, self.conf)  




    def Configure(self):
        self.outputprecision = 0
        self.outputwidth = 0

        self.TileNumRows = self.conf.numRowCMTile
        self.TileNumCols = self.conf.numColCMTile
        self.PENumRows = self.conf.numRowCMPE
        self.PENumCols = self.conf.numColCMPE
        self.SubarrayRows = self.conf.numRowSubArray
        self.SubarrayCols = self.conf.numColSubArray
        self.PhotoTile.Configure()

        self.outputprecision = self.PhotoTile.outputprecision
        self.outputwidth = self.PhotoTile.outputwidth * self.numTiles
        self.outputwidth = self.outputwidth / self.TilePerFanin

        self.outputprecision = self.outputprecision


    def CalculateArea(self):
        enable_config = self.conf.enable_components.get('area') if self.conf.enable_components else None
        self.area_meter = MetricMeter(meter_name="Layer", metric_name="area", enable_components_config=enable_config)

        self.PhotoTile.CalculateArea()
        self.area_meter.add_meter(self.PhotoTile.area_meter, val_mul=self.numTiles)

        self.width = self.PhotoTile.width 
        self.height = self.PhotoTile.height * self.numTiles 


    def CalculatePerformance(self):
        enable_latency_config = self.conf.enable_components.get('latency') if self.conf.enable_components else None
        enable_energy_config = self.conf.enable_components.get('energy') if self.conf.enable_components else None
        self.latency_meter = MetricMeter(meter_name="Layer", metric_name="latency", enable_components_config=enable_latency_config)
        self.energy_meter = MetricMeter(meter_name="Layer", metric_name="energy", enable_components_config=enable_energy_config)

        self.OPoutputprecision = 0
        
        inputmatrix = torch.zeros(self.conf.batchsize, self.cin, self.H, self.W)
        weightmartix = torch.zeros(self.cout, self.cin, self.k1, self.k2)

        unfoldmap = torch.nn.Unfold((self.k1,self.k2), dilation=1, padding=self.pad, stride=self.s1)
        weightmartix = weightmartix.view(self.cout,-1)
        inputmatrix = unfoldmap(inputmatrix)

        input_section = inputmatrix[:, 0:self.TileFaninWidth, :] 
        weight_section = weightmartix[0:int(self.TileFanoutWidth/self.TileDigitPerWeight), 0:self.TileFaninWidth] 
        self.PhotoTile.CalculatePerformance(input_section, weight_section) 
        self.laser_power = self.PhotoTile.laser_power

        if self.conf.MappingSetting == "fixed" and (self.numTiles >= self.numTilesNeed):
            
            self.latency_meter.add_meter(self.PhotoTile.latency_meter, val_mul = 1 / self.DupTile)
            self.energy_meter.add_meter(self.PhotoTile.energy_meter, val_mul = self.numTilesNeed) 
            self.OPoutputprecision = self.PhotoTile.OPoutputprecision

        elif (self.conf.MappingSetting == "fixed" and (self.numTiles < self.numTilesNeed)):

            self.latency_meter.add_meter(self.PhotoTile.latency_meter, val_mul = self.ReuseTime)
            self.energy_meter.add_meter(self.PhotoTile.energy_meter, val_mul = self.numTilesNeed) 
            self.OPoutputprecision = self.PhotoTile.OPoutputprecision

        else:
            raise ValueError

        ############################

        if self.TilePerFaninNeed > 1:
            self.outputmap_H = (self.H + 2*self.pad - self.k1+1) / self.s1
            self.outputmap_W = (self.W + 2*self.pad - self.k2+1) / self.s2
            self.numInVector = self.outputmap_H * self.outputmap_W






class PhotoTile():
    def __init__(self,input_param,tech,cell,conf):
        self.conf = conf  
        self.input_param = input_param
        self.tech = tech
        self.cell = cell
        self.PhotoPE = PhotoPE(input_param,tech,cell,conf)  

        self.HTree =  neurosim.HTree(self.input_param,self.tech,self.cell)
        self.AdderTree =  neurosim.AdderTree(self.input_param,tech,self.cell)
        self.bufferInputCore =  neurosim.Buffer(self.input_param,self.tech,self.cell)
        self.bufferOutputCore = neurosim.Buffer(self.input_param,self.tech,self.cell)
        self.bufferCoreRow = 32 
        self.bufferCoreCol = 32 
        self.height = 0
        self.width = 0
        self.outputprecision = 0
        self.outputwidth = 0
        self.wire = Wire(self.conf)
        self.DigitPerWeight = int(np.ceil(self.conf.precision.num_bit_weight / self.conf.weight_bit_each_device)) 

        self.IOinterface = IOinterface(self.conf)



    def Configure(self):
        self.NumRows = self.conf.arch.num_row_tile
        self.NumCols = self.conf.arch.num_col_tile
        self.PENumRows = self.conf.arch.num_row_pe
        self.PENumCols = self.conf.arch.num_col_pe
        self.SubarrayRows = self.conf.arch.num_row_sa
        self.SubarrayCols = self.conf.arch.num_col_sa
        
        self.PhotoPE.Configure()

        self.outputprecision = self.PhotoPE.outputprecision
        self.outputwidth = int(self.PhotoPE.outputwidth * self.NumCols)

        
        self.AdderTree.Configure(self.NumRows, int(self.outputprecision), self.outputwidth, self.conf.hardware.clk_freq ) 
        self.outputprecision += int(np.log2(self.NumRows))
        
        #on Tile input buffer
        self.bufferInputCore.Configure((self.bufferCoreRow * self.bufferCoreCol), self.bufferCoreRow, 1, self.wire.unitLengthWireResistance, self.conf.hardware.clk_freq, self.conf.arch.tile_input_buffer_sram)
        self.bufferInputcoreNum = np.ceil(self.conf.arch.tile_input_buffer_bit / (self.bufferCoreRow*self.bufferCoreCol)) 


        # on Tile output buffer
        self.bufferOutputCore.Configure((self.bufferCoreRow * self.bufferCoreCol), self.bufferCoreCol, 1, self.wire.unitLengthWireResistance, self.conf.clkFreq, self.conf.arch.tile_output_buffer_sram)
        self.bufferOutputcoreNum = np.ceil( self.conf.arch.tile_output_buffer_bit / (self.bufferCoreRow * self.bufferCoreCol))  

        
        self.HTree.Configure(self.NumRows, self.NumCols, 0.1,  self.NumRows * self.SubarrayRows, self.conf.hardware.clk_freq)



    def CalculateArea(self):
        enable_config = self.conf.enable_components.get('area') if self.conf.enable_components else None
        self.area_meter = MetricMeter(meter_name="Tile", metric_name="area", enable_components_config=enable_config)

        self.PhotoPE.CalculateArea()

        self.bufferInputCore.CalculateArea(self.PhotoPE.height * self.NumRows, 0, neurosim.AreaModify.NONE) 
        self.HTree.CalculateArea(self.PhotoPE.height, self.PhotoPE.width, 16)
        self.AdderTree.CalculateArea(0, self.PhotoPE.width * self.NumCols, neurosim.AreaModify.NONE)
        self.bufferOutputCore.CalculateArea(0, self.PhotoPE.width * self.NumCols, neurosim.AreaModify.NONE)

        self.area_meter.add_meter(self.PhotoPE.area_meter, val_mul=self.NumRows * self.NumCols)
        self.area_meter.update(self.AdderTree.area, "Tile_adder")
        self.area_meter.update(self.HTree.area, "Tile_ic")
        self.area_meter.update(self.bufferInputCore.area, "Tile_buffer_input", self.bufferInputcoreNum)
        self.area_meter.update(self.bufferOutputCore.area, "Tile_buffer_output", self.bufferOutputcoreNum)

        self.height = np.sqrt(self.area_meter.val())
        self.width = self.area_meter.val() / self.height



    def CalculatePerformance(self, input, weight):
        enable_latency_config = self.conf.enable_components.get('latency') if self.conf.enable_components else None
        enable_energy_config = self.conf.enable_components.get('energy') if self.conf.enable_components else None
        self.latency_meter = MetricMeter(meter_name="Tile", metric_name="latency", enable_components_config=enable_latency_config)
        self.energy_meter = MetricMeter(meter_name="Tile", metric_name="energy", enable_components_config=enable_energy_config)


        weight = weight.transpose(0,1) 

        weight_fin = weight.shape[0]
        weight_fanout = weight.shape[1]

        input_fanin = input.shape[1]
        num_vector = input.shape[2]
        
        assert weight_fin == input_fanin, "weight_fin and input_fanin do not match"


        self.OPoutputprecision = 0
        writeLatency = 0
        writeDynamicEnergy = 0


        

        weight_row = int(np.ceil(weight.shape[0] / (self.PENumRows * self.SubarrayRows))) 
        weight_col = int(np.ceil(weight.shape[1] * self.DigitPerWeight /(self.PENumCols * self.SubarrayCols))) 

        NumPE_need = weight_row * weight_col 
        self.NumPE_need = NumPE_need

        Dup_col = 1 if int(self.NumCols/weight_col) == 0  else int(self.NumCols/weight_col) 
        Dup_row = 1 if int(self.NumRows/weight_row) == 0  else int(self.NumRows/weight_row)
        DupPEnum = Dup_row * Dup_col

        input_section = input[:, 0:(self.PENumRows*self.SubarrayRows), :]
        weight_section = weight[0:(self.PENumRows*self.SubarrayRows), 0:int((self.PENumCols*self.SubarrayCols)/self.DigitPerWeight)]

        self.PhotoPE.CalculatePerformance(input_section, weight_section)
        self.laser_power = self.PhotoPE.laser_power
        self.latency_meter.add_meter(self.PhotoPE.latency_meter, val_mul = 1 / DupPEnum)
        self.energy_meter.add_meter(self.PhotoPE.energy_meter, val_mul = NumPE_need)

                
        self.OPoutputprecision = self.PhotoPE.OPoutputprecision
        
        if weight_row > 1:
            self.AdderTree.CalculateLatency(self.conf.model.batch_size * num_vector * (self.conf.hardware.num_col_muxed/self.DigitPerWeight), weight_row, 0) # TODO didn't get it
            self.AdderTree.CalculatePower(self.conf.model.batch_size * num_vector * (self.conf.hardware.num_col_muxed/self.DigitPerWeight), weight_row) # TODO didn't get it

            self.latency_meter.update(self.AdderTree.readLatency, "Tile_adder")
            self.energy_meter.update(self.AdderTree.readDynamicEnergy, "Tile_adder")


        self.OPoutputprecision += np.ceil(np.log2(weight_row))

        ################# write. load weight from external buffer to input buffer
        numBitLoadin_write = np.ceil(weight.shape[0] * weight.shape[1]) * self.conf.precision.num_bit_weight
        self.bufferInputCore.CalculateLatency(self.bufferInputCore.interface_width,                                 
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width )),    
                                          self.bufferInputCore.interface_width,                                     
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width )))    

        self.bufferInputCore.CalculatePower(self.bufferInputCore.interface_width,
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width)),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width)))


        HTree_inputBuffer_bus_ratio = self.HTree.busWidth /self.bufferInputCore.interface_width
        HTree_outputBuffer_bus_ratio = self.HTree.busWidth /self.bufferOutputCore.interface_width
        if HTree_inputBuffer_bus_ratio >= 1:
            HTree_inputBuffer_bus_ratio = np.floor(HTree_inputBuffer_bus_ratio)
        if HTree_outputBuffer_bus_ratio >= 1:
            HTree_outputBuffer_bus_ratio = np.floor(HTree_outputBuffer_bus_ratio)

        writeLatency +=  (self.bufferInputCore.readLatency / np.minimum(self.bufferInputcoreNum, HTree_inputBuffer_bus_ratio)
                            + self.bufferInputCore.writeLatency / np.minimum(self.bufferInputcoreNum, HTree_inputBuffer_bus_ratio))
        
        
        writeDynamicEnergy +=  self.bufferInputCore.readDynamicEnergy + self.bufferInputCore.writeDynamicEnergy
        

        self.HTree.CalculateLatency(0, 0, 1, 1, self.PhotoPE.height, self.PhotoPE.width, (numBitLoadin_write)/self.HTree.busWidth )
        self.HTree.CalculatePower(0, 0, 1, 1, self.PhotoPE.height, self.PhotoPE.width, self.HTree.busWidth, (numBitLoadin_write)/self.HTree.busWidth )

        writeLatency +=  self.HTree.readLatency
        writeDynamicEnergy +=  self.HTree.readDynamicEnergy
        
        
        
        if self.conf.hardware.consider_write_movement_budget:
            self.latency_meter.update(writeLatency, "Tile_write")
            self.energy_meter.update(writeDynamicEnergy, "Tile_write")



        ############## read and write input data from external buffer to input buffer 

        numBitToLoadIn = self.conf.precision.num_bit_input * input.shape[2] * weight.shape[0] * self.conf.model.batch_size

        self.bufferInputCore.CalculateLatency(self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width )),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width )))
        self.bufferInputCore.CalculatePower(self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width)),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width)))
        

        self.latency_meter.update((self.bufferInputCore.readLatency / np.minimum(self.bufferInputcoreNum, HTree_inputBuffer_bus_ratio) 
                                   + self.bufferInputCore.writeLatency / np.minimum(self.bufferInputcoreNum, HTree_inputBuffer_bus_ratio)), 
                                   "Tile_buffer_input"
                                )

        self.energy_meter.update(self.bufferInputCore.readDynamicEnergy + self.bufferInputCore.writeDynamicEnergy, 
                                "Tile_buffer_input"
                                )


        self.HTree.CalculateLatency(0, 0, 1, 1, self.PhotoPE.height, self.PhotoPE.width,(numBitToLoadIn)/self.HTree.busWidth )
        self.HTree.CalculatePower(0, 0, 1, 1, self.PhotoPE.height, self.PhotoPE.width,self.HTree.busWidth,(numBitToLoadIn)/self.HTree.busWidth )
        
        self.latency_meter.update(self.HTree.readLatency, "Tile_ic_input")
        self.energy_meter.update(self.HTree.readDynamicEnergy, "Tile_ic_input")

            

        ##########  read and write output data from input buffer to external buffer
        numBitToLoadOut = self.OPoutputprecision * input.shape[2] * weight.shape[1] * self.conf.model.batch_size

        self.HTree.CalculateLatency(0, 0, 1, 1, self.PhotoPE.height, self.PhotoPE.width,(numBitToLoadOut)/self.HTree.busWidth )
        self.HTree.CalculatePower(0, 0, 1, 1, self.PhotoPE.height, self.PhotoPE.width,self.HTree.busWidth,(numBitToLoadOut)/self.HTree.busWidth )
        
        self.latency_meter.update(self.HTree.readLatency, "Tile_ic_output")
        self.energy_meter.update(self.HTree.readDynamicEnergy, "Tile_ic_output")

        
        self.bufferOutputCore.CalculateLatency(self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)),
                                          self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)))
        self.bufferOutputCore.CalculatePower(self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)),
                                          self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)))
        
        self.latency_meter.update((self.bufferOutputCore.readLatency/np.minimum(self.bufferOutputcoreNum, HTree_outputBuffer_bus_ratio) \
                            + self.bufferOutputCore.writeLatency/np.minimum(self.bufferOutputcoreNum, HTree_outputBuffer_bus_ratio)),
                            "Tile_buffer_output"
        )

        self.energy_meter.update(self.bufferOutputCore.writeDynamicEnergy + self.bufferOutputCore.readDynamicEnergy,
            "Tile_buffer_output"
        )




class PhotoPE():
    def __init__(self,input_param, tech, cell, conf):
        self.conf = conf  
        self.cell = cell
        self.input_param = input_param
        self.tech = tech
        self.PSA = MZI_PTC(input_param, tech, cell, conf)



        self.AdderTree = neurosim.AdderTree(input_param,tech,cell)
        self.bufferInputCore =  neurosim.DFF(input_param,tech,cell)
        self.bufferOutputCore =  neurosim.DFF(input_param,tech,cell)
        self.busInput  = neurosim.Bus(input_param,tech,cell)
        self.busOutput = neurosim.Bus(input_param,tech,cell)
         
    def Configure(self):
        self.bufferInputcoreNum = 1
        self.bufferOutputcoreNum = 1

        self.NumRows = self.conf.arch.num_row_pe 
        self.NumCols = self.conf.arch.num_col_pe 
        self.SubarrayRows = self.conf.arch.num_row_sa
        self.SubarrayCols = self.conf.arch.num_col_sa

        self.DigitPerWeight = int(np.ceil(self.conf.precision.num_bit_weight / self.conf.weight_bit_each_device)) 
        
        self.outputprecision = 0
        self.outputwidth = 0

        
        self.PSA.Configure()
        self.PSA.CalculateArea() 

        self.outputprecision += self.PSA.outputprecision
        self.outputwidth  += self.PSA.outputwidth *  self.NumCols 

    

        self.wire = Wire(self.conf)
        self.bufferInputCore.Configure(int(self.conf.arch.pe_input_buffer_bit), self.conf.clkFreq) 
        
        self.busInput.Configure(neurosim.BusMode.HORIZONTAL, self.NumRows, self.NumCols, 0, self.conf.businput.width,  self.PSA.height, self.PSA.width, self.conf.clkFreq,self.wire.wireWidth, self.wire.unitLengthWireResistance,self.conf.synchronous )

        
        self.AdderTree.Configure(self.NumRows, int(self.outputprecision), self.outputwidth, self.conf.clkFreq) 
        self.outputprecision += np.log2(self.NumRows) 

        self.busOutput.Configure(neurosim.BusMode.VERTICAL, self.NumRows, self.NumCols, 0, self.conf.busoutput.width, self.PSA.height, self.PSA.width, self.conf.clkFreq,self.wire.wireWidth, self.wire.unitLengthWireResistance,self.conf.synchronous)
        self.bufferOutputCore.Configure(int(self.conf.arch.pe_output_buffer_bit), self.conf.clkFreq) 




    def CalculateArea(self):
        enable_config = self.conf.enable_components.get('area') if self.conf.enable_components else None
        self.area_meter = MetricMeter(meter_name="PE", metric_name="area", enable_components_config=enable_config)

        self.PSA.CalculateArea()
        Arraygroup_height = self.PSA.height * self.NumRows
        Arraygroup_width = self.PSA.width * self.NumCols


        self.bufferInputCore.CalculateArea(Arraygroup_height, 0, neurosim.AreaModify.NONE)
        self.bufferOutputCore.CalculateArea(0, Arraygroup_width, neurosim.AreaModify.NONE)

        self.AdderTree.CalculateArea(0, Arraygroup_width/self.NumCols, neurosim.AreaModify.NONE) 

        self.busInput.CalculateArea(1, 1)
        self.busOutput.CalculateArea(1, 1)



        self.area_meter.update(self.busInput.area, "PE_bus_input")
        self.area_meter.update(self.busOutput.area, "PE_bus_output")

        self.area_meter.update(self.bufferInputCore.area, "PE_buffer_input", self.bufferInputcoreNum)
        self.area_meter.add_meter(self.PSA.area_meter, val_mul=self.NumRows * self.NumCols)
        self.area_meter.update(self.bufferOutputCore.area, "PE_buffer_output", self.bufferOutputcoreNum)
        self.area_meter.update(self.AdderTree.area, "PE_adder")



        self.width = np.sqrt(self.area_meter.val())
        self.height = self.area_meter.val() / self.width


    def CalculatePerformance(self, input, weight):
        enable_latency_config = self.conf.enable_components.get('latency') if self.conf.enable_components else None
        enable_energy_config = self.conf.enable_components.get('energy') if self.conf.enable_components else None
        self.latency_meter = MetricMeter(meter_name="PE", metric_name="latency", enable_components_config=enable_latency_config)
        self.energy_meter = MetricMeter(meter_name="PE", metric_name="energy", enable_components_config=enable_energy_config)

    
        trace_batch = input.shape[0] 
        num_vector = input.shape[2] 
        input_fanin = input.shape[1]
        
        weight_fanin = weight.shape[0]
        weight_fanout = weight.shape[1]
        
        assert input_fanin == weight_fanin, "fanin of input and weight do not match"
        
        
        self.numBitLoadin =  0
        self.numBitLoadout = 0
        self.OPoutputwidth = 0
        self.OPoutputprecision = 0


        self.numBitLoadin_write = 0

        writeLatency = 0
        writeDynamicEnergy = 0


        weight_row  = int(np.ceil(weight.shape[0]/self.SubarrayRows)) 
        weight_col = int(np.ceil(weight.shape[1] * self.DigitPerWeight/self.SubarrayCols)) 

        Dup_col = 1 if int(self.NumCols/weight_col) == 0  else int(self.NumCols/weight_col) 
        Dup_row = 1 if int(self.NumRows/weight_row) == 0  else int(self.NumRows/weight_row) 

        DupSubarrayNum = Dup_row * Dup_col 

        if self.conf.model.batch_size > trace_batch:
            raise ValueError("trace batchsize smaller than desired batchsize for test")


        input_vector = input[0, 0:self.SubarrayRows, 0]
        average_weight_subarray = weight[0:self.SubarrayRows, 0:int(self.SubarrayCols / self.DigitPerWeight)]
        

        self.PSA.CalculatePerformance(input_vector, average_weight_subarray)
        self.laser_power = self.PSA.laser_power

        self.latency_meter.add_meter(self.PSA.latency_meter, val_mul=self.conf.model.batch_size * num_vector / DupSubarrayNum)
        self.energy_meter.add_meter(self.PSA.energy_meter, val_mul=self.conf.model.batch_size * num_vector * (weight_col*weight_row) )

        self.latency_meter.update(self.PSA.weight_programming_latency, "weight_programming")
        self.energy_meter.update(self.PSA.weight_programming_energy, "weight_programming", val_mul=weight_col*weight_row)


        self.OPoutputprecision = self.PSA.outputprecision
        self.OPoutputwidth = self.PSA.outputwidth * weight_col


        if weight_row > 1: 
            self.AdderTree.CalculateLatency(self.conf.model.batch_size * num_vector * (weight_col * self.SubarrayCols/(self.PSA.outputwidth*weight_col)), weight_row, 0)  

            self.latency_meter.update(self.AdderTree.readLatency, "PE_adder")
            self.AdderTree.CalculatePower(self.conf.model.batch_size * num_vector * (weight_col * self.SubarrayCols/(self.PSA.outputwidth*weight_col)), weight_row)
            self.energy_meter.update(self.AdderTree.readDynamicEnergy, "PE_adder")

            self.OPoutputprecision = self.OPoutputprecision + np.log2(weight_row)



        ################ load weight from PE's input buffer
        self.numBitLoadin_write = np.ceil(weight.shape[0] * weight.shape[1]) * self.conf.precision.num_bit_weight
        
        self.bufferInputCore.CalculateLatency(0, self.numBitLoadin_write/self.bufferInputCore.numDff)
        self.bufferInputCore.CalculatePower(self.numBitLoadin_write/self.bufferInputCore.numDff, self.bufferInputCore.numDff, 0)
        
        self.busInput.CalculateLatency(self.numBitLoadin_write / self.busInput.busWidth)
        self.busInput.CalculatePower(self.busInput.busWidth, self.numBitLoadin_write / self.busInput.busWidth)

        writeLatency += self.bufferInputCore.readLatency
        writeLatency += self.busInput.readLatency
        writeDynamicEnergy += self.bufferInputCore.readDynamicEnergy
        writeDynamicEnergy += self.busInput.readDynamicEnergy
        self.writeLatency = writeLatency
        self.writeDynamicEnergy = writeDynamicEnergy

        
        if self.conf.hardware.consider_write_movement_budget:
            self.latency_meter.update(self.writeLatency, "PE_write") 
            self.energy_meter.update(self.writeDynamicEnergy, "PE_write") 


        ################ load input data from PE input buffer to subarray 
        self.numBitLoadin = np.ceil(weight.shape[0]) * self.conf.precision.num_bit_input * num_vector * self.conf.model.batch_size
        self.bufferInputCore.CalculateLatency(0, self.numBitLoadin/self.bufferInputCore.numDff)


        self.bufferInputCore.CalculatePower(self.numBitLoadin/self.bufferInputCore.numDff, self.bufferInputCore.numDff, 0)

        ################ load output data from subarray to PE outputbuffer
        self.numBitLoadout = np.ceil(weight.shape[1]) * self.OPoutputprecision * num_vector * self.conf.model.batch_size
        self.bufferOutputCore.CalculateLatency(0, self.numBitLoadout/self.bufferOutputCore.numDff)
        self.bufferOutputCore.CalculatePower(self.numBitLoadout/self.bufferOutputCore.numDff, self.bufferOutputCore.numDff, 0)
        
        self.latency_meter.update(self.bufferInputCore.readLatency, "PE_buffer_input")
        self.latency_meter.update(self.bufferOutputCore.readLatency, "PE_buffer_output")
        self.energy_meter.update(self.bufferInputCore.readDynamicEnergy, "PE_buffer_input")
        self.energy_meter.update(self.bufferOutputCore.readDynamicEnergy, "PE_buffer_output")
        
        self.busInput.CalculateLatency(self.numBitLoadin / self.busInput.busWidth)
        self.busInput.CalculatePower(self.busInput.busWidth, self.numBitLoadin / self.busInput.busWidth)

        self.busOutput.CalculateLatency(self.numBitLoadout / (self.NumRows * self.busOutput.busWidth))

        self.busOutput.CalculatePower((self.NumRows * self.busOutput.busWidth), self.numBitLoadout / (self.NumRows * self.busOutput.busWidth))

        self.latency_meter.update(self.busInput.readLatency, "PE_ic_input")
        self.latency_meter.update(self.busOutput.readLatency, "PE_ic_output")
        self.energy_meter.update(self.busInput.readDynamicEnergy, "PE_ic_input")
        self.energy_meter.update(self.busOutput.readDynamicEnergy, "PE_ic_output")





class ADC:
    def __init__(self, precision, ADC_conf):
        self.power = ADC_conf.power * (2 ** precision) / (2 ** ADC_conf.precision)
        self.area = ADC_conf.area * (precision ** 2) / (ADC_conf.precision ** 2)


class DAC:
    def __init__(self, precision, DAC_conf):
        self.power = DAC_conf.power * (2 ** precision) / (2 ** DAC_conf.precision)
        self.area = DAC_conf.area * (precision ** 2) / (DAC_conf.precision ** 2)






class MZI_PTC():
    def __init__(self, input_param, tech, cell, conf):
        self.conf = conf
        self.input_param = input_param
        self.tech = tech
        self.cell = cell
        self.height = 0
        self.width = 0
        self.outputprecision = 0
        self.outputwidth = 0


    def Configure(self):
        """配置ADEPT Photo-Core"""
        self.outputprecision = self.conf['precision']['num_bit_output']
        self.outputwidth = self.conf['arch']['num_col_sa'] 
        self.OPoutputprecision=self.conf['precision']['num_bit_output']
        self.input_DAC = DAC(precision=self.conf['precision']['num_bit_input'], DAC_conf=self.conf['inputDAC'])
        self.output_ADC = ADC(precision=self.conf['precision']['num_bit_output'], ADC_conf=self.conf['outputADC'])
        self.weight_DAC = DAC(precision=self.conf['precision']['num_bit_weight'], DAC_conf=self.conf['weightDAC'])


    def CalculateArea(self):
        enable_config = self.conf.enable_components.get('area') if self.conf.enable_components else None
        self.area_meter = MetricMeter(meter_name="SA", metric_name="area", enable_components_config=enable_config)
        
        num_weight_phase=self.conf.arch.num_row_sa*(self.conf.arch.num_row_sa-1)/2 + self.conf.arch.num_col_sa*(self.conf.arch.num_col_sa-1)/2 + min(self.conf.arch.num_col_sa, self.conf.arch.num_row_sa)
        weight_programming_latency = float(self.conf.MZI.programming_latency)
        num_DACops_perDAC = weight_programming_latency*float(self.conf.weightDAC.sampling_rate)
        num_weightDAC = np.ceil(num_weight_phase/num_DACops_perDAC) 



        if not hasattr(self, "photocore_graph") or self.photocore_graph is None:
            self.GraphConstruction()
        G = self.photocore_graph

        self.area_meter.update(G.graph["photocore_area"], "PIC", 1)
        self.area_meter.update(self.input_DAC.area, "inputDAC", self.conf.arch.num_row_sa)
        self.area_meter.update(self.output_ADC.area, "outputADC", self.conf.arch.num_col_sa)
        self.area_meter.update(self.weight_DAC.area, "weightDAC", num_weightDAC)

        self.width = np.sqrt(self.area_meter.val())
        self.height = self.area_meter.val() / self.width
        self.laser_power=G.graph["laser_power"]
 


    def CalculatePerformance(self,input, weight):
        enable_latency_config = self.conf.enable_components.get('latency') if self.conf.enable_components else None
        enable_energy_config = self.conf.enable_components.get('energy') if self.conf.enable_components else None
        self.latency_meter = MetricMeter(meter_name="SA", metric_name="latency", enable_components_config=enable_latency_config)
        self.energy_meter = MetricMeter(meter_name="SA", metric_name="energy", enable_components_config=enable_energy_config)

        
        input_fanin = input.shape[0]
        weight_fanin = weight.shape[0]
        assert weight_fanin == input_fanin, 'input vector length = input ports of PhotoCore'


        # Inference
        # latency
        cycle_time = 1/float(self.conf.hardware.clk_freq)
        inference_latency = cycle_time # 1ns when 1GHz


        # energy
        modulator_energy = float(self.conf.modulator.energy)*float(self.conf.modulator.precision)*self.conf.arch.num_row_sa
        detector_energy = float(self.conf.detector.energy)*self.conf.precision.num_bit_output*self.conf.arch.num_col_sa

        inputDAC_energy=self.input_DAC.power*cycle_time*self.conf.arch.num_row_sa
        outputADC_energy = self.output_ADC.power *cycle_time*self.conf.arch.num_col_sa




        # Weight Programming
        weight_programming_latency = float(self.conf.MZI.programming_latency)

        num_weight_phase=self.conf.arch.num_row_sa*(self.conf.arch.num_row_sa-1)/2 + self.conf.arch.num_col_sa*(self.conf.arch.num_col_sa-1)/2 + min(self.conf.arch.num_col_sa, self.conf.arch.num_row_sa)
        if self.conf.arch.num_col_sa==self.conf.arch.num_row_sa:
            assert num_weight_phase == self.conf.arch.num_row_sa*self.conf.arch.num_row_sa, "num weight phase = mxm"
        self.num_weight_phase = num_weight_phase
        num_DACops_perDAC = weight_programming_latency*float(self.conf.weightDAC.sampling_rate)
        num_weightDAC = np.ceil(num_weight_phase/num_DACops_perDAC) 

        weight_DAC_energy = self.weight_DAC.power * weight_programming_latency * num_weightDAC


        MZI_inference_energy = self.conf.MZI.power * (cycle_time) * num_weight_phase
        MZI_programming_energy = self.conf.MZI.power * (self.conf.MZI.programming_latency) * num_weight_phase



        self.energy_meter.update(inputDAC_energy, "inputDAC")
        self.energy_meter.update(outputADC_energy, "outputADC")
        self.energy_meter.update(detector_energy, "detector")
        self.energy_meter.update(modulator_energy, "modulator")
        self.energy_meter.update(weight_DAC_energy, "weightDAC")
        self.energy_meter.update(MZI_inference_energy, "MZI_inference")
        self.weight_programming_energy = MZI_programming_energy

        self.latency_meter.update(inference_latency, "MVM")
        self.weight_programming_latency = weight_programming_latency


    def BuildMZIMeshContent(self, num_splits=1, size=1.0, vertical_gap=300.0):
        """
        Generate MZIMesh content: coordinates, dimensions, and insertion loss.

        - Coordinates and dimensions: generate Clements U+V coordinates via get_stacked_coordinates,
          coordinate unit is um, then convert to m to get mzi_mesh_width and mzi_mesh_height.
        - Insertion loss (dB):

        """
        block_dim = int(self.conf.arch.num_row_sa)
        MZImesh_coordinates = get_stacked_coordinates(
            block_dim=block_dim,
            num_splits=num_splits,
            size=size,
            vertical_gap=vertical_gap,
        )
        
        _coord_to_m = 1e-6
        mzi_mesh_width_um = float(MZImesh_coordinates[:, 0].max())
        mzi_mesh_height_um = float(MZImesh_coordinates[:, 1].max())
        mzi_mesh_width = mzi_mesh_width_um * _coord_to_m
        mzi_mesh_height = mzi_mesh_height_um * _coord_to_m

        m = self.conf.arch.num_row_pe
        mzi_loss_db = float(self.conf.photonic_loss.mzi_loss_db)
        insertion_loss_db = (2 * m + 1) * mzi_loss_db

        self.MZImesh_coordinates = MZImesh_coordinates
        self.mzi_mesh_width = mzi_mesh_width
        self.mzi_mesh_height = mzi_mesh_height
        self.mzi_mesh_insertion_loss_db = insertion_loss_db
        return {
            "coordinates": MZImesh_coordinates,
            "width": mzi_mesh_width,
            "height": mzi_mesh_height,
            "insertion_loss_db": insertion_loss_db,
        }

    def GraphConstruction(self):

        G = nx.DiGraph()

        laser2coupler_d = float(self.conf.photonic_loss.laser2coupler_d)
        coupler2modulator_d = float(self.conf.photonic_loss.coupler2modulator_d)
        modulator2mzi_d = float(self.conf.photonic_loss.modulator2mzi_d)
        mzi2pd_d = float(self.conf.photonic_loss.mzi2pd_d)

        mzi_size = float(getattr(self.conf.layout, "mzi_coordinates_size", 1))
        mesh_content = self.BuildMZIMeshContent(num_splits=1, size=mzi_size, vertical_gap=300.0)
        mzi_mesh_width = mesh_content["width"]
        mzi_mesh_height = mesh_content["height"]
        insertion_loss_db = mesh_content["insertion_loss_db"]

        num_row_pe = self.conf.arch.num_row_pe
        G.add_node('Laser', type='Laser', insertion_loss_db=0.0)
        G.add_node('Coupler', type='Coupler', insertion_loss_db=float(self.conf.photonic_loss.coupler_loss_db))
        G.add_node('Modulator', type='Modulator', width=float(self.conf.modulator.width), height=float(self.conf.modulator.height), scale_height_by_num_row_pe=True, insertion_loss_db=float(self.conf.photonic_loss.modulator_loss_db))
        G.add_node('MZIMesh', type='MZIMesh', width=mzi_mesh_width, height=mzi_mesh_height, insertion_loss_db=insertion_loss_db, scale_height_by_num_row_pe=False)
        G.add_node('PD', type='PD', width=float(self.conf.detector.width), height=float(self.conf.detector.height), scale_height_by_num_row_pe=True, insertion_loss_db=-10 * np.log10(float(self.conf.photonic_loss.pd_eff)))


        G.add_edge('Laser', 'Coupler', distance=laser2coupler_d)
        G.add_edge('Coupler', 'Modulator', distance=coupler2modulator_d)
        G.add_edge('Modulator', 'MZIMesh', distance=modulator2mzi_d)
        G.add_edge('MZIMesh', 'PD', distance=mzi2pd_d)


        insertion_loss_total_db = sum(data.get("insertion_loss_db", 0) for n, data in G.nodes(data=True))
        G.graph["path_loss_db"] = insertion_loss_total_db 


        height_from_stacked = 0.0
        for n, data in G.nodes(data=True):
            if data.get("scale_height_by_num_row_pe") and "height" in data:
                h = num_row_pe * data["height"]
                height_from_stacked = max(height_from_stacked, h)
        height = max(height_from_stacked, mzi_mesh_height)


        width_from_nodes = sum(data.get("width", 0) for n, data in G.nodes(data=True) if "width" in data)
        width_from_edges = sum(data.get("distance", 0) for u, v, data in G.edges(data=True))
        width = width_from_nodes + width_from_edges

        area = height * width
        G.graph["photocore_height"] = height
        G.graph["photocore_width"] = width
        G.graph["photocore_area"] = area


        path_loss_db = G.graph["path_loss_db"]
        eta_ptc_wolaserpd = 10 ** (-path_loss_db / 10)
        q = 1.602e-19
        delta_f = 40e9
        bout = self.conf.precision.num_bit_output
        snr_overall = 2**bout
        eta_det = float(self.conf.detector.efficiency)
        eta_laser = float(self.conf.laser.efficiency)
        numerator = (snr_overall) ** 2 * (q * delta_f / 4)
        denominator = eta_det * eta_ptc_wolaserpd * eta_laser
        laser_power = numerator / denominator
        laser_power = laser_power * self.conf.arch.num_row_pe
        self.laser_power = laser_power
        G.graph["laser_power"] = laser_power
        self.photocore_graph = G
        return G





class MRR_PTC():
    def __init__(self, input_param, tech, cell, conf):
        self.conf = conf
        self.input_param = input_param
        self.tech = tech
        self.cell = cell
        self.height = 0
        self.width = 0
        self.outputprecision = 0
        self.outputwidth = 0


    def Configure(self):
        self.outputprecision = self.conf['precision']['num_bit_output']
        self.outputwidth = self.conf['arch']['num_col_sa'] 
        self.OPoutputprecision=self.conf['precision']['num_bit_output']
        self.input_DAC = DAC(precision=self.conf['precision']['num_bit_input'], DAC_conf=self.conf['inputDAC'])
        self.output_ADC = ADC(precision=self.conf['precision']['num_bit_output'], ADC_conf=self.conf['outputADC'])
        self.weight_DAC = DAC(precision=self.conf['precision']['num_bit_weight'], DAC_conf=self.conf['weightDAC'])


    def CalculateArea(self):
        enable_config = self.conf.enable_components.get('area') if self.conf.enable_components else None
        self.area_meter = MetricMeter(meter_name="SA", metric_name="area", enable_components_config=enable_config)
        
        num_weight_phase=self.conf.arch.num_row_sa*(self.conf.arch.num_row_sa-1)/2 + self.conf.arch.num_col_sa*(self.conf.arch.num_col_sa-1)/2 + min(self.conf.arch.num_col_sa, self.conf.arch.num_row_sa)
        weight_programming_latency = float(self.conf.MZI.programming_latency)
        num_DACops_perDAC = weight_programming_latency*float(self.conf.weightDAC.sampling_rate)
        num_weightDAC = np.ceil(num_weight_phase/num_DACops_perDAC) 



        self.area_meter.update(self.conf.MZI.PS_area, "MZI", num_weight_phase)
        self.area_meter.update(self.conf.modulator.area, "modulator", self.conf.arch.num_row_sa)
        self.area_meter.update(self.conf.detector.area, "PD", self.conf.arch.num_col_sa)
        self.area_meter.update(self.input_DAC.area, "inputDAC", self.conf.arch.num_row_sa)
        self.area_meter.update(self.output_ADC.area, "outputADC", self.conf.arch.num_col_sa)
        self.area_meter.update(self.weight_DAC.area, "weightDAC", num_weightDAC)

        self.width = np.sqrt(self.area_meter.val())
        self.height = self.area_meter.val() / self.width

 


    def CalculatePerformance(self,input, weight):
        enable_latency_config = self.conf.enable_components.get('latency') if self.conf.enable_components else None
        enable_energy_config = self.conf.enable_components.get('energy') if self.conf.enable_components else None
        self.latency_meter = MetricMeter(meter_name="SA", metric_name="latency", enable_components_config=enable_latency_config)
        self.energy_meter = MetricMeter(meter_name="SA", metric_name="energy", enable_components_config=enable_energy_config)

        
        input_fanin = input.shape[0]
        weight_fanin = weight.shape[0]
        assert weight_fanin == input_fanin, 'input vector length = input ports of PhotoCore'


        # Inference
        # latency
        cycle_time = 1/float(self.conf.hardware.clk_freq)
        inference_latency = cycle_time


        # energy
        modulator_energy = float(self.conf.modulator.energy)*float(self.conf.modulator.precision)*self.conf.arch.num_row_sa
        detector_energy = float(self.conf.detector.energy)*self.conf.precision.num_bit_output*self.conf.arch.num_col_sa

        inputDAC_energy=self.input_DAC.power*cycle_time*self.conf.arch.num_row_sa
        outputADC_energy = self.output_ADC.power *cycle_time*self.conf.arch.num_col_sa




        # Weight Programming
        weight_programming_latency = float(self.conf.MZI.programming_latency)

        num_weight_phase=self.conf.arch.num_row_sa*(self.conf.arch.num_row_sa-1)/2 + self.conf.arch.num_col_sa*(self.conf.arch.num_col_sa-1)/2 + min(self.conf.arch.num_col_sa, self.conf.arch.num_row_sa)
        if self.conf.arch.num_col_sa==self.conf.arch.num_row_sa:
            assert num_weight_phase == self.conf.arch.num_row_sa*self.conf.arch.num_row_sa, "num weight phase = mxm"
        self.num_weight_phase = num_weight_phase
        num_DACops_perDAC = weight_programming_latency*float(self.conf.weightDAC.sampling_rate)
        num_weightDAC = np.ceil(num_weight_phase/num_DACops_perDAC) 

        weight_DAC_energy = self.weight_DAC.power * weight_programming_latency * num_weightDAC

        MZI_inference_energy = self.conf.MZI.power * (cycle_time) * num_weight_phase
        MZI_programming_energy = self.conf.MZI.power * (self.conf.MZI.programming_latency) * num_weight_phase



        self.energy_meter.update(inputDAC_energy, "inputDAC")
        self.energy_meter.update(outputADC_energy, "outputADC")
        self.energy_meter.update(detector_energy, "detector")
        self.energy_meter.update(modulator_energy, "modulator")
        self.energy_meter.update(weight_DAC_energy, "weightDAC")
        self.energy_meter.update(MZI_inference_energy, "MZI_inference")
        self.weight_programming_energy = MZI_programming_energy

        self.latency_meter.update(inference_latency, "MVM")
        self.weight_programming_latency = weight_programming_latency













def print_PPA(model):    
    print("latency: ", model.latency_meter.val() / 1e-9, "ns")
    print("energy: ", model.energy_meter.val() / 1e-12, "pJ")
    print("area: ", model.area_meter.val() * 1e6, "mm2")    
    print('totalOP', model.totalOP)
    print('TOPS/W',float(model.totalOP)/1e12/model.energy_meter.val())
    print('TOPS/W/mm^2',float(model.totalOP)/1e12/model.energy_meter.val()/(model.area_meter.val()*1e6))
    print('Throughput TOPS:',float(model.totalOP)/1e12/model.latency_meter.val())
    print('Compute efficiency TOPS/mm^2:',float(model.totalOP)/(1e12*model.energy_meter.val())/(model.area_meter.val()*1e6))
    print(f"FPS: {model.conf.model.batch_size / model.latency_meter.val() :.2f} FPS")
    print(f"FPS/W: {model.conf.model.batch_size / model.latency_meter.val() / (model.energy_meter.val() / model.latency_meter.val()) :.2f} FPS/W")

def test_whole():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yml')
    config = configuration(config_path)

    # # SCNN
    # layer_list = [
    #     [5, 5, 1, 20, 28, 28, 1, 1, 0, 'conv1', 'Conv'],
    #     [5, 5, 20, 50, 12, 12, 1, 1, 0, 'conv2', 'Conv'],
    #     [1, 1, 800, 500, 1, 1, 1, 1, 0, 'fc1', 'FC'],
    #     [1, 1, 500, 10, 1, 1, 1, 1, 0, 'fc2', 'FC']
    # ]

    # AlexNet
    # layer_list = [
    #     [11, 11,   3,   96, 224, 224,  4,  4,  2, 'conv1',  'Conv'], # Output: 55x55. After 3x3 MaxPool(s=2) -> 27x27
    #     [5,  5,   96,  256,  27,  27,  1,  1,  2, 'conv2',  'Conv'], # Output: 27x27. After 3x3 MaxPool(s=2) -> 13x13
    #     [3,  3,  256,  384,  13,  13,  1,  1,  1, 'conv3',  'Conv'], # Output: 13x13
    #     [3,  3,  384,  384,  13,  13,  1,  1,  1, 'conv4',  'Conv'], # Output: 13x13
    #     [3,  3,  384,  256,  13,  13,  1,  1,  1, 'conv5',  'Conv'], # Output: 13x13. After 3x3 MaxPool(s=2) -> 6x6
    #     [1,  1, 9216, 4096,   1,   1,  1,  1,  0, 'fc1',    'FC'],   # Flatten: 256 * 6 * 6 = 9216
    #     [1,  1, 4096, 4096,   1,   1,  1,  1,  0, 'fc2',    'FC'],
    #     [1,  1, 4096, 1000,   1,   1,  1,  1,  0, 'fc3',    'FC']
    # ]



    # # Lenet-5
    layer_list = [
        [5, 5, 1, 20, 28, 28, 1, 1, 0, 'conv1', 'Conv'],
        [5, 5, 20, 50, 12, 12, 1, 1, 0, 'conv2', 'Conv'],
        [1, 1, 800, 500, 1, 1, 1, 1, 0, 'fc1', 'FC'],
        [1, 1, 500, 10, 1, 1, 1, 1, 0, 'fc2', 'FC']
    ]

    resnet_holylight_layer_list = [
        [7, 7, 3, 64, 224, 224, 2, 2, 3, 'conv1', 'Conv'],
        [3, 3, 64, 64, 56, 56, 1, 1, 1, 'layer1.0.conv1', 'Conv'],
        [3, 3, 64, 64, 56, 56, 1, 1, 1, 'layer1.0.conv2', 'Conv'],
        [3, 3, 64, 64, 56, 56, 1, 1, 1, 'layer1.1.conv1', 'Conv'],
        [3, 3, 64, 64, 56, 56, 1, 1, 1, 'layer1.1.conv2', 'Conv'],
        [3, 3, 64, 128, 56, 56, 2, 2, 1, 'layer2.0.conv1', 'Conv'],
        [3, 3, 128, 128, 28, 28, 1, 1, 1, 'layer2.0.conv2', 'Conv'],
        [1, 1, 64, 128, 56, 56, 2, 2, 0, 'layer2.0.downsample.0', 'Conv'],
        [3, 3, 128, 128, 28, 28, 1, 1, 1, 'layer2.1.conv1', 'Conv'],
        [3, 3, 128, 128, 28, 28, 1, 1, 1, 'layer2.1.conv2', 'Conv'],
        [3, 3, 128, 256, 28, 28, 2, 2, 1, 'layer3.0.conv1', 'Conv'],
        [3, 3, 256, 256, 14, 14, 1, 1, 1, 'layer3.0.conv2', 'Conv'],
        [1, 1, 128, 256, 28, 28, 2, 2, 0, 'layer3.0.downsample.0', 'Conv'],
        [3, 3, 256, 256, 14, 14, 1, 1, 1, 'layer3.1.conv1', 'Conv'],
        [3, 3, 256, 256, 14, 14, 1, 1, 1, 'layer3.1.conv2', 'Conv'],
        [3, 3, 256, 512, 14, 14, 2, 2, 1, 'layer4.0.conv1', 'Conv'],
        [3, 3, 512, 512, 7, 7, 1, 1, 1, 'layer4.0.conv2', 'Conv'],
        [1, 1, 256, 512, 14, 14, 2, 2, 0, 'layer4.0.downsample.0', 'Conv'],
        [3, 3, 512, 512, 7, 7, 1, 1, 1, 'layer4.1.conv1', 'Conv'],
        [3, 3, 512, 512, 7, 7, 1, 1, 1, 'layer4.1.conv2', 'Conv'],
        [1, 1, 512, 1000, 1, 1, 1, 1, 0, 'fc', 'FC']
    ]


    



    print("\n=== Network ===") 
    model = Network(layer_list, config)  
    model.Map()
    model.Configure()
    model.CalculateArea()
    model.CalculatePerformance()


    print_PPA(model)



if __name__ == "__main__":
    test_whole()



