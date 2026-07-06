from collections import OrderedDict


class MetricMeter():
    def __init__(self, meter_name, metric_name, enable_components_config=None):
        """
        初始化MetricMeter
        
        Args:
            meter_name: meter的名字
            metric_name: metric的类型 (area, energy, latency)
            enable_components_config: 启用组件的配置字典，如果为None则所有组件都启用
        """
        self.metric_val = 0 # 量
        self.metric_type = metric_name # 这个metric的名字
        self.meter_name = meter_name # 这个meter的名字
        self.sub_meter = []
        self.metric_type_dict = OrderedDict()
        self.enable_components_config = enable_components_config  # 启用配置

    def certain_val(self, name):
        return self.metric_type_dict[name]

    def _check_component_enabled(self, metric_type):
        """
        检查组件是否启用，如果配置中未定义则报错
        
        Args:
            metric_type: 组件类型名称
            
        Returns:
            bool: 组件是否启用
            
        Raises:
            KeyError: 如果组件在配置中未定义
        """
        if self.enable_components_config is None:
            # 如果没有配置，默认所有组件启用
            return True
        
        if metric_type not in self.enable_components_config:
            raise KeyError(
                f"组件 '{metric_type}' 在 enable_components.{self.metric_type} 配置中未定义！\n"
                f"请在配置文件的 enable_components.{self.metric_type} 中添加 '{metric_type}: true/false'"
            )
        
        return self.enable_components_config[metric_type]

    def update(self, metric_val, metric_type, val_mul=1):
        """
        用来更新metric_val, 直接让自己加上val_mul*metric_val, 同时用一个字典存下来
        只有当组件在配置中启用时才会更新
        
        metric_val: 要更新的值
        val_mul: 要对这个更新的值乘以的倍数
        metric_type: 这个metric的类型, 比如metric_name是"Energy", 那么metric_type就是"IC_Energy"或"Buffer_Energy"等, 分量存在metric_type_dict里
        """
        # 检查组件是否启用
        if not self._check_component_enabled(metric_type):
            # 组件未启用，跳过更新
            return
        
        # 计算更新值
        update_val = val_mul * metric_val
        
        # 更新总指标值
        self.metric_val += update_val
        
        # 更新分类指标字典
        if metric_type in self.metric_type_dict:
            self.metric_type_dict[metric_type] += update_val
        else:
            self.metric_type_dict[metric_type] = update_val

    def val(self):
        return self.metric_val

    def add_meter(self, metric_meter, val_mul=1):
        """
        用另外的metric_meter来更新metric_val, 把metric_meter里的数据乘以val_mul叠加在self.metric_type_dict里, 并相应更新self.metric_val, 并相应更新sub_meter, 把(metric_meter, val_mul)添入self.sub_meter里
        要检查metric_meter.metric_type和self.metric_type是否一致, 不一致则报错
        metric_meter: 要继承的metric_meter
        val_mul: 要对这个更新的值乘以的倍数.
        """
        # 检查metric_type是否一致
        if metric_meter.metric_type != self.metric_type:
            raise ValueError(f"Metric type mismatch: {metric_meter.metric_type} != {self.metric_type}")
        
        # 更新总指标值
        self.metric_val += val_mul * metric_meter.metric_val
        
        # 更新分类指标字典
        for metric_type, val in metric_meter.metric_type_dict.items():
            if metric_type in self.metric_type_dict:
                self.metric_type_dict[metric_type] += val_mul * val
            else:
                self.metric_type_dict[metric_type] = val_mul * val
        
        # 添加子meter
        self.sub_meter.append((metric_meter, val_mul))

    def get_metric_breakdown(self):
        """
        获取指标分解信息
        Returns:
            dict: 包含总指标值和各分类指标值的字典
        """
        return {
            'total': self.metric_val,
            'breakdown': self.metric_type_dict.copy(),
            'sub_meters': [(meter.meter_name, mul) for meter, mul in self.sub_meter]
        }

    def reset(self):
        """
        重置所有指标
        """
        self.metric_val = 0
        self.metric_type_dict.clear()
        self.sub_meter.clear()

    def print_summary(self):
        """
        打印指标摘要
        """
        print(f"\n=== {self.meter_name} ({self.metric_type}) Summary ===")
        print(f"Total {self.metric_type}: {self.metric_val:.6e}")
        
        if self.metric_type_dict:
            print(f"\n{self.metric_type} Breakdown:")
            for metric_type, val in self.metric_type_dict.items():
                percentage = (val / self.metric_val * 100) if self.metric_val != 0 else 0
                print(f"  {metric_type}: {val:.6e} ({percentage:.4f}%)")

    def get_ordered_breakdown(self):
        """
        获取有序的指标分解信息
        Returns:
            OrderedDict: 按添加顺序排列的指标分解
        """
        return OrderedDict(self.metric_type_dict)

    def __str__(self):
        """
        字符串表示
        """
        return f"MetricMeter({self.meter_name}, {self.metric_type}): {self.metric_val:.6e}"

    def __repr__(self):
        """
        详细字符串表示
        """
        return f"MetricMeter(name='{self.meter_name}', type='{self.metric_type}', val={self.metric_val:.6e}, sub_meters={len(self.sub_meter)})"