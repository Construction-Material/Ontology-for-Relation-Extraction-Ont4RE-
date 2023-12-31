def mapping():
  special_chars_to_entity = {
    "1台": "数量",
    "2台": "数量",
    "3台": "数量",
    "4台": "数量",
    "5台": "数量",
    "18人": "数量",
    "20人": "数量",
    "10人": "数量",
    "5人": "数量",
    "25人": "数量",
    "20辆": "数量",
    "大量": "数量",
    "km/h": "速度",
    "振幅": "振动",
    "Hz": "振动",
    "Km/h": "速度",
    "千米/小时": "速度",
    "m/s": "速度",
    "米/秒": "速度",
    "KN": "压力",
    "℃": "温度",
    "SMA-13改性": "材料类型",
    "AC-": "材料类型",
    "4cm细粒式": "材料类型",
    "6cm中粒式": "材料类型",
    "SBR改性": "材料类型",
    "SBS改性": "材料类型",
    "改性": "材料类型",
    "无溶剂型改性": "材料类型",
    "收缩性": "功能",
    "可灌性": "功能",
    "渗透性": "功能",
    "<0.15mm": "几何尺寸",
    "5～10mm": "几何尺寸",
    "1mm": "几何尺寸",
    "≥0.15mm": "几何尺寸",
    "0.1mm": "几何尺寸",
    "20-30mm": "几何尺寸",
    "2-3mm": "几何尺寸",
    "5mm":  "几何尺寸",
    "200-400mm": "几何尺寸",
    "<1.5mm": "几何尺寸",
    "≥1mm": "几何尺寸",
    "0.02mm": "几何尺寸",
    "350-500mm": "几何尺寸",
    "5-10mm": "几何尺寸",
    "80-100mm": "几何尺寸",
    "0.15mm": "几何尺寸",
    "16cm": "几何尺寸",
    "20-40cm": "几何尺寸",
    "2-3cm": "几何尺寸",
    "3.5cm": "几何尺寸",
    "30-40cm": "几何尺寸",
    "3cm-4cm": "几何尺寸",
    "4.5cm": "几何尺寸",
    "4cm": "几何尺寸",
    "5cm": "几何尺寸",
    "6cm": "几何尺寸",
    "7-8cm": "几何尺寸",
    "8cm": "几何尺寸",
    "3m": "几何尺寸",
    "7米": "几何尺寸",
    "9米": "几何尺寸",
    "厘米": "几何尺寸",
    "毫米": "几何尺寸",
    "min": "时间",
    "深度": "几何尺寸",
    "厚度": "几何尺寸",
    "深": "几何尺寸",
    "厚": "几何尺寸",
    "摊铺宽度": "几何尺寸",
    "宽度": "几何尺寸",
    "宽": "几何尺寸",
    "长度": "几何尺寸",
    "300KPA": "型号",
    "砂轮": "型号",
    "L/m2": "数量",
    "L/m3": "数量",
    "kg/m3": "数量",
    "7d": "时间",
    "2h": "时间",
    "3h": "时间",
    "7天": "时间",
    "10分钟": "时间",
    "\\d+度": "温度",
    "\\d+t":  "重量",
    "\\d+T":  "重量",
    "\\d+吨": "重量",
    "KPA": "型号",
    "\\d+型": "型号",
    "DANAPA": "型号",
    "C25": "型号",
    "快裂型": "型号",
    "轻型": "型号",
    "ABG7820": "型号",
    "智能型": "型号",
    "GQF-F-1型": "型号",
    "大型": "型号",
    "MPa": "压力",
    "气压": "压力",
    "kN/m3": "重量",
    "KN/m3": "重量",
    "30T": "重量",
    "kg": "重量"
  }

  return special_chars_to_entity
