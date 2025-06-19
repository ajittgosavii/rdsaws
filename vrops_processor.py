# vrops_processor.py
import pandas as pd
from typing import Dict, List

def create_vrops_sample_template() -> pd.DataFrame:
    """Create sample vROps template"""
    sample_data = {
        'VM_Name': ['DB-PROD-01', 'DB-PROD-02', 'DB-QA-01', 'DB-DEV-01'],
        'Environment': ['Production', 'Production', 'QA', 'Development'],
        'Max_CPU_Usage_Percent': [85.2, 78.9, 45.6, 32.1],
        'Avg_CPU_Usage_Percent': [65.4, 58.7, 28.9, 18.5],
        'CPU_Ready_Time_ms': [2500, 1800, 500, 200],
        'CPU_Cores_Allocated': [16, 12, 8, 4],
        'Max_Memory_Usage_Percent': [78.9, 82.1, 45.2, 38.7],
        'Avg_Memory_Usage_Percent': [68.5, 71.3, 35.8, 28.9],
        'Memory_Allocated_GB': [64, 48, 32, 16],
        'Memory_Balloon_GB': [0, 0.5, 0, 0],
        'Memory_Swapped_GB': [0, 0, 0, 0],
        'Max_IOPS_Total': [8500, 6200, 2100, 800],
        'Avg_IOPS_Total': [5200, 3800, 1200, 450],
        'Max_Disk_Latency_ms': [12.5, 15.8, 8.2, 6.1],
        'Avg_Disk_Latency_ms': [8.9, 11.2, 5.4, 3.8],
        'Max_Disk_Throughput_MBps': [250, 180, 85, 35],
        'Storage_Allocated_GB': [2000, 1500, 500, 200],
        'Storage_Used_GB': [1600, 1200, 350, 120],
        'Max_Network_Throughput_Mbps': [180, 125, 45, 25],
        'Database_Connections_Max': [450, 320, 125, 45],
        'Database_Size_GB': [1200, 900, 250, 80],
        'Application_Type': ['OLTP', 'OLTP', 'Mixed', 'OLTP'],
        'Peak_Hours_Start': [8, 8, 9, 9],
        'Peak_Hours_End': [18, 18, 17, 17],
        'Observation_Period_Days': [90, 90, 60, 30]
    }
    return pd.DataFrame(sample_data)

def process_vrops_data(df: pd.DataFrame, analyzer) -> Dict:
    """Process uploaded vROps data into environment specifications"""
    # Implementation from original code
    pass

def create_comprehensive_template() -> pd.DataFrame:
    """Create comprehensive template with all metrics"""
    template_data = {
        # Basic Info
        'Environment_Name': ['Production-DB1', 'Staging-DB1', 'QA-DB1', 'Dev-DB1'],
        'VM_Name': ['PROD-SQL-01', 'STAGE-SQL-01', 'QA-SQL-01', 'DEV-SQL-01'],
        'Environment_Type': ['Production', 'Staging', 'QA', 'Development'],
        
        # CPU Metrics
        'CPU_Cores_Allocated': [16, 8, 4, 2],
        # ... (other metrics from original implementation)
    }
    return pd.DataFrame(template_data)

def auto_detect_column_mappings(columns: List[str]) -> Dict[str, str]:
    """Auto-detect column mappings based on common naming patterns"""
    mappings = {}
    columns_lower = [col.lower() for col in columns]
    
    # Mapping patterns
    patterns = {
        'environment_name': ['environment', 'env', 'environment_name'],
        'vm_name': ['vm', 'server', 'host', 'vm_name', 'server_name'],
        # ... (other patterns from original implementation)
    }
    
    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower:
                    mappings[field] = columns[i]
                    break
            if field in mappings:
                break
    return mappings

def process_enhanced_data(df: pd.DataFrame, mappings: Dict[str, str]) -> Dict:
    """Process enhanced data with comprehensive mappings"""
    environments = {}
    
    for _, row in df.iterrows():
        # Get environment name
        env_name = str(row[mappings.get('environment_name', df.columns[0])])
        
        # Build comprehensive metrics dictionary
        env_metrics = {}
        
        for field, column in mappings.items():
            if column and column in df.columns:
                value = row[column]
                if pd.notna(value):
                    env_metrics[field] = float(value) if isinstance(value, (int, float)) else value
        
        # Add defaults for missing values
        defaults = {
            'observation_period_days': 30,
            'application_type': 'Mixed',
            'peak_hours_start': 9,
            'peak_hours_end': 17,
            'weekend_usage_factor': 0.3,
            'growth_rate_percent_annual': 10,
            'cpu_cores': env_metrics.get('cpu_cores_allocated', 4),
            'ram_gb': env_metrics.get('memory_allocated_gb', 16),
            'storage_gb': env_metrics.get('storage_allocated_gb', 500)
        }
        
        for key, default_value in defaults.items():
            env_metrics.setdefault(key, default_value)
        
        environments[env_name] = env_metrics
    
    return environments