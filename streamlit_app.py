import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import requests
import json
import boto3
from datetime import datetime, timedelta
import base64
import io
import zipfile
from typing import Dict, List, Tuple, Any, Optional
import asyncio
import hashlib
import tempfile
import os

# PDF Generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Try to import optional dependencies
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Enterprise AWS Database Migration Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# ===========================
# CSS STYLING
# ===========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .enterprise-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .enterprise-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .config-section {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }
    
    .environment-card {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #38b2ac;
        margin: 1rem 0;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    .risk-high { border-left-color: #e53e3e !important; }
    .risk-medium { border-left-color: #d69e2e !important; }
    .risk-low { border-left-color: #38a169 !important; }
    
    .strategy-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# CORE CLASSES
# ===========================

class DatabaseEngine:
    """Database engine definitions and compatibility matrix"""
    
    ENGINES = {
        'oracle-ee': {
            'name': 'Oracle Enterprise Edition',
            'features': ['Advanced Analytics', 'Partitioning', 'Advanced Security', 'RAC'],
            'aws_targets': ['oracle-ee', 'postgres', 'aurora-postgresql'],
            'complexity_multiplier': 1.0
        },
        'oracle-se': {
            'name': 'Oracle Standard Edition',
            'features': ['Basic Analytics', 'Standard Security'],
            'aws_targets': ['oracle-se', 'postgres', 'aurora-postgresql'],
            'complexity_multiplier': 0.8
        },
        'postgres': {
            'name': 'PostgreSQL',
            'features': ['JSON Support', 'Advanced Indexing', 'Extensions'],
            'aws_targets': ['postgres', 'aurora-postgresql'],
            'complexity_multiplier': 0.5
        },
        'mysql': {
            'name': 'MySQL',
            'features': ['InnoDB', 'MyISAM', 'Replication'],
            'aws_targets': ['mysql', 'aurora-mysql'],
            'complexity_multiplier': 0.6
        },
        'sql-server': {
            'name': 'Microsoft SQL Server',
            'features': ['T-SQL', 'SSRS', 'SSIS', 'Analysis Services'],
            'aws_targets': ['sql-server', 'postgres', 'aurora-postgresql'],
            'complexity_multiplier': 1.2
        },
        'mariadb': {
            'name': 'MariaDB',
            'features': ['MySQL Compatibility', 'Columnar Storage'],
            'aws_targets': ['mariadb', 'mysql', 'aurora-mysql'],
            'complexity_multiplier': 0.4
        },
        'aurora-postgresql': {
            'name': 'Amazon Aurora PostgreSQL',
            'features': ['Auto-scaling', 'Global Database', 'Serverless'],
            'aws_targets': ['aurora-postgresql'],
            'complexity_multiplier': 0.3
        },
        'aurora-mysql': {
            'name': 'Amazon Aurora MySQL',
            'features': ['Auto-scaling', 'Global Database', 'Serverless'],
            'aws_targets': ['aurora-mysql'],
            'complexity_multiplier': 0.3
        }
    }
    
    @classmethod
    def get_migration_type(cls, source_engine: str, target_engine: str) -> str:
        """Determine migration type based on source and target engines"""
        if source_engine == target_engine:
            return "homogeneous"
        
        mysql_family = ['mysql', 'mariadb', 'aurora-mysql']
        postgres_family = ['postgres', 'aurora-postgresql']
        oracle_family = ['oracle-ee', 'oracle-se']
        
        source_family = None
        target_family = None
        
        for family in [mysql_family, postgres_family, oracle_family]:
            if source_engine in family:
                source_family = family
            if target_engine in family:
                target_family = family
        
        if source_family and source_family == target_family:
            return "homogeneous"
        else:
            return "heterogeneous"
    
    @classmethod
    def get_complexity_multiplier(cls, source_engine: str, target_engine: str) -> float:
        """Get complexity multiplier for migration"""
        if source_engine == target_engine:
            return 1.0
        
        source_complexity = cls.ENGINES.get(source_engine, {}).get('complexity_multiplier', 1.0)
        target_complexity = cls.ENGINES.get(target_engine, {}).get('complexity_multiplier', 1.0)
        
        migration_type = cls.get_migration_type(source_engine, target_engine)
        
        if migration_type == "heterogeneous":
            if 'oracle' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.5
            elif 'sql-server' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.0
            else:
                return source_complexity * 1.5
        else:
            return max(source_complexity, target_complexity)

# ===========================
# VROPS METRICS ANALYZER
# ===========================

class VRopsMetricsAnalyzer:
    """Comprehensive vROps metrics analysis for accurate AWS sizing"""
    
    def __init__(self):
        self.required_metrics = self._initialize_required_metrics()
        self.aws_instance_specs = self._initialize_aws_instance_specs()
        self.performance_buffers = self._initialize_performance_buffers()
    
    def _initialize_required_metrics(self) -> Dict:
        """Initialize comprehensive vROps metrics mapping"""
        return {
            'cpu_metrics': {
                'max_cpu_usage_percent': {'required': True, 'description': 'Peak CPU utilization over observation period'},
                'avg_cpu_usage_percent': {'required': True, 'description': 'Average CPU utilization'},
                'cpu_ready_time_ms': {'required': False, 'description': 'CPU ready time indicating resource contention'},
                'cpu_costop_ms': {'required': False, 'description': 'CPU co-stop time for multi-vCPU VMs'},
                'cpu_demand_mhz': {'required': False, 'description': 'CPU demand in MHz'},
                'cpu_cores_allocated': {'required': True, 'description': 'Number of allocated vCPU cores'},
                'cpu_sockets': {'required': False, 'description': 'Number of CPU sockets'}
            },
            'memory_metrics': {
                'max_memory_usage_percent': {'required': True, 'description': 'Peak memory utilization'},
                'avg_memory_usage_percent': {'required': True, 'description': 'Average memory utilization'},
                'memory_consumed_gb': {'required': True, 'description': 'Actual memory consumed by VM'},
                'memory_active_gb': {'required': False, 'description': 'Currently active memory'},
                'memory_balloon_gb': {'required': False, 'description': 'Ballooned memory (overcommit indicator)'},
                'memory_compressed_gb': {'required': False, 'description': 'Compressed memory'},
                'memory_swapped_gb': {'required': False, 'description': 'Swapped memory (performance issue indicator)'},
                'memory_allocated_gb': {'required': True, 'description': 'Total allocated memory to VM'}
            },
            'storage_metrics': {
                'max_iops_total': {'required': True, 'description': 'Peak total IOPS (read + write)'},
                'max_iops_read': {'required': False, 'description': 'Peak read IOPS'},
                'max_iops_write': {'required': False, 'description': 'Peak write IOPS'},
                'avg_iops_total': {'required': True, 'description': 'Average total IOPS'},
                'max_disk_latency_ms': {'required': True, 'description': 'Peak disk latency in milliseconds'},
                'avg_disk_latency_ms': {'required': True, 'description': 'Average disk latency'},
                'max_disk_throughput_mbps': {'required': True, 'description': 'Peak disk throughput MB/s'},
                'avg_disk_throughput_mbps': {'required': True, 'description': 'Average disk throughput'},
                'disk_queue_depth': {'required': False, 'description': 'Average disk queue depth'},
                'storage_allocated_gb': {'required': True, 'description': 'Total allocated storage'},
                'storage_used_gb': {'required': True, 'description': 'Actually used storage'}
            },
            'network_metrics': {
                'max_network_throughput_mbps': {'required': True, 'description': 'Peak network throughput'},
                'avg_network_throughput_mbps': {'required': True, 'description': 'Average network throughput'},
                'max_network_packets_per_sec': {'required': False, 'description': 'Peak network packets per second'},
                'network_latency_ms': {'required': False, 'description': 'Network latency in milliseconds'},
                'network_packet_loss_percent': {'required': False, 'description': 'Network packet loss percentage'}
            },
            'database_metrics': {
                'max_database_connections': {'required': True, 'description': 'Peak concurrent database connections'},
                'avg_database_connections': {'required': True, 'description': 'Average database connections'},
                'max_transaction_rate_per_sec': {'required': False, 'description': 'Peak transactions per second'},
                'avg_query_response_time_ms': {'required': False, 'description': 'Average query response time'},
                'buffer_cache_hit_ratio_percent': {'required': False, 'description': 'Database buffer cache hit ratio'},
                'lock_wait_time_ms': {'required': False, 'description': 'Average lock wait time'},
                'database_size_gb': {'required': True, 'description': 'Current database size'},
                'log_file_size_gb': {'required': False, 'description': 'Transaction log file size'},
                'tempdb_usage_gb': {'required': False, 'description': 'TempDB usage (SQL Server)'}
            },
            'workload_patterns': {
                'peak_hours_start': {'required': False, 'description': 'Peak workload start time (24hr format)'},
                'peak_hours_end': {'required': False, 'description': 'Peak workload end time (24hr format)'},
                'weekend_usage_factor': {'required': False, 'description': 'Weekend usage as % of weekday'},
                'seasonal_peak_factor': {'required': False, 'description': 'Seasonal peak multiplier'},
                'growth_rate_percent_annual': {'required': False, 'description': 'Expected annual growth rate'},
                'observation_period_days': {'required': True, 'description': 'Data collection period in days'},
                'availability_percent': {'required': False, 'description': 'System availability percentage'},
                'batch_processing_hours': {'required': False, 'description': 'Hours per day for batch processing'}
            },
            'application_metrics': {
                'application_type': {'required': True, 'description': 'Type of application (OLTP, OLAP, Mixed)'},
                'concurrent_users_max': {'required': False, 'description': 'Maximum concurrent users'},
                'concurrent_users_avg': {'required': False, 'description': 'Average concurrent users'},
                'response_time_sla_ms': {'required': False, 'description': 'Application response time SLA'},
                'downtime_tolerance_minutes': {'required': False, 'description': 'Maximum acceptable downtime'},
                'backup_window_hours': {'required': False, 'description': 'Backup window duration'},
                'maintenance_window_hours': {'required': False, 'description': 'Maintenance window duration'}
            }
        }
    
    def _initialize_aws_instance_specs(self) -> Dict:
        """Initialize AWS instance specifications for accurate mapping"""
        return {
            # T3 instances (Burstable)
            'db.t3.micro': {'vcpu': 2, 'memory_gb': 1, 'network_gbps': 1.5, 'ebs_optimized': True, 'baseline_cpu': 20},
            'db.t3.small': {'vcpu': 2, 'memory_gb': 2, 'network_gbps': 1.5, 'ebs_optimized': True, 'baseline_cpu': 20},
            'db.t3.medium': {'vcpu': 2, 'memory_gb': 4, 'network_gbps': 1.5, 'ebs_optimized': True, 'baseline_cpu': 20},
            'db.t3.large': {'vcpu': 2, 'memory_gb': 8, 'network_gbps': 1.5, 'ebs_optimized': True, 'baseline_cpu': 30},
            'db.t3.xlarge': {'vcpu': 4, 'memory_gb': 16, 'network_gbps': 1.5, 'ebs_optimized': True, 'baseline_cpu': 40},
            'db.t3.2xlarge': {'vcpu': 8, 'memory_gb': 32, 'network_gbps': 1.5, 'ebs_optimized': True, 'baseline_cpu': 40},
            
            # R5 instances (Memory Optimized)
            'db.r5.large': {'vcpu': 2, 'memory_gb': 16, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.xlarge': {'vcpu': 4, 'memory_gb': 32, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.2xlarge': {'vcpu': 8, 'memory_gb': 64, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.4xlarge': {'vcpu': 16, 'memory_gb': 128, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.8xlarge': {'vcpu': 32, 'memory_gb': 256, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.12xlarge': {'vcpu': 48, 'memory_gb': 384, 'network_gbps': 12, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.16xlarge': {'vcpu': 64, 'memory_gb': 512, 'network_gbps': 20, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.24xlarge': {'vcpu': 96, 'memory_gb': 768, 'network_gbps': 25, 'ebs_optimized': True, 'baseline_cpu': 100}
        }
    
    def _initialize_performance_buffers(self) -> Dict:
        """Initialize performance buffers based on environment type and workload"""
        return {
            'production': {
                'cpu_buffer': 0.3,      # 30% headroom
                'memory_buffer': 0.25,   # 25% headroom
                'iops_buffer': 0.4,      # 40% headroom
                'network_buffer': 0.3    # 30% headroom
            },
            'staging': {
                'cpu_buffer': 0.2,
                'memory_buffer': 0.15,
                'iops_buffer': 0.3,
                'network_buffer': 0.2
            },
            'testing': {
                'cpu_buffer': 0.15,
                'memory_buffer': 0.1,
                'iops_buffer': 0.2,
                'network_buffer': 0.15
            },
            'development': {
                'cpu_buffer': 0.1,
                'memory_buffer': 0.05,
                'iops_buffer': 0.15,
                'network_buffer': 0.1
            }
        }
    
    def analyze_vrops_metrics(self, environment_specs: Dict) -> Dict:
        """Analyze vROps metrics and provide AWS sizing recommendations"""
        
        analysis_results = {}
        
        for env_name, metrics in environment_specs.items():
            env_analysis = self._analyze_single_environment(env_name, metrics)
            analysis_results[env_name] = env_analysis
        
        # Generate overall recommendations
        analysis_results['overall_recommendations'] = self._generate_overall_recommendations(analysis_results)
        
        return analysis_results
    
    def _analyze_single_environment(self, env_name: str, metrics: Dict) -> Dict:
        """Analyze a single environment's vROps metrics"""
        
        # Determine environment type
        env_type = self._categorize_environment(env_name)
        
        # Get performance buffers
        buffers = self.performance_buffers[env_type]
        
        # Analyze each metric category
        cpu_analysis = self._analyze_cpu_metrics(metrics, buffers)
        memory_analysis = self._analyze_memory_metrics(metrics, buffers)
        storage_analysis = self._analyze_storage_metrics(metrics, buffers)
        network_analysis = self._analyze_network_metrics(metrics, buffers)
        workload_analysis = self._analyze_workload_patterns(metrics)
        
        # Generate AWS instance recommendations
        instance_recommendations = self._recommend_aws_instances(
            cpu_analysis, memory_analysis, storage_analysis, network_analysis, env_type
        )
        
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(metrics)
        
        # Generate optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            metrics, cpu_analysis, memory_analysis, storage_analysis
        )
        
        return {
            'environment_type': env_type,
            'cpu_analysis': cpu_analysis,
            'memory_analysis': memory_analysis,
            'storage_analysis': storage_analysis,
            'network_analysis': network_analysis,
            'workload_analysis': workload_analysis,
            'instance_recommendations': instance_recommendations,
            'performance_scores': performance_scores,
            'optimization_opportunities': optimization_opportunities,
            'risk_indicators': self._identify_risk_indicators(metrics)
        }
    
    def _categorize_environment(self, env_name: str) -> str:
        """Categorize environment type from name"""
        env_lower = env_name.lower()
        if any(term in env_lower for term in ['prod', 'production', 'prd']):
            return 'production'
        elif any(term in env_lower for term in ['stag', 'staging', 'preprod']):
            return 'staging'
        elif any(term in env_lower for term in ['qa', 'test', 'uat', 'sqa']):
            return 'testing'
        elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
            return 'development'
        return 'production'  # Default to production for safety
    
    def _analyze_cpu_metrics(self, metrics: Dict, buffers: Dict) -> Dict:
        """Analyze CPU metrics and requirements"""
        
        max_cpu = metrics.get('max_cpu_usage_percent', 0)
        avg_cpu = metrics.get('avg_cpu_usage_percent', 0)
        cpu_cores = metrics.get('cpu_cores_allocated', 2)
        cpu_ready = metrics.get('cpu_ready_time_ms', 0)
        
        # Calculate required capacity with buffer
        required_cpu_percent = max_cpu * (1 + buffers['cpu_buffer'])
        
        # Determine if there are performance issues
        has_cpu_contention = cpu_ready > 5000  # 5+ seconds of ready time indicates contention
        
        # Recommend CPU scaling
        if required_cpu_percent > 80:
            cpu_scaling_recommendation = "Scale up CPU - current peak exceeds comfort zone"
        elif has_cpu_contention:
            cpu_scaling_recommendation = "Address CPU contention - consider fewer vCPUs or higher CPU limits"
        else:
            cpu_scaling_recommendation = "CPU sizing appears adequate"
        
        return {
            'max_usage_percent': max_cpu,
            'avg_usage_percent': avg_cpu,
            'required_capacity_percent': required_cpu_percent,
            'current_cores': cpu_cores,
            'has_contention': has_cpu_contention,
            'cpu_ready_time_ms': cpu_ready,
            'scaling_recommendation': cpu_scaling_recommendation,
            'utilization_efficiency': avg_cpu / max_cpu if max_cpu > 0 else 0
        }
    
    def _analyze_memory_metrics(self, metrics: Dict, buffers: Dict) -> Dict:
        """Analyze memory metrics and requirements"""
        
        max_memory = metrics.get('max_memory_usage_percent', 0)
        avg_memory = metrics.get('avg_memory_usage_percent', 0)
        memory_allocated = metrics.get('memory_allocated_gb', 8)
        memory_balloon = metrics.get('memory_balloon_gb', 0)
        memory_swapped = metrics.get('memory_swapped_gb', 0)
        
        # Calculate required capacity with buffer
        required_memory_percent = max_memory * (1 + buffers['memory_buffer'])
        
        # Identify memory pressure indicators
        has_memory_pressure = memory_balloon > 0 or memory_swapped > 0
        memory_pressure_severity = 'High' if memory_swapped > 0 else 'Medium' if memory_balloon > 0 else 'None'
        
        # Memory sizing recommendation
        if required_memory_percent > 85:
            memory_scaling_recommendation = "Increase memory allocation - peak usage too high"
        elif has_memory_pressure:
            memory_scaling_recommendation = f"Address memory pressure ({memory_pressure_severity}) - increase memory allocation"
        else:
            memory_scaling_recommendation = "Memory sizing appears adequate"
        
        return {
            'max_usage_percent': max_memory,
            'avg_usage_percent': avg_memory,
            'required_capacity_percent': required_memory_percent,
            'allocated_gb': memory_allocated,
            'has_pressure': has_memory_pressure,
            'pressure_severity': memory_pressure_severity,
            'balloon_gb': memory_balloon,
            'swapped_gb': memory_swapped,
            'scaling_recommendation': memory_scaling_recommendation,
            'utilization_efficiency': avg_memory / max_memory if max_memory > 0 else 0
        }
    
    def _analyze_storage_metrics(self, metrics: Dict, buffers: Dict) -> Dict:
        """Analyze storage performance metrics"""
        
        max_iops = metrics.get('max_iops_total', 0)
        avg_iops = metrics.get('avg_iops_total', 0)
        max_latency = metrics.get('max_disk_latency_ms', 0)
        avg_latency = metrics.get('avg_disk_latency_ms', 0)
        max_throughput = metrics.get('max_disk_throughput_mbps', 0)
        storage_used = metrics.get('storage_used_gb', 0)
        storage_allocated = metrics.get('storage_allocated_gb', 0)
        
        # Calculate required IOPS with buffer
        required_iops = max_iops * (1 + buffers['iops_buffer'])
        
        # Analyze performance characteristics
        storage_utilization = (storage_used / storage_allocated * 100) if storage_allocated > 0 else 0
        
        # Performance assessment
        if avg_latency > 20:
            latency_assessment = "High latency detected - consider faster storage"
        elif avg_latency > 10:
            latency_assessment = "Moderate latency - monitor closely"
        else:
            latency_assessment = "Latency within acceptable range"
        
        # Storage type recommendation based on IOPS and latency
        if required_iops > 20000 or max_latency > 20:
            recommended_storage_type = "io2 (high IOPS, low latency required)"
        elif required_iops > 3000:
            recommended_storage_type = "gp3 (balanced performance)"
        else:
            recommended_storage_type = "gp3 (general purpose)"
        
        return {
            'max_iops': max_iops,
            'avg_iops': avg_iops,
            'required_iops': required_iops,
            'max_latency_ms': max_latency,
            'avg_latency_ms': avg_latency,
            'max_throughput_mbps': max_throughput,
            'storage_utilization_percent': storage_utilization,
            'latency_assessment': latency_assessment,
            'recommended_storage_type': recommended_storage_type,
            'iops_efficiency': avg_iops / max_iops if max_iops > 0 else 0
        }
    
    def _analyze_network_metrics(self, metrics: Dict, buffers: Dict) -> Dict:
        """Analyze network performance metrics"""
        
        max_throughput = metrics.get('max_network_throughput_mbps', 0)
        avg_throughput = metrics.get('avg_network_throughput_mbps', 0)
        network_latency = metrics.get('network_latency_ms', 0)
        packet_loss = metrics.get('network_packet_loss_percent', 0)
        
        # Calculate required bandwidth with buffer
        required_bandwidth_mbps = max_throughput * (1 + buffers['network_buffer'])
        
        # Network performance assessment
        if packet_loss > 0.1:
            network_health = "Poor - packet loss detected"
        elif network_latency > 50:
            network_health = "Fair - high latency detected"
        else:
            network_health = "Good - network performance within normal range"
        
        # Bandwidth recommendation
        if required_bandwidth_mbps > 1000:
            bandwidth_recommendation = "Consider Enhanced Networking for high bandwidth requirements"
        else:
            bandwidth_recommendation = "Standard networking should be sufficient"
        
        return {
            'max_throughput_mbps': max_throughput,
            'avg_throughput_mbps': avg_throughput,
            'required_bandwidth_mbps': required_bandwidth_mbps,
            'latency_ms': network_latency,
            'packet_loss_percent': packet_loss,
            'network_health': network_health,
            'bandwidth_recommendation': bandwidth_recommendation,
            'throughput_efficiency': avg_throughput / max_throughput if max_throughput > 0 else 0
        }
    
    def _analyze_workload_patterns(self, metrics: Dict) -> Dict:
        """Analyze workload patterns and usage characteristics"""
        
        application_type = metrics.get('application_type', 'Mixed')
        peak_start = metrics.get('peak_hours_start', 9)
        peak_end = metrics.get('peak_hours_end', 17)
        weekend_factor = metrics.get('weekend_usage_factor', 0.3)
        growth_rate = metrics.get('growth_rate_percent_annual', 10)
        observation_days = metrics.get('observation_period_days', 30)
        
        # Calculate peak duration
        peak_duration_hours = peak_end - peak_start if peak_end > peak_start else (24 - peak_start) + peak_end
        
        # Workload classification
        if peak_duration_hours <= 8:
            workload_classification = "Highly Variable - Strong Peak Pattern"
        elif peak_duration_hours <= 12:
            workload_classification = "Moderate Variability - Extended Peak"
        else:
            workload_classification = "Steady State - Minimal Peak Variation"
        
        # Growth planning
        if growth_rate > 20:
            growth_planning = "High growth - plan for significant scaling"
        elif growth_rate > 10:
            growth_planning = "Moderate growth - include scaling buffer"
        else:
            growth_planning = "Stable growth - standard planning sufficient"
        
        return {
            'application_type': application_type,
            'peak_hours': f"{peak_start:02d}:00 - {peak_end:02d}:00",
            'peak_duration_hours': peak_duration_hours,
            'weekend_usage_factor': weekend_factor,
            'annual_growth_rate': growth_rate,
            'observation_period_days': observation_days,
            'workload_classification': workload_classification,
            'growth_planning': growth_planning,
            'data_quality_score': min(100, observation_days * 3.33)  # 30+ days = 100%
        }
    
    def _recommend_aws_instances(self, cpu_analysis: Dict, memory_analysis: Dict, 
                               storage_analysis: Dict, network_analysis: Dict, env_type: str) -> List[Dict]:
        """Recommend AWS RDS instances based on analyzed metrics"""
        
        # Calculate requirements
        required_cpu_cores = max(2, int(cpu_analysis['required_capacity_percent'] / 100 * cpu_analysis['current_cores']))
        required_memory_gb = max(4, memory_analysis['allocated_gb'] * (memory_analysis['required_capacity_percent'] / 100))
        required_iops = storage_analysis['required_iops']
        required_bandwidth_mbps = network_analysis['required_bandwidth_mbps']
        
        recommendations = []
        
        # Filter instances that meet requirements
        for instance_type, specs in self.aws_instance_specs.items():
            if (specs['vcpu'] >= required_cpu_cores and 
                specs['memory_gb'] >= required_memory_gb):
                
                # Calculate fit score
                cpu_efficiency = required_cpu_cores / specs['vcpu']
                memory_efficiency = required_memory_gb / specs['memory_gb']
                overall_efficiency = (cpu_efficiency + memory_efficiency) / 2
                
                # Penalize over-provisioning, reward good fit
                fit_score = overall_efficiency * 100
                if overall_efficiency < 0.5:
                    fit_score *= 0.7  # Penalty for over-provisioning
                
                # Consider network requirements
                if required_bandwidth_mbps > specs['network_gbps'] * 1000 * 0.8:
                    fit_score *= 0.8  # Penalty for potential network bottleneck
                
                # Environment-specific adjustments
                if env_type == 'production' and 't3' in instance_type:
                    fit_score *= 0.9  # Slight penalty for burstable in production
                elif env_type == 'development' and 'r5' in instance_type:
                    fit_score *= 0.9  # Slight penalty for memory-optimized in dev
                
                recommendations.append({
                    'instance_type': instance_type,
                    'vcpu': specs['vcpu'],
                    'memory_gb': specs['memory_gb'],
                    'network_gbps': specs['network_gbps'],
                    'fit_score': fit_score,
                    'cpu_efficiency': cpu_efficiency,
                    'memory_efficiency': memory_efficiency,
                    'recommendation_reason': self._generate_recommendation_reason(
                        instance_type, cpu_efficiency, memory_efficiency, env_type
                    )
                })
        
        # Sort by fit score and return top 3
        recommendations.sort(key=lambda x: x['fit_score'], reverse=True)
        return recommendations[:3]
    
    def _generate_recommendation_reason(self, instance_type: str, cpu_eff: float, mem_eff: float, env_type: str) -> str:
        """Generate human-readable recommendation reasoning"""
        
        instance_family = instance_type.split('.')[1][:2]
        
        reasons = []
        
        if instance_family == 't3':
            reasons.append("Cost-effective burstable performance")
        elif instance_family == 'r5':
            reasons.append("Memory-optimized for database workloads")
        
        if cpu_eff > 0.8:
            reasons.append("excellent CPU utilization")
        elif cpu_eff > 0.6:
            reasons.append("good CPU utilization")
        else:
            reasons.append("conservative CPU sizing for headroom")
        
        if mem_eff > 0.8:
            reasons.append("excellent memory utilization")
        elif mem_eff > 0.6:
            reasons.append("good memory utilization")
        else:
            reasons.append("conservative memory sizing for headroom")
        
        return f"Recommended for {env_type}: " + ", ".join(reasons)
    
    def _calculate_performance_scores(self, metrics: Dict) -> Dict:
        """Calculate overall performance health scores"""
        
        scores = {}
        
        # CPU Health Score
        cpu_usage = metrics.get('avg_cpu_usage_percent', 0)
        cpu_ready = metrics.get('cpu_ready_time_ms', 0)
        
        if cpu_usage < 20:
            cpu_score = 60  # Under-utilized
        elif cpu_usage < 70:
            cpu_score = 100  # Optimal
        elif cpu_usage < 85:
            cpu_score = 80  # High but acceptable
        else:
            cpu_score = 40  # Over-utilized
        
        if cpu_ready > 5000:
            cpu_score *= 0.7  # Penalty for contention
        
        scores['cpu_health'] = min(100, cpu_score)
        
        # Memory Health Score
        memory_usage = metrics.get('avg_memory_usage_percent', 0)
        memory_balloon = metrics.get('memory_balloon_gb', 0)
        memory_swapped = metrics.get('memory_swapped_gb', 0)
        
        if memory_usage < 30:
            memory_score = 70  # Under-utilized
        elif memory_usage < 80:
            memory_score = 100  # Optimal
        elif memory_usage < 90:
            memory_score = 75  # High but acceptable
        else:
            memory_score = 50  # Over-utilized
        
        if memory_swapped > 0:
            memory_score *= 0.5  # Severe penalty for swapping
        elif memory_balloon > 0:
            memory_score *= 0.8  # Moderate penalty for ballooning
        
        scores['memory_health'] = min(100, memory_score)
        
        # Storage Health Score
        avg_latency = metrics.get('avg_disk_latency_ms', 0)
        max_latency = metrics.get('max_disk_latency_ms', 0)
        
        if avg_latency < 5:
            storage_score = 100  # Excellent
        elif avg_latency < 10:
            storage_score = 90   # Good
        elif avg_latency < 20:
            storage_score = 70   # Acceptable
        else:
            storage_score = 40   # Poor
        
        if max_latency > 100:
            storage_score *= 0.8  # Penalty for high peak latency
        
        scores['storage_health'] = min(100, storage_score)
        
        # Overall Health Score
        scores['overall_health'] = (scores['cpu_health'] + scores['memory_health'] + scores['storage_health']) / 3
        
        return scores
    
    def _identify_optimization_opportunities(self, metrics: Dict, cpu_analysis: Dict, 
                                           memory_analysis: Dict, storage_analysis: Dict) -> List[Dict]:
        """Identify optimization opportunities"""
        
        opportunities = []
        
        # CPU optimization
        if cpu_analysis['avg_usage_percent'] < 30:
            opportunities.append({
                'category': 'CPU Optimization',
                'opportunity': 'Right-size CPU allocation',
                'description': f"Average CPU usage is only {cpu_analysis['avg_usage_percent']:.1f}% - consider reducing vCPU allocation",
                'potential_savings': 'Medium',
                'effort': 'Low'
            })
        
        # Memory optimization
        if memory_analysis['avg_usage_percent'] < 40:
            opportunities.append({
                'category': 'Memory Optimization',
                'opportunity': 'Right-size memory allocation',
                'description': f"Average memory usage is only {memory_analysis['avg_usage_percent']:.1f}% - consider reducing memory allocation",
                'potential_savings': 'Medium',
                'effort': 'Low'
            })
        
        # Storage optimization
        storage_util = metrics.get('storage_used_gb', 0) / metrics.get('storage_allocated_gb', 1) * 100
        if storage_util < 50:
            opportunities.append({
                'category': 'Storage Optimization',
                'opportunity': 'Optimize storage allocation',
                'description': f"Storage utilization is only {storage_util:.1f}% - consider reducing allocated storage",
                'potential_savings': 'Low',
                'effort': 'Low'
            })
        
        # Performance optimization
        if storage_analysis['avg_latency_ms'] > 15:
            opportunities.append({
                'category': 'Performance Optimization',
                'opportunity': 'Upgrade storage type',
                'description': f"Average disk latency is {storage_analysis['avg_latency_ms']:.1f}ms - consider faster storage",
                'potential_savings': 'N/A (Performance)',
                'effort': 'Medium'
            })
        
        return opportunities
    
    def _identify_risk_indicators(self, metrics: Dict) -> List[Dict]:
        """Identify performance and reliability risk indicators"""
        
        risks = []
        
        # CPU risks
        max_cpu = metrics.get('max_cpu_usage_percent', 0)
        cpu_ready = metrics.get('cpu_ready_time_ms', 0)
        
        if max_cpu > 90:
            risks.append({
                'category': 'CPU Risk',
                'risk': 'High CPU utilization',
                'description': f"Peak CPU usage reached {max_cpu:.1f}% - may cause performance degradation",
                'severity': 'High' if max_cpu > 95 else 'Medium',
                'recommendation': 'Increase CPU allocation or optimize CPU-intensive processes'
            })
        
        if cpu_ready > 10000:
            risks.append({
                'category': 'CPU Risk',
                'risk': 'CPU contention detected',
                'description': f"CPU ready time is {cpu_ready/1000:.1f} seconds - indicates resource contention",
                'severity': 'High',
                'recommendation': 'Reduce vCPU count or increase CPU resource limits'
            })
        
        # Memory risks
        memory_swapped = metrics.get('memory_swapped_gb', 0)
        memory_balloon = metrics.get('memory_balloon_gb', 0)
        
        if memory_swapped > 0:
            risks.append({
                'category': 'Memory Risk',
                'risk': 'Memory swapping detected',
                'description': f"{memory_swapped:.1f} GB of memory is swapped - severe performance impact",
                'severity': 'Critical',
                'recommendation': 'Immediately increase memory allocation'
            })
        
        if memory_balloon > 0:
            risks.append({
                'category': 'Memory Risk',
                'risk': 'Memory ballooning active',
                'description': f"{memory_balloon:.1f} GB of memory is ballooned - indicates memory pressure",
                'severity': 'Medium',
                'recommendation': 'Consider increasing memory allocation'
            })
        
        # Storage risks
        max_latency = metrics.get('max_disk_latency_ms', 0)
        avg_latency = metrics.get('avg_disk_latency_ms', 0)
        
        if avg_latency > 20:
            risks.append({
                'category': 'Storage Risk',
                'risk': 'High storage latency',
                'description': f"Average disk latency is {avg_latency:.1f}ms - may impact application performance",
                'severity': 'High' if avg_latency > 50 else 'Medium',
                'recommendation': 'Consider faster storage tier or optimize I/O patterns'
            })
        
        return risks
    
    def _generate_overall_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate overall migration recommendations based on all environments"""
        
        recommendations = {
            'migration_strategy': '',
            'infrastructure_recommendations': [],
            'cost_optimization_opportunities': [],
            'performance_considerations': [],
            'risk_mitigation_actions': []
        }
        
        # Analyze patterns across environments
        env_count = len([k for k in analysis_results.keys() if k != 'overall_recommendations'])
        prod_envs = [k for k, v in analysis_results.items() 
                    if k != 'overall_recommendations' and v.get('environment_type') == 'production']
        
        # Migration strategy recommendation
        if env_count <= 3:
            recommendations['migration_strategy'] = "Small-scale migration - consider phased approach starting with non-production"
        elif env_count <= 6:
            recommendations['migration_strategy'] = "Medium-scale migration - implement parallel migration streams"
        else:
            recommendations['migration_strategy'] = "Large-scale migration - consider automation tools and dedicated migration team"
        
        # Infrastructure recommendations based on analysis
        total_cores = sum([v.get('cpu_analysis', {}).get('current_cores', 0) 
                          for v in analysis_results.values() if isinstance(v, dict)])
        total_memory = sum([v.get('memory_analysis', {}).get('allocated_gb', 0) 
                           for v in analysis_results.values() if isinstance(v, dict)])
        
        if total_cores > 200:
            recommendations['infrastructure_recommendations'].append(
                "Consider AWS Enterprise Support for large-scale infrastructure"
            )
        
        if len(prod_envs) > 1:
            recommendations['infrastructure_recommendations'].append(
                "Implement Multi-AZ deployment for all production environments"
            )
        
        return recommendations

# ===========================
# NETWORK TRANSFER ANALYZER
# ===========================

class NetworkTransferAnalyzer:
    """Comprehensive network transfer analysis for AWS database migration"""
    
    def __init__(self):
        self.transfer_patterns = self._initialize_transfer_patterns()
        self.aws_regions_bandwidth = self._initialize_region_bandwidth()
        
    def _initialize_transfer_patterns(self) -> Dict:
        """Initialize supported network transfer patterns"""
        return {
            'internet_dms': {
                'name': 'Internet + DMS',
                'description': 'Standard internet connection with AWS Database Migration Service',
                'components': ['Internet Gateway', 'DMS Replication Instance'],
                'use_cases': ['Small to medium databases (<1TB)', 'Standard migrations', 'Cost-sensitive projects'],
                'pros': ['Low setup cost', 'Quick to implement', 'No additional infrastructure'],
                'cons': ['Variable bandwidth', 'Higher data transfer costs', 'Potential latency issues'],
                'security_level': 'Standard',
                'complexity': 'Low'
            },
            'dx_dms': {
                'name': 'Direct Connect + DMS',
                'description': 'Dedicated connection with AWS Database Migration Service',
                'components': ['AWS Direct Connect', 'DMS Replication Instance', 'Virtual Interface'],
                'use_cases': ['Large databases (>1TB)', 'Consistent bandwidth needs', 'Ongoing hybrid connectivity'],
                'pros': ['Predictable bandwidth', 'Lower data transfer costs', 'Reduced latency'],
                'cons': ['Higher setup cost', 'Longer setup time', 'Requires dedicated circuit'],
                'security_level': 'High',
                'complexity': 'Medium'
            },
            'dx_datasync_vpc': {
                'name': 'Direct Connect + DataSync + VPC Endpoints',
                'description': 'Dedicated connection with DataSync using VPC endpoints for private connectivity',
                'components': ['AWS Direct Connect', 'DataSync Agent', 'VPC Endpoints', 'S3 Gateway Endpoint'],
                'use_cases': ['File-based data transfer', 'Object storage migration', 'High security requirements'],
                'pros': ['Private connectivity', 'Optimized for file transfer', 'No internet routing'],
                'cons': ['Complex setup', 'Additional VPC endpoint costs', 'Limited to file-based transfers'],
                'security_level': 'Very High',
                'complexity': 'High'
            },
            'vpn_dms': {
                'name': 'VPN + DMS',
                'description': 'Site-to-site VPN with AWS Database Migration Service',
                'components': ['VPN Gateway', 'Customer Gateway', 'DMS Replication Instance'],
                'use_cases': ['Medium databases', 'Secure connectivity required', 'Temporary migration setup'],
                'pros': ['Secure connection', 'Quick setup', 'Cost-effective'],
                'cons': ['Internet-dependent', 'Bandwidth limitations', 'Potential latency'],
                'security_level': 'High',
                'complexity': 'Medium'
            },
            'hybrid_snowball_dms': {
                'name': 'Snowball + DMS Hybrid',
                'description': 'Initial bulk transfer via Snowball, ongoing sync with DMS',
                'components': ['AWS Snowball', 'DMS Replication Instance', 'S3 Bucket'],
                'use_cases': ['Very large databases (>10TB)', 'Limited bandwidth', 'Minimal downtime required'],
                'pros': ['Fast initial transfer', 'Minimal bandwidth usage', 'Reduced downtime'],
                'cons': ['Complex orchestration', 'Physical device handling', 'Higher coordination effort'],
                'security_level': 'High',
                'complexity': 'Very High'
            },
            'multipath_redundant': {
                'name': 'Multi-path Redundant Transfer',
                'description': 'Redundant connections using both Direct Connect and VPN',
                'components': ['AWS Direct Connect', 'VPN Gateway', 'DMS Replication Instance', 'Route Tables'],
                'use_cases': ['Mission-critical migrations', 'Zero-tolerance for failures', 'High availability requirements'],
                'pros': ['Maximum reliability', 'Automatic failover', 'Redundant paths'],
                'cons': ['Highest cost', 'Complex configuration', 'Over-engineering for most use cases'],
                'security_level': 'Very High',
                'complexity': 'Very High'
            }
        }
    
    def _initialize_region_bandwidth(self) -> Dict:
        """Initialize typical bandwidth capabilities by region"""
        return {
            'us-east-1': {'max_dx_gbps': 100, 'typical_internet_mbps': 1000},
            'us-west-2': {'max_dx_gbps': 100, 'typical_internet_mbps': 1000},
            'eu-west-1': {'max_dx_gbps': 100, 'typical_internet_mbps': 800},
            'ap-southeast-1': {'max_dx_gbps': 50, 'typical_internet_mbps': 500},
            'ap-northeast-1': {'max_dx_gbps': 100, 'typical_internet_mbps': 800}
        }
    
    def calculate_transfer_analysis(self, migration_params: Dict) -> Dict:
        """Calculate comprehensive transfer analysis for all patterns"""
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        region = migration_params.get('region', 'us-east-1')
        available_bandwidth_mbps = migration_params.get('bandwidth_mbps', 1000)
        security_requirements = migration_params.get('security_requirements', 'standard')
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        budget_constraints = migration_params.get('budget_constraints', 'medium')
        
        results = {}
        
        for pattern_id, pattern_info in self.transfer_patterns.items():
            results[pattern_id] = self._calculate_pattern_metrics(
                pattern_id, pattern_info, data_size_gb, region, 
                available_bandwidth_mbps, security_requirements, timeline_weeks
            )
        
        # Add recommendation engine
        results['recommendations'] = self._generate_recommendations(
            results, migration_params
        )
        
        return results
    
    def _calculate_pattern_metrics(self, pattern_id: str, pattern_info: Dict, 
                                 data_size_gb: int, region: str, bandwidth_mbps: int,
                                 security_req: str, timeline_weeks: int) -> Dict:
        """Calculate metrics for a specific transfer pattern"""
        
        # Base calculations
        data_size_tb = data_size_gb / 1024
        
        # Pattern-specific calculations
        if pattern_id == 'internet_dms':
            return self._calc_internet_dms(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'dx_dms':
            return self._calc_dx_dms(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'dx_datasync_vpc':
            return self._calc_dx_datasync_vpc(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'vpn_dms':
            return self._calc_vpn_dms(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'hybrid_snowball_dms':
            return self._calc_hybrid_snowball(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'multipath_redundant':
            return self._calc_multipath_redundant(data_size_gb, bandwidth_mbps, region)
        
        return {}
    
    def _calc_internet_dms(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate Internet + DMS pattern metrics"""
        
        # Transfer time calculation (with 70% efficiency factor for internet)
        effective_bandwidth = bandwidth_mbps * 0.7
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)  # Convert to hours
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.09  # $0.09 per GB for internet transfer
        dms_instance_cost = 0.34 * (transfer_time_hours / 24)  # t3.large DMS instance per day
        setup_cost = 2000  # Basic setup costs
        
        total_cost = data_transfer_cost + dms_instance_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': dms_instance_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 70,  # Percentage
            'reliability_score': 75,
            'security_score': 60,
            'complexity_score': 20
        }
    
    def _calc_dx_dms(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate Direct Connect + DMS pattern metrics"""
        
        # Transfer time calculation (with 95% efficiency for DX)
        effective_bandwidth = bandwidth_mbps * 0.95
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.02  # $0.02 per GB for DX transfer
        dx_port_cost = 500 * (transfer_time_hours / (24 * 30))  # $500/month for 1Gbps port
        dms_instance_cost = 0.34 * (transfer_time_hours / 24)
        setup_cost = 8000  # DX setup costs
        
        total_cost = data_transfer_cost + dx_port_cost + dms_instance_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': dx_port_cost + dms_instance_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 95,
            'reliability_score': 95,
            'security_score': 85,
            'complexity_score': 60
        }
    
    def _calc_dx_datasync_vpc(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate DX + DataSync + VPC Endpoints pattern metrics"""
        
        # Transfer time calculation (with 90% efficiency)
        effective_bandwidth = bandwidth_mbps * 0.90
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.0125  # DataSync per GB
        dx_port_cost = 500 * (transfer_time_hours / (24 * 30))
        vpc_endpoint_cost = 22.5 * (transfer_time_hours / (24 * 30))  # $22.50/month per endpoint
        datasync_agent_cost = 0.048 * (transfer_time_hours / 24)  # m5.large for agent
        setup_cost = 12000  # Higher setup due to VPC endpoints configuration
        
        total_cost = data_transfer_cost + dx_port_cost + vpc_endpoint_cost + datasync_agent_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': dx_port_cost + vpc_endpoint_cost + datasync_agent_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 90,
            'reliability_score': 90,
            'security_score': 95,
            'complexity_score': 85
        }
    
    def _calc_vpn_dms(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate VPN + DMS pattern metrics"""
        
        # Transfer time calculation (with 80% efficiency for VPN)
        effective_bandwidth = min(bandwidth_mbps * 0.8, 1250)  # VPN Gateway limit ~1.25 Gbps
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.09  # Same as internet
        vpn_gateway_cost = 36 * (transfer_time_hours / (24 * 30))  # $36/month
        dms_instance_cost = 0.34 * (transfer_time_hours / 24)
        setup_cost = 3000  # VPN setup costs
        
        total_cost = data_transfer_cost + vpn_gateway_cost + dms_instance_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': vpn_gateway_cost + dms_instance_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 80,
            'reliability_score': 85,
            'security_score': 90,
            'complexity_score': 50
        }
    
    def _calc_hybrid_snowball(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate Snowball + DMS hybrid pattern metrics"""
        
        # Assume 80% via Snowball, 20% via DMS for ongoing sync
        snowball_data_gb = data_size_gb * 0.8
        dms_data_gb = data_size_gb * 0.2
        
        # Snowball calculations
        snowball_devices = max(1, int(snowball_data_gb / (80 * 1024)))  # 80TB per device
        snowball_cost = snowball_devices * 250  # $250 per device
        snowball_shipping_days = 5  # Average shipping time
        
        # DMS for ongoing sync
        effective_bandwidth = bandwidth_mbps * 0.7
        dms_transfer_hours = (dms_data_gb * 8 * 1024) / (effective_bandwidth * 3600)
        dms_cost = dms_data_gb * 0.09 + 0.34 * (dms_transfer_hours / 24)
        
        setup_cost = 15000  # Complex orchestration setup
        
        total_transfer_days = snowball_shipping_days + (dms_transfer_hours / 24)
        total_cost = snowball_cost + dms_cost + setup_cost
        
        return {
            'transfer_time_hours': total_transfer_days * 24,
            'transfer_time_days': total_transfer_days,
            'data_transfer_cost': snowball_cost,
            'infrastructure_cost': dms_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 95,  # High for Snowball portion
            'reliability_score': 90,
            'security_score': 85,
            'complexity_score': 95
        }
    
    def _calc_multipath_redundant(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate multi-path redundant pattern metrics"""
        
        # Use both DX and VPN, with load balancing
        effective_bandwidth = bandwidth_mbps * 1.2  # 20% boost from dual paths
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
              # Cost calculations (combination of DX and VPN costs)
        dx_costs = self._calc_dx_dms(data_size_gb, bandwidth_mbps, region)
        vpn_costs = self._calc_vpn_dms(data_size_gb, bandwidth_mbps, region)
        
        # Take higher infrastructure costs + additional setup
        data_transfer_cost = min(dx_costs['data_transfer_cost'], vpn_costs['data_transfer_cost'])
        infrastructure_cost = dx_costs['infrastructure_cost'] + vpn_costs['infrastructure_cost']
        setup_cost = 20000  # Complex dual-path setup
        
        total_cost = data_transfer_cost + infrastructure_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': infrastructure_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 98,
            'reliability_score': 99,
            'security_score': 95,
            'complexity_score': 100
        }
    
    def _generate_recommendations(self, results: Dict, migration_params: Dict) -> Dict:
        """Generate AI-style recommendations for network transfer patterns"""
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        budget_level = migration_params.get('budget_constraints', 'medium')
        
        # Score each pattern
        pattern_scores = {}
        
        for pattern_id, metrics in results.items():
            if pattern_id == 'recommendations':
                continue
                
            # Calculate composite score based on multiple factors
            cost_score = self._score_cost(metrics['total_cost'], budget_level)
            time_score = self._score_time(metrics['transfer_time_days'], timeline_weeks)
            security_score = metrics['security_score']
            reliability_score = metrics['reliability_score']
            complexity_penalty = 100 - metrics['complexity_score']
            
            # Weighted scoring
            composite_score = (
                cost_score * 0.25 +
                time_score * 0.25 +
                security_score * 0.20 +
                reliability_score * 0.20 +
                complexity_penalty * 0.10
            )
            
            pattern_scores[pattern_id] = {
                'composite_score': composite_score,
                'cost_score': cost_score,
                'time_score': time_score,
                'metrics': metrics
            }
        
        # Sort by composite score
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        # Generate recommendations
        best_pattern_id, best_scores = sorted_patterns[0]
        pattern_info = self.transfer_patterns[best_pattern_id]
        
        return {
            'primary_recommendation': {
                'pattern_id': best_pattern_id,
                'pattern_name': pattern_info['name'],
                'description': pattern_info['description'],
                'score': best_scores['composite_score'],
                'reasoning': f"Best overall fit for {data_size_gb:,}GB migration within {timeline_weeks} weeks",
                'implementation_steps': pattern_info['components'],
                'key_considerations': pattern_info['pros'][:3]
            },
            'alternative_options': [
                {
                    'pattern_name': self.transfer_patterns[pattern_id]['name'],
                    'score': scores['composite_score'],
                    'best_for': self.transfer_patterns[pattern_id]['use_cases'][0]
                }
                for pattern_id, scores in sorted_patterns[1:3]
            ],
            'cost_optimization': [
                "Consider Direct Connect for large data volumes to reduce transfer costs",
                "Implement data compression to reduce transfer time",
                "Schedule transfers during off-peak hours"
            ],
            'risk_considerations': [
                "Test connectivity before full migration",
                "Plan for rollback scenarios",
                "Monitor bandwidth utilization during transfer"
            ]
        }
    
    def _score_cost(self, total_cost: float, budget_level: str) -> float:
        """Score based on cost relative to budget constraints"""
        budget_thresholds = {'low': 10000, 'medium': 50000, 'high': 200000}
        threshold = budget_thresholds.get(budget_level, 50000)
        
        if total_cost <= threshold * 0.5:
            return 100
        elif total_cost <= threshold:
            return 80
        elif total_cost <= threshold * 1.5:
            return 60
        else:
            return 40
    
    def _score_time(self, transfer_days: float, timeline_weeks: int) -> float:
        """Score based on time relative to migration timeline"""
        available_days = timeline_weeks * 7 * 0.3  # 30% of timeline for data transfer
        
        if transfer_days <= available_days * 0.5:
            return 100
        elif transfer_days <= available_days:
            return 80
        else:
            return 40

# ===========================
# MIGRATION ANALYZER
# ===========================

class MigrationAnalyzer:
    """Enhanced migration analyzer with vROps and network analysis"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.anthropic_api_key = anthropic_api_key
        self.pricing_api = self._initialize_pricing()
    
    def _initialize_pricing(self):
        """Initialize pricing data"""
        return {
            'us-east-1': {
                'postgres': {
                    'db.t3.medium': {'hourly': 0.102, 'storage_gb': 0.115},
                    'db.t3.large': {'hourly': 0.204, 'storage_gb': 0.115},
                    'db.r5.large': {'hourly': 0.24, 'storage_gb': 0.115},
                    'db.r5.xlarge': {'hourly': 0.48, 'storage_gb': 0.115},
                    'db.r5.2xlarge': {'hourly': 0.96, 'storage_gb': 0.115}
                }
            }
        }
    
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs.get('cpu_cores', specs.get('cpu_cores_allocated', 4))
            ram_gb = specs.get('ram_gb', specs.get('memory_allocated_gb', 16))
            storage_gb = specs.get('storage_gb', specs.get('storage_allocated_gb', 500))
            
            # Determine environment type
            environment_type = self._categorize_environment(env_name)
            
            # Calculate instance class
            instance_class = self._calculate_instance_class(cpu_cores, ram_gb, environment_type)
            
            # Multi-AZ recommendation
            multi_az = environment_type in ['production', 'staging']
            
            recommendations[env_name] = {
                'environment_type': environment_type,
                'instance_class': instance_class,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'multi_az': multi_az,
                'daily_usage_hours': specs.get('daily_usage_hours', 24),
                'peak_connections': specs.get('peak_connections', specs.get('max_database_connections', 100))
            }
        
        return recommendations
    
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs"""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            env_costs = self._calculate_environment_cost(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
        
        # Migration service costs
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks
        
        # Data transfer costs
        data_transfer_cost = data_size_gb * 0.09  # Internet transfer
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': data_transfer_cost,
            'professional_services': ps_cost,
            'total': dms_instance_cost + data_transfer_cost + ps_cost
        }
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs
        }
    
    def _categorize_environment(self, env_name: str) -> str:
        """Categorize environment type from name"""
        env_lower = env_name.lower()
        if any(term in env_lower for term in ['prod', 'production', 'prd']):
            return 'production'
        elif any(term in env_lower for term in ['stag', 'staging', 'preprod']):
            return 'staging'
        elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
            return 'development'
        return 'production'
    
    def _calculate_instance_class(self, cpu_cores: int, ram_gb: int, env_type: str) -> str:
        """Calculate appropriate instance class"""
        
        if cpu_cores <= 2 and ram_gb <= 8:
            instance_class = 'db.t3.medium'
        elif cpu_cores <= 4 and ram_gb <= 16:
            instance_class = 'db.t3.large'
        elif cpu_cores <= 8 and ram_gb <= 32:
            instance_class = 'db.r5.large'
        elif cpu_cores <= 16 and ram_gb <= 64:
            instance_class = 'db.r5.xlarge'
        else:
            instance_class = 'db.r5.2xlarge'
        
        return instance_class
    
    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate cost for a single environment"""
        
        # Get pricing
        pricing_data = self.pricing_api.get(region, {}).get(target_engine, {})
        instance_pricing = pricing_data.get(rec['instance_class'], {'hourly': 0.5, 'storage_gb': 0.115})
        
        # Calculate monthly hours
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        # Instance cost
        instance_cost = instance_pricing['hourly'] * monthly_hours
        
        # Storage cost
        storage_cost = rec['storage_gb'] * instance_pricing['storage_gb']
        
        # Backup cost (estimate 20% of storage)
        backup_cost = storage_cost * 0.2
        
        # Total monthly cost
        total_monthly = instance_cost + storage_cost + backup_cost
        
        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly
        }

# ===========================
# STREAMLIT UI FUNCTIONS
# ===========================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'environment_specs': {},
        'migration_params': {},
        'analysis_results': None,
        'recommendations': None,
        'vrops_analysis': None,
        'vrops_analyzer': None,
        'network_analysis': None,
        'transfer_analysis': None,
        'risk_assessment': None,
        'ai_insights': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def show_migration_configuration():
    """Show migration configuration interface"""
    
    st.markdown("## 🔧 Migration Configuration")
    
    # Source and target engine selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Source Database")
        source_engine = st.selectbox(
            "Source Engine",
            options=list(DatabaseEngine.ENGINES.keys()),
            format_func=lambda x: DatabaseEngine.ENGINES[x]['name'],
            key="source_engine"
        )
        
        if source_engine:
            st.info(f"**Features:** {', '.join(DatabaseEngine.ENGINES[source_engine]['features'])}")
    
    with col2:
        st.markdown("### 📤 Target AWS Database")
        
        if source_engine:
            target_options = DatabaseEngine.ENGINES[source_engine]['aws_targets']
            target_engine = st.selectbox(
                "Target Engine",
                options=target_options,
                format_func=lambda x: DatabaseEngine.ENGINES.get(x, {'name': x.title()})['name'] if x in DatabaseEngine.ENGINES else x.replace('-', ' ').title(),
                key="target_engine"
            )
            
            if target_engine:
                migration_type = DatabaseEngine.get_migration_type(source_engine, target_engine)
                st.markdown(f"**Migration Type:** {migration_type.title()}")
    
    # Migration parameters
    st.markdown("### ⚙️ Migration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💾 Data Configuration")
        data_size_gb = st.number_input("Total Data Size (GB)", min_value=1, max_value=100000, value=1000)
        num_applications = st.number_input("Connected Applications", min_value=1, max_value=50, value=3)
    
    with col2:
        st.markdown("#### ⏱️ Timeline & Resources")
        migration_timeline_weeks = st.slider("Migration Timeline (weeks)", min_value=4, max_value=52, value=12)
        team_size = st.number_input("Team Size", min_value=2, max_value=20, value=5)
    
    with col3:
        st.markdown("#### 🌐 Infrastructure")
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"], index=0)
        bandwidth_mbps = st.selectbox("Bandwidth (Mbps)", [100, 1000, 10000], index=1)
        migration_budget = st.number_input("Migration Budget ($)", min_value=10000, max_value=5000000, value=500000)
    
    # AI Configuration
    st.markdown("### 🤖 AI Integration")
    
    anthropic_api_key = st.text_input(
        "Anthropic API Key (Optional)",
        type="password",
        help="Provide your Anthropic API key for AI-powered insights"
    )
    
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        st.session_state.migration_params = {
            'source_engine': source_engine,
            'target_engine': target_engine,
            'data_size_gb': data_size_gb,
            'num_applications': num_applications,
            'migration_timeline_weeks': migration_timeline_weeks,
            'team_size': team_size,
            'region': region,
            'bandwidth_mbps': bandwidth_mbps,
            'migration_budget': migration_budget,
            'anthropic_api_key': anthropic_api_key
        }
        
        st.success("✅ Configuration saved! Proceed to Environment Setup.")
        st.balloons()

def show_environment_setup():
    """Show environment setup interface with vROps support"""
    
    st.markdown("## 📊 Environment Configuration")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Initialize vROps analyzer
    if 'vrops_analyzer' not in st.session_state:
        st.session_state.vrops_analyzer = VRopsMetricsAnalyzer()
    
    analyzer = st.session_state.vrops_analyzer
    
    # Configuration method selection
    st.markdown("### 🔧 Configuration Method")
    
    config_method = st.radio(
        "Choose configuration method:",
        [
            "📊 vROps Metrics Import", 
            "📝 Manual Detailed Entry",
            "📁 Bulk CSV Upload",
            "🔄 Simple Configuration"
        ],
        horizontal=True
    )
    
    if config_method == "📊 vROps Metrics Import":
        show_vrops_import_interface(analyzer)
    elif config_method == "📝 Manual Detailed Entry":
        show_manual_detailed_entry(analyzer)
    elif config_method == "📁 Bulk CSV Upload":
        show_bulk_upload_interface()
    else:
        show_simple_configuration()

def show_vrops_import_interface(analyzer: VRopsMetricsAnalyzer):
    """Show vROps metrics import interface"""
    
    st.markdown("### 📊 vROps Metrics Import")
    
    # Sample vROps export template
    with st.expander("📋 Download vROps Export Template", expanded=False):
        st.markdown("""
        **vROps Data Collection Instructions:**
        
        1. **Export Performance Data** from vROps for your database VMs
        2. **Time Period**: Minimum 30 days, recommended 90 days
        3. **Metrics to Include**: Use the template below or export all available metrics
        4. **Format**: CSV export with hourly or daily aggregation
        """)
        
        # Generate sample template
        sample_metrics = create_vrops_sample_template()
        csv_data = sample_metrics.to_csv(index=False)
        
        st.dataframe(sample_metrics.head(), use_container_width=True)
        
        st.download_button(
            label="📥 Download vROps Template (CSV)",
            data=csv_data,
            file_name="vrops_metrics_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload vROps Export File",
        type=['csv', 'xlsx'],
        help="Upload your vROps performance metrics export"
    )
    
    if uploaded_file is not None:
        try:
            # Load the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show data preview
            st.markdown("#### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data mapping interface
            st.markdown("#### 🔗 Map vROps Metrics to Standard Fields")
            
            processed_environments = process_vrops_data(df, analyzer)
            
            if processed_environments:
                st.session_state.environment_specs = processed_environments
                st.success(f"✅ Successfully processed {len(processed_environments)} environments!")
                
                # Show processed summary
                show_vrops_processing_summary(processed_environments, analyzer)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def create_vrops_sample_template() -> pd.DataFrame:
    """Create sample vROps template"""
    
    sample_data = {
        'VM_Name': ['DB-PROD-01', 'DB-PROD-02', 'DB-QA-01', 'DB-DEV-01'],
        'Environment': ['Production', 'Production', 'QA', 'Development'],
        'Max_CPU_Usage_Percent': [85.2, 78.9, 45.6, 32.1],
        'Avg_CPU_Usage_Percent': [65.4, 58.7, 28.9, 18.5],
        'CPU_Cores_Allocated': [16, 12, 8, 4],
        'Max_Memory_Usage_Percent': [78.9, 82.1, 45.2, 38.7],
        'Avg_Memory_Usage_Percent': [68.5, 71.3, 35.8, 28.9],
        'Memory_Allocated_GB': [64, 48, 32, 16],
        'Max_IOPS_Total': [8500, 6200, 2100, 800],
        'Avg_IOPS_Total': [5200, 3800, 1200, 450],
        'Max_Disk_Latency_ms': [12.5, 15.8, 8.2, 6.1],
        'Avg_Disk_Latency_ms': [8.9, 11.2, 5.4, 3.8],
        'Storage_Allocated_GB': [2000, 1500, 500, 200],
        'Storage_Used_GB': [1600, 1200, 350, 120],
        'Database_Size_GB': [1200, 900, 250, 80],
        'Max_Database_Connections': [450, 320, 125, 45],
        'Application_Type': ['OLTP', 'OLTP', 'Mixed', 'OLTP']
    }
    
    return pd.DataFrame(sample_data)

def process_vrops_data(df: pd.DataFrame, analyzer: VRopsMetricsAnalyzer) -> Dict:
    """Process uploaded vROps data into environment specifications"""
    
    st.markdown("##### 🔗 Column Mapping")
    
    # Create mapping interface
    mappings = {}
    available_columns = [''] + list(df.columns)
    
    # Key mappings in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Environment Identification**")
        mappings['vm_name'] = st.selectbox("VM/Server Name", available_columns, key="vm_name_col")
        mappings['environment'] = st.selectbox("Environment Name", available_columns, key="env_name_col")
        
        st.markdown("**CPU Metrics**")
        mappings['max_cpu_usage_percent'] = st.selectbox("Max CPU Usage %", available_columns, key="max_cpu_col")
        mappings['avg_cpu_usage_percent'] = st.selectbox("Avg CPU Usage %", available_columns, key="avg_cpu_col")
        mappings['cpu_cores_allocated'] = st.selectbox("CPU Cores", available_columns, key="cpu_cores_col")
    
    with col2:
        st.markdown("**Memory Metrics**")
        mappings['max_memory_usage_percent'] = st.selectbox("Max Memory Usage %", available_columns, key="max_mem_col")
        mappings['memory_allocated_gb'] = st.selectbox("Memory Allocated GB", available_columns, key="mem_alloc_col")
        
        st.markdown("**Storage Metrics**")
        mappings['max_iops_total'] = st.selectbox("Max IOPS", available_columns, key="max_iops_col")
        mappings['storage_allocated_gb'] = st.selectbox("Storage Allocated GB", available_columns, key="storage_col")
        mappings['database_size_gb'] = st.selectbox("Database Size GB", available_columns, key="db_size_col")
    
    if st.button("🔄 Process vROps Data", type="primary"):
        
        # Validate required mappings
        required_fields = ['vm_name', 'max_cpu_usage_percent', 'cpu_cores_allocated', 'memory_allocated_gb']
        
        missing_fields = [field for field in required_fields if not mappings.get(field)]
        
        if missing_fields:
            st.error(f"❌ Please map required fields: {', '.join(missing_fields)}")
            return None
        
        # Process the data
        try:
            environments = {}
            
            for _, row in df.iterrows():
                # Get environment name
                vm_name = str(row[mappings['vm_name']]) if mappings['vm_name'] else 'Unknown'
                env_name = str(row[mappings['environment']]) if mappings['environment'] else vm_name
                
                # Build metrics dictionary
                env_metrics = {}
                
                for metric_key, column_name in mappings.items():
                    if column_name and column_name in df.columns:
                        value = row[column_name]
                        if pd.notna(value):
                            env_metrics[metric_key] = float(value) if isinstance(value, (int, float)) else value
                
                # Add default values for compatibility
                env_metrics.setdefault('application_type', 'Mixed')
                env_metrics.setdefault('cpu_cores', env_metrics.get('cpu_cores_allocated', 4))
                env_metrics.setdefault('ram_gb', env_metrics.get('memory_allocated_gb', 16))
                env_metrics.setdefault('storage_gb', env_metrics.get('storage_allocated_gb', 500))
                
                environments[env_name] = env_metrics
            
            return environments
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            return None
    
    return None

def show_vrops_processing_summary(environments: Dict, analyzer: VRopsMetricsAnalyzer):
    """Show summary of processed vROps data"""
    
    st.markdown("#### 📊 Processing Summary")
    
    # Environment overview
    env_summary = []
    for env_name, metrics in environments.items():
        env_summary.append({
            'Environment': env_name,
            'CPU Cores': metrics.get('cpu_cores_allocated', 'N/A'),
            'Memory GB': metrics.get('memory_allocated_gb', 'N/A'),
            'Max CPU %': f"{metrics.get('max_cpu_usage_percent', 0):.1f}%",
            'Max IOPS': metrics.get('max_iops_total', 'N/A'),
            'DB Size GB': metrics.get('database_size_gb', 'N/A')
        })
    
    summary_df = pd.DataFrame(env_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    # Run analysis
    if st.button("🚀 Analyze vROps Metrics", type="primary"):
        with st.spinner("🔄 Analyzing performance metrics..."):
            analysis_results = analyzer.analyze_vrops_metrics(environments)
            st.session_state.vrops_analysis = analysis_results
            
            st.success("✅ vROps analysis complete!")
            show_vrops_analysis_summary(analysis_results)

def show_vrops_analysis_summary(analysis_results: Dict):
    """Show summary of vROps analysis results"""
    
    st.markdown("#### 🎯 Analysis Results Summary")
    
    # Performance health overview
    col1, col2, col3 = st.columns(3)
    
    # Calculate overall health scores
    health_scores = []
    for env_name, analysis in analysis_results.items():
        if env_name != 'overall_recommendations' and isinstance(analysis, dict):
            scores = analysis.get('performance_scores', {})
            health_scores.append(scores.get('overall_health', 0))
    
    avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
    
    with col1:
        st.metric("Overall Health Score", f"{avg_health:.1f}/100")
    
    with col2:
        at_risk_envs = len([score for score in health_scores if score < 70])
        st.metric("Environments at Risk", at_risk_envs)
    
    with col3:
        total_envs = len([k for k in analysis_results.keys() if k != 'overall_recommendations'])
        st.metric("Total Environments", total_envs)
    
    # Top recommendations
    st.markdown("#### 💡 Key Recommendations")
    
    for env_name, analysis in analysis_results.items():
        if env_name != 'overall_recommendations' and isinstance(analysis, dict):
            recommendations = analysis.get('instance_recommendations', [])
            if recommendations:
                top_rec = recommendations[0]
                st.markdown(f"**{env_name}:** {top_rec['instance_type']} - Fit Score: {top_rec['fit_score']:.1f}")

                      
def show_manual_detailed_entry(analyzer: VRopsMetricsAnalyzer):
    """Show manual detailed entry interface"""
    
    st.markdown("### 📝 Manual Detailed Entry")
    
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=2)
    
    environments = {}
    
    for i in range(num_environments):
        with st.expander(f"🏢 Environment {i+1} - Detailed Configuration", expanded=i == 0):
            env_name = st.text_input(f"Environment Name", value=f"Environment_{i+1}", key=f"detailed_env_name_{i}")
            
            # Environment type
            env_type = st.selectbox(
                "Environment Type",
                ["Production", "Staging", "QA", "Development"],
                key=f"env_type_{i}"
            )
            
            # Create tabs for different metric categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["💻 CPU", "🧠 Memory", "💾 Storage", "🌐 Network", "📊 Database"])
            
            env_metrics = {'environment_type': env_type}
            
            with tab1:
                show_cpu_metrics_input(env_metrics, i)
            
            with tab2:
                show_memory_metrics_input(env_metrics, i)
            
            with tab3:
                show_storage_metrics_input(env_metrics, i)
            
            with tab4:
                show_network_metrics_input(env_metrics, i)
            
            with tab5:
                show_database_metrics_input(env_metrics, i)
            
            environments[env_name] = env_metrics
    
    if st.button("💾 Save Detailed Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environments
        st.success("✅ Detailed environment configuration saved!")
        
        # Run analysis
        with st.spinner("🔄 Analyzing detailed metrics..."):
            analysis_results = analyzer.analyze_vrops_metrics(environments)
            st.session_state.vrops_analysis = analysis_results
            
        st.success("✅ Analysis complete! Check the Results Dashboard for detailed insights.")

def show_cpu_metrics_input(env_metrics: Dict, env_index: int):
    """Show CPU metrics input interface"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_metrics['cpu_cores_allocated'] = st.number_input(
            "CPU Cores Allocated", min_value=1, max_value=128, value=8, key=f"cpu_cores_{env_index}"
        )
        env_metrics['max_cpu_usage_percent'] = st.slider(
            "Max CPU Usage %", min_value=0, max_value=100, value=75, key=f"max_cpu_{env_index}"
        )
        env_metrics['avg_cpu_usage_percent'] = st.slider(
            "Average CPU Usage %", min_value=0, max_value=100, value=50, key=f"avg_cpu_{env_index}"
        )
    
    with col2:
        env_metrics['cpu_ready_time_ms'] = st.number_input(
            "CPU Ready Time (ms)", min_value=0, max_value=50000, value=1000, key=f"cpu_ready_{env_index}",
            help="Time VM waited for CPU resources"
        )
        env_metrics['cpu_costop_ms'] = st.number_input(
            "CPU Co-stop Time (ms)", min_value=0, max_value=10000, value=0, key=f"cpu_costop_{env_index}",
            help="Time multi-vCPU VM waited for all CPUs"
        )

def show_memory_metrics_input(env_metrics: Dict, env_index: int):
    """Show memory metrics input interface"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_metrics['memory_allocated_gb'] = st.number_input(
            "Memory Allocated (GB)", min_value=1, max_value=1024, value=32, key=f"mem_allocated_{env_index}"
        )
        env_metrics['max_memory_usage_percent'] = st.slider(
            "Max Memory Usage %", min_value=0, max_value=100, value=80, key=f"max_mem_{env_index}"
        )
        env_metrics['avg_memory_usage_percent'] = st.slider(
            "Average Memory Usage %", min_value=0, max_value=100, value=60, key=f"avg_mem_{env_index}"
        )
    
    with col2:
        env_metrics['memory_balloon_gb'] = st.number_input(
            "Memory Balloon (GB)", min_value=0.0, max_value=100.0, value=0.0, key=f"mem_balloon_{env_index}",
            help="Ballooned memory indicates over-commitment"
        )
        env_metrics['memory_swapped_gb'] = st.number_input(
            "Memory Swapped (GB)", min_value=0.0, max_value=100.0, value=0.0, key=f"mem_swapped_{env_index}",
            help="Swapped memory indicates severe memory pressure"
        )

def show_storage_metrics_input(env_metrics: Dict, env_index: int):
    """Show storage metrics input interface"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_metrics['storage_allocated_gb'] = st.number_input(
            "Storage Allocated (GB)", min_value=20, max_value=50000, value=1000, key=f"storage_allocated_{env_index}"
        )
        env_metrics['storage_used_gb'] = st.number_input(
            "Storage Used (GB)", min_value=10, max_value=50000, value=750, key=f"storage_used_{env_index}"
        )
        env_metrics['max_iops_total'] = st.number_input(
            "Max IOPS (Total)", min_value=100, max_value=100000, value=5000, key=f"max_iops_{env_index}"
        )
        env_metrics['avg_iops_total'] = st.number_input(
            "Average IOPS", min_value=50, max_value=50000, value=2500, key=f"avg_iops_{env_index}"
        )
    
    with col2:
        env_metrics['max_disk_latency_ms'] = st.number_input(
            "Max Disk Latency (ms)", min_value=0.1, max_value=1000.0, value=15.0, key=f"max_latency_{env_index}"
        )
        env_metrics['avg_disk_latency_ms'] = st.number_input(
            "Average Disk Latency (ms)", min_value=0.1, max_value=500.0, value=8.0, key=f"avg_latency_{env_index}"
        )
        env_metrics['max_disk_throughput_mbps'] = st.number_input(
            "Max Throughput (MB/s)", min_value=10, max_value=10000, value=250, key=f"max_throughput_{env_index}"
        )

def show_network_metrics_input(env_metrics: Dict, env_index: int):
    """Show network metrics input interface"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_metrics['max_network_throughput_mbps'] = st.number_input(
            "Max Network Throughput (Mbps)", min_value=10, max_value=10000, value=200, key=f"max_net_throughput_{env_index}"
        )
        env_metrics['avg_network_throughput_mbps'] = st.number_input(
            "Avg Network Throughput (Mbps)", min_value=5, max_value=5000, value=100, key=f"avg_net_throughput_{env_index}"
        )
    
    with col2:
        env_metrics['network_latency_ms'] = st.number_input(
            "Network Latency (ms)", min_value=0.1, max_value=1000.0, value=5.0, key=f"net_latency_{env_index}"
        )
        env_metrics['network_packet_loss_percent'] = st.number_input(
            "Packet Loss %", min_value=0.0, max_value=10.0, value=0.0, key=f"packet_loss_{env_index}"
        )

def show_database_metrics_input(env_metrics: Dict, env_index: int):
    """Show database metrics input interface"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_metrics['database_size_gb'] = st.number_input(
            "Database Size (GB)", min_value=1, max_value=100000, value=500, key=f"db_size_{env_index}"
        )
        env_metrics['max_database_connections'] = st.number_input(
            "Max Database Connections", min_value=10, max_value=10000, value=300, key=f"max_connections_{env_index}"
        )
        env_metrics['avg_database_connections'] = st.number_input(
            "Avg Database Connections", min_value=5, max_value=5000, value=150, key=f"avg_connections_{env_index}"
        )
        env_metrics['application_type'] = st.selectbox(
            "Application Type", ["OLTP", "OLAP", "Mixed", "Data Warehouse"], key=f"app_type_{env_index}"
        )
    
    with col2:
        env_metrics['max_transaction_rate_per_sec'] = st.number_input(
            "Max Transactions/sec", min_value=1, max_value=100000, value=1000, key=f"max_tps_{env_index}"
        )
        env_metrics['avg_query_response_time_ms'] = st.number_input(
            "Avg Query Response Time (ms)", min_value=1, max_value=10000, value=100, key=f"avg_response_{env_index}"
        )
        env_metrics['buffer_cache_hit_ratio_percent'] = st.slider(
            "Buffer Cache Hit Ratio %", min_value=50, max_value=100, value=95, key=f"cache_hit_{env_index}"
        )

def show_bulk_upload_interface():
    """Show bulk upload interface for environments"""
    
    st.markdown("### 📁 Bulk Environment Upload")
    
    # Sample template
    with st.expander("📋 Download Sample Template", expanded=False):
        sample_data = {
            'environment': ['Development', 'QA', 'SQA', 'Production'],
            'cpu_cores': [4, 8, 16, 32],
            'ram_gb': [16, 32, 64, 128],
            'storage_gb': [100, 500, 1000, 2000],
            'daily_usage_hours': [8, 12, 16, 24],
            'peak_connections': [20, 50, 100, 500]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        csv_data = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_data,
            file_name="environment_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file with environment specifications"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['environment', 'cpu_cores', 'ram_gb', 'storage_gb']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # Process environments
            environment_specs = {}
            for _, row in df.iterrows():
                env_name = str(row['environment']).strip()
                environment_specs[env_name] = {
                    'cpu_cores': int(row['cpu_cores']),
                    'ram_gb': int(row['ram_gb']),
                    'storage_gb': int(row['storage_gb']),
                    'daily_usage_hours': int(row.get('daily_usage_hours', 24)),
                    'peak_connections': int(row.get('peak_connections', 100))
                }
            
            st.session_state.environment_specs = environment_specs
            st.success(f"✅ Successfully loaded {len(environment_specs)} environments!")
            
            # Display loaded data
            st.markdown("#### 📊 Loaded Environments")
            display_df = pd.DataFrame.from_dict(environment_specs, orient='index')
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_simple_configuration():
    """Show simple configuration for backward compatibility"""
    
    st.markdown("### 🔄 Simple Configuration")
    
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    cols = st.columns(min(num_environments, 3))
    
    for i in range(num_environments):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            with st.expander(f"🏢 Environment {i+1}", expanded=True):
                env_name = st.text_input(
                    "Environment Name",
                    value=default_names[i] if i < len(default_names) else f"Environment_{i+1}",
                    key=f"env_name_{i}"
                )
                
                cpu_cores = st.number_input(
                    "CPU Cores",
                    min_value=1, max_value=128,
                    value=[4, 8, 16, 32][min(i, 3)],
                    key=f"cpu_{i}"
                )
                
                ram_gb = st.number_input(
                    "RAM (GB)",
                    min_value=4, max_value=1024,
                    value=[16, 32, 64, 128][min(i, 3)],
                    key=f"ram_{i}"
                )
                
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"storage_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"usage_{i}"
                )
                
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"connections_{i}"
                )
                
                environment_specs[env_name] = {
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'daily_usage_hours': daily_usage_hours,
                    'peak_connections': peak_connections
                }
    
    if st.button("💾 Save Environment Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Environment configuration saved!")
        
        # Display summary
        st.markdown("#### 📊 Configuration Summary")
        summary_df = pd.DataFrame.from_dict(environment_specs, orient='index')
        st.dataframe(summary_df, use_container_width=True)

def show_network_transfer_analysis():
    """Show network transfer analysis interface"""
    
    st.markdown("## 🌐 Network Transfer Analysis")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Initialize network analyzer
    if 'network_analyzer' not in st.session_state:
        st.session_state.network_analyzer = NetworkTransferAnalyzer()
    
    analyzer = st.session_state.network_analyzer
    
    # Network-specific parameters
    st.markdown("### 🔧 Network Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📶 Connectivity")
        available_bandwidth = st.selectbox(
            "Available Bandwidth",
            [100, 500, 1000, 10000],
            index=2,
            format_func=lambda x: f"{x} Mbps"
        )
        
        has_direct_connect = st.checkbox("Direct Connect Available", value=False)
        has_vpn_capability = st.checkbox("VPN Capability", value=True)
    
    with col2:
        st.markdown("#### 🔒 Security Requirements")
        security_level = st.selectbox(
            "Security Requirements",
            ["standard", "high", "very_high"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        compliance_requirements = st.multiselect(
            "Compliance Requirements",
            ["HIPAA", "PCI-DSS", "SOX", "GDPR", "None"],
            default=["None"]
        )
    
    with col3:
        st.markdown("#### 💰 Budget Constraints")
        budget_level = st.selectbox(
            "Budget Level",
            ["low", "medium", "high"],
            index=1,
            format_func=lambda x: x.title()
        )
        
        max_setup_cost = st.number_input(
            "Maximum Setup Cost ($)",
            min_value=1000,
            max_value=100000,
            value=25000
        )
    
    # Update migration params with network-specific settings
    network_params = st.session_state.migration_params.copy()
    network_params.update({
        'bandwidth_mbps': available_bandwidth,
        'has_direct_connect': has_direct_connect,
        'has_vpn_capability': has_vpn_capability,
        'security_requirements': security_level,
        'compliance_requirements': compliance_requirements,
        'budget_constraints': budget_level,
        'max_setup_cost': max_setup_cost
    })
    
    # Run network analysis
    if st.button("🚀 Analyze Network Transfer Options", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing network transfer patterns..."):
            
            transfer_analysis = analyzer.calculate_transfer_analysis(network_params)
            st.session_state.transfer_analysis = transfer_analysis
            
            st.success("✅ Network analysis complete!")
    
    # Display results if available
    if hasattr(st.session_state, 'transfer_analysis') and st.session_state.transfer_analysis is not None:
        show_network_analysis_results()
    else:
        st.info("ℹ️ Run the network analysis to see results and recommendations.")

def show_network_analysis_results():
    """Display network analysis results"""
    
    if not hasattr(st.session_state, 'transfer_analysis') or st.session_state.transfer_analysis is None:
        st.error("No network analysis results available. Please run the network analysis first.")
        return
    
    transfer_analysis = st.session_state.transfer_analysis
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recommendations",
        "📊 Pattern Comparison", 
        "💰 Cost Analysis",
        "⏱️ Timeline Analysis"
    ])
    
    with tab1:
        show_network_recommendations(transfer_analysis)
    
    with tab2:
        show_pattern_comparison(transfer_analysis)
    
    with tab3:
        show_network_cost_analysis(transfer_analysis)
    
    with tab4:
        show_network_timeline_analysis(transfer_analysis)

def show_network_recommendations(transfer_analysis: Dict):
    """Show network recommendations"""
    
    st.markdown("### 🎯 Network Recommendations")
    
    recommendations = transfer_analysis.get('recommendations', {})
    
    if not recommendations:
        st.error("No recommendations available")
        return
    
    # Primary recommendation
    primary = recommendations.get('primary_recommendation', {})
    
    if primary:
        st.markdown(f"""
        <div class="ai-insight-card">
            <h3>🏆 Primary Recommendation: {primary['pattern_name']}</h3>
            <p><strong>Score:</strong> {primary['score']:.1f}/100</p>
            <p><strong>Description:</strong> {primary['description']}</p>
            <p><strong>Reasoning:</strong> {primary['reasoning']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Implementation details
        st.markdown("#### 📋 Implementation Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Implementation Steps:**")
            for i, step in enumerate(primary.get('implementation_steps', []), 1):
                st.markdown(f"{i}. {step}")
        
        with col2:
            st.markdown("**Key Benefits:**")
            for benefit in primary.get('key_considerations', []):
                st.markdown(f"• {benefit}")
    
    # Alternative options
    alternatives = recommendations.get('alternative_options', [])
    
    if alternatives:
        st.markdown("#### 🔄 Alternative Options")
        
        for i, alt in enumerate(alternatives, 1):
            with st.expander(f"Alternative {i}: {alt['pattern_name']} (Score: {alt['score']:.1f})"):
                st.markdown(f"**Best for:** {alt['best_for']}")

def show_pattern_comparison(transfer_analysis: Dict):
    """Show pattern comparison visualizations"""
    
    st.markdown("### 📊 Network Pattern Comparison")
    
    # Create comparison table
    comparison_data = []
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        analyzer = NetworkTransferAnalyzer()
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        comparison_data.append({
            'Pattern': pattern_info['name'],
            'Total Cost': f"${metrics['total_cost']:,.0f}",
            'Transfer Duration': f"{metrics['transfer_time_days']:.1f} days",
            'Setup Cost': f"${metrics['setup_cost']:,.0f}",
            'Reliability Score': f"{metrics['reliability_score']}/100",
            'Security Score': f"{metrics['security_score']}/100",
            'Complexity': pattern_info['complexity']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def show_network_cost_analysis(transfer_analysis: Dict):
    """Show detailed network cost analysis"""
    
    st.markdown("### 💰 Network Cost Analysis")
    
    analyzer = NetworkTransferAnalyzer()
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        with st.expander(f"💵 {pattern_info['name']} - Total Cost: ${metrics['total_cost']:,.0f}"):
             
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Setup Cost", f"${metrics['setup_cost']:,.0f}")
                st.metric("Data Transfer Cost", f"${metrics['data_transfer_cost']:,.0f}")
            
            with col2:
                st.metric("Infrastructure Cost", f"${metrics['infrastructure_cost']:,.0f}")
                st.metric("Total Cost", f"${metrics['total_cost']:,.0f}")
            
            with col3:
                data_size = st.session_state.migration_params.get('data_size_gb', 1000)
                cost_per_gb = metrics['total_cost'] / data_size
                st.metric("Cost per GB", f"${cost_per_gb:.2f}")

def show_network_timeline_analysis(transfer_analysis: Dict):
    """Show network timeline analysis"""
    
    st.markdown("### ⏱️ Timeline Analysis")
    
    # Create timeline comparison chart
    patterns = []
    durations = []
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        analyzer = NetworkTransferAnalyzer()
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        patterns.append(pattern_info['name'])
        durations.append(metrics['transfer_time_days'])
    
    # Show timeline table
    timeline_data = []
    for pattern, duration in zip(patterns, durations):
        timeline_data.append({
            'Pattern': pattern,
            'Duration (days)': f"{duration:.1f}",
            'Duration (hours)': f"{duration * 24:.0f}",
            'Fits in 30-day window': "✅ Yes" if duration <= 30 else "❌ No"
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)

def show_analysis_section():
    """Show analysis and recommendations section"""
    
    st.markdown("## 🚀 Migration Analysis & Recommendations")
    
    # Check prerequisites
    if not st.session_state.migration_params:
        st.error("❌ Migration configuration required")
        st.info("👆 Please complete the 'Migration Configuration' section first")
        return
    
    if not st.session_state.environment_specs:
        st.error("❌ Environment configuration required")
        st.info("👆 Please complete the 'Environment Setup' section first")
        return
    
    # Display current configuration
    st.markdown("### 📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        params = st.session_state.migration_params
        st.markdown(f"""
        **Migration Type:** {params['source_engine']} → {params['target_engine']}  
        **Data Size:** {params['data_size_gb']:,} GB  
        **Timeline:** {params['migration_timeline_weeks']} weeks  
        **Team Size:** {params['team_size']} members  
        **Budget:** ${params['migration_budget']:,}
        """)
    
    with col2:
        envs = st.session_state.environment_specs
        st.markdown(f"**Environments:** {len(envs)}")
        
        # Show first few environments
        count = 0
        for env_name, specs in envs.items():
            if count < 4:
                cpu_cores = specs.get('cpu_cores', specs.get('cpu_cores_allocated', 'N/A'))
                ram_gb = specs.get('ram_gb', specs.get('memory_allocated_gb', 'N/A'))
                st.markdown(f"• **{env_name}:** {cpu_cores} cores, {ram_gb} GB RAM")
                count += 1
        
        if len(envs) > 4:
            st.markdown(f"• ... and {len(envs) - 4} more environments")
    
    # Run analysis
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing migration requirements..."):
            run_migration_analysis()

def run_migration_analysis():
    """Run comprehensive migration analysis"""
    
    try:
        # Initialize analyzer
        anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
        analyzer = MigrationAnalyzer(anthropic_api_key)
        
        # Step 1: Calculate recommendations
        st.write("📊 Calculating instance recommendations...")
        recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
        st.session_state.recommendations = recommendations
        
        # Step 2: Calculate costs
        st.write("💰 Analyzing costs...")
        cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.analysis_results = cost_analysis
        
        # Step 3: Risk assessment
        st.write("⚠️ Assessing risks...")
        risk_assessment = create_risk_assessment(st.session_state.migration_params, recommendations)
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: AI insights (if available)
        if anthropic_api_key and ANTHROPIC_AVAILABLE:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = generate_ai_insights(cost_analysis, st.session_state.migration_params)
                st.session_state.ai_insights = ai_insights
                st.success("✅ AI insights generated")
            except Exception as e:
                st.warning(f"AI insights failed: {str(e)}")
                st.session_state.ai_insights = {'error': str(e)}
        else:
            st.info("ℹ️ Provide Anthropic API key for AI insights")
        
        st.success("✅ Analysis complete!")
        
        # Show summary
        show_analysis_summary()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")

def create_risk_assessment(migration_params: Dict, recommendations: Dict) -> Dict:
    """Create risk assessment based on migration parameters and recommendations"""
    
    # Calculate risk factors
    data_size_gb = migration_params.get('data_size_gb', 1000)
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    team_size = migration_params.get('team_size', 5)
    num_environments = len(recommendations)
    source_engine = migration_params.get('source_engine', 'unknown')
    target_engine = migration_params.get('target_engine', 'unknown')
    
    # Calculate individual risk scores (0-100)
    data_complexity_risk = min(80, (data_size_gb / 1000) * 20)  # Scale with data size
    timeline_risk = max(20, 100 - (timeline_weeks * 5))  # Less time = more risk
    team_risk = max(20, 100 - (team_size * 10))  # Smaller team = more risk
    engine_compatibility_risk = 60 if source_engine != target_engine else 20
    environment_complexity_risk = min(70, num_environments * 10)  # More envs = more risk
    
    # Overall risk calculation
    risk_scores = [data_complexity_risk, timeline_risk, team_risk, engine_compatibility_risk, environment_complexity_risk]
    overall_risk = sum(risk_scores) / len(risk_scores)
    
    # Determine risk level
    if overall_risk < 30:
        risk_level = {'level': 'Low', 'color': '#38a169', 'action': 'Standard monitoring sufficient'}
    elif overall_risk < 50:
        risk_level = {'level': 'Medium', 'color': '#d69e2e', 'action': 'Active monitoring recommended'}
    elif overall_risk < 70:
        risk_level = {'level': 'High', 'color': '#e53e3e', 'action': 'Immediate mitigation required'}
    else:
        risk_level = {'level': 'Critical', 'color': '#9f1239', 'action': 'Project review required'}
    
    # Generate mitigation strategies
    mitigation_strategies = []
    
    if data_complexity_risk > 50:
        mitigation_strategies.append({
            'risk': 'Data Complexity',
            'strategy': 'Implement phased migration approach with comprehensive data validation',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    if timeline_risk > 50:
        mitigation_strategies.append({
            'risk': 'Timeline Pressure',
            'strategy': 'Add parallel workstreams and consider extending timeline',
            'timeline': 'Immediate',
            'cost_impact': 'High'
        })
    
    if team_risk > 50:
        mitigation_strategies.append({
            'risk': 'Team Capacity',
            'strategy': 'Augment team with external expertise or AWS Professional Services',
            'timeline': '1-2 weeks',
            'cost_impact': 'High'
        })
    
    if engine_compatibility_risk > 40:
        mitigation_strategies.append({
            'risk': 'Engine Compatibility',
            'strategy': 'Use AWS Schema Conversion Tool and conduct thorough testing',
            'timeline': '2-4 weeks',
            'cost_impact': 'Medium'
        })
    
    return {
        'overall_score': overall_risk,
        'risk_level': risk_level,
        'technical_risks': {
            'data_complexity': data_complexity_risk,
            'engine_compatibility': engine_compatibility_risk,
            'environment_complexity': environment_complexity_risk
        },
        'business_risks': {
            'timeline_risk': timeline_risk,
            'team_capacity': team_risk
        },
        'mitigation_strategies': mitigation_strategies
    }

def generate_ai_insights(cost_analysis: Dict, migration_params: Dict) -> Dict:
    """Generate AI insights using Anthropic (if available)"""
    
    if not ANTHROPIC_AVAILABLE:
        return {
            'error': 'Anthropic library not available',
            'fallback_insights': get_fallback_insights(cost_analysis, migration_params)
        }
    
    try:
        client = anthropic.Anthropic(api_key=migration_params.get('anthropic_api_key'))
        
        # Prepare context
        context = f"""
        Migration Analysis Context:
        - Source: {migration_params.get('source_engine', 'Unknown')}
        - Target: {migration_params.get('target_engine', 'Unknown')}
        - Data Size: {migration_params.get('data_size_gb', 0):,} GB
        - Timeline: {migration_params.get('migration_timeline_weeks', 0)} weeks
        - Monthly Cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
        - Migration Cost: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}
        - Environments: {len(cost_analysis.get('environment_costs', {}))}
        """
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": f"Analyze this AWS database migration project and provide insights on cost optimization, risks, and recommendations. {context}"
            }]
        )
        
        return {
            'ai_analysis': message.content[0].text,
            'source': 'Claude AI',
            'success': True
        }
        
    except Exception as e:
        return {
            'error': f'AI analysis failed: {str(e)}',
            'fallback_insights': get_fallback_insights(cost_analysis, migration_params)
        }

def get_fallback_insights(cost_analysis: Dict, migration_params: Dict) -> Dict:
    """Generate fallback insights when AI is not available"""
    
    monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
    migration_cost = cost_analysis.get('migration_costs', {}).get('total', 0)
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    
    insights = {
        'cost_efficiency': f"Monthly AWS cost of ${monthly_cost:,.0f} is within typical range for enterprise migrations",
        'migration_strategy': f"Timeline of {timeline_weeks} weeks allows for proper testing and validation phases",
        'risk_assessment': "Migration complexity appears manageable with standard AWS best practices",
        'recommendations': [
            "Consider Reserved Instances for 30-40% cost savings",
            "Implement phased migration starting with development environments",
            "Use AWS DMS for minimal downtime data transfer",
            "Plan comprehensive testing at each migration phase"
        ]
    }
    
    return insights

def show_analysis_summary():
    """Show analysis summary"""
    
    st.markdown("#### 🎯 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    results = st.session_state.analysis_results
    
    with col1:
        st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
    
    with col2:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    
    with col3:
        if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
            risk_level = st.session_state.risk_assessment['risk_level']['level']
            st.metric("Risk Level", risk_level)
    
    st.info("📈 View detailed results in the 'Results Dashboard' section")

def show_results_dashboard():
    """Show comprehensive results dashboard"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run the migration analysis first.")
        return
    
    st.markdown("## 📊 Migration Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Cost Summary",
        "📊 vROps Analysis",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "🤖 AI Insights",
        "📅 Timeline"
    ])
    
    with tab1:
        show_cost_summary()
    
    with tab2:
        show_vrops_results_tab()
    
    with tab3:
        show_risk_assessment_tab()
    
    with tab4:
        show_environment_analysis_tab()
    
    with tab5:
        show_ai_insights_tab()
    
    with tab6:
        show_timeline_analysis_tab()

def show_cost_summary():
    """Show cost summary from analysis results"""
    
    results = st.session_state.analysis_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monthly_cost = results.get('monthly_aws_cost', 0)
        st.metric("Monthly AWS Cost", f"${monthly_cost:,.0f}")
    
    with col2:
        annual_cost = monthly_cost * 12
        st.metric("Annual AWS Cost", f"${annual_cost:,.0f}")
    
    with col3:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    
    with col4:
        total_cost = annual_cost + migration_cost
        st.metric("Total First Year", f"${total_cost:,.0f}")
    
    # Environment costs breakdown
    st.markdown("### 💰 Cost Breakdown by Environment")
    
    env_costs = results.get('environment_costs', {})
    if env_costs:
        for env_name, costs in env_costs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Instance Cost", f"${costs.get('instance_cost', 0):,.2f}/month")
                
                with col2:
                    st.metric("Storage Cost", f"${costs.get('storage_cost', 0):,.2f}/month")
                
                with col3:
                    st.metric("Total Monthly", f"${costs.get('total_monthly', 0):,.2f}")

def show_vrops_results_tab():
    """Show vROps analysis results in the dashboard"""
    
    if hasattr(st.session_state, 'vrops_analysis') and st.session_state.vrops_analysis:
        st.markdown("### 📊 vROps Performance Analysis")
        
        analysis_results = st.session_state.vrops_analysis
        
        # Performance health overview
        col1, col2, col3 = st.columns(3)
        
        # Calculate overall health scores
        health_scores = []
        env_count = 0
        
        for env_name, analysis in analysis_results.items():
            if env_name != 'overall_recommendations' and isinstance(analysis, dict):
                env_count += 1
                scores = analysis.get('performance_scores', {})
                health_scores.append(scores.get('overall_health', 0))
        
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
        
        with col1:
            st.metric("Overall Health Score", f"{avg_health:.1f}/100")
        
        with col2:
            at_risk_envs = len([score for score in health_scores if score < 70])
            st.metric("At Risk Environments", at_risk_envs)
        
        with col3:
            st.metric("Total Environments", env_count)
        
        # Environment details
        st.markdown("#### 🏢 Environment Performance Analysis")
        
        for env_name, analysis in analysis_results.items():
            if env_name != 'overall_recommendations' and isinstance(analysis, dict):
                with st.expander(f"📊 {env_name} Performance Details"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**CPU Analysis**")
                        cpu_analysis = analysis.get('cpu_analysis', {})
                        st.write(f"Max Usage: {cpu_analysis.get('max_usage_percent', 0):.1f}%")
                        st.write(f"Avg Usage: {cpu_analysis.get('avg_usage_percent', 0):.1f}%")
                    
                    with col2:
                        st.markdown("**Memory Analysis**")
                        memory_analysis = analysis.get('memory_analysis', {})
                        st.write(f"Max Usage: {memory_analysis.get('max_usage_percent', 0):.1f}%")
                        st.write(f"Allocated: {memory_analysis.get('allocated_gb', 0)} GB")
                    
                    with col3:
                        st.markdown("**Storage Analysis**")
                        storage_analysis = analysis.get('storage_analysis', {})
                        st.write(f"Max IOPS: {storage_analysis.get('max_iops', 0):,}")
                        st.write(f"Avg Latency: {storage_analysis.get('avg_latency_ms', 0):.1f}ms")
    
    else:
        st.info("📊 vROps analysis not available. Use the enhanced environment setup with vROps metrics import to access detailed performance analysis.")

def show_risk_assessment_tab():
    """Show risk assessment results"""
    
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        show_risk_assessment_display()
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")

def show_risk_assessment_display():
    """Display risk assessment results"""
    
    st.markdown("### ⚠️ Migration Risk Assessment")
    
    risk_assessment = st.session_state.risk_assessment
    
    # Overall risk level display
    risk_level = risk_assessment.get('risk_level', {'level': 'Unknown', 'color': '#666666'})
    overall_score = risk_assessment.get('overall_score', 50)
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {risk_level['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 2rem; font-weight: bold; color: {risk_level['color']};">
                    {risk_level['level']} Risk
                </div>
                <div style="font-size: 1.1rem; color: #666; margin: 5px 0;">
                    Overall Score: {overall_score:.1f}/100
                </div>
                <div style="font-weight: 500; color: #333;">
                    📋 {risk_level['action']}
                </div>
            </div>
            <div style="font-size: 3rem;">
                {'🟢' if overall_score < 30 else '🟡' if overall_score < 50 else '🟠' if overall_score < 70 else '🔴'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Technical Risks")
        tech_risks = risk_assessment.get('technical_risks', {})
        
        for risk_name, score in tech_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#38a169'
            
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span><strong>{risk_name.replace('_', ' ').title()}</strong></span>
                    <span style="color: {color}; font-weight: bold;">{score:.0f}/100</span>
                </div>
                <div style="background: #f0f0f0; border-radius: 10px; height: 8px; margin: 5px 0;">
                    <div style="background: {color}; width: {score}%; height: 8px; border-radius: 10px;"></div>
                </div>
                <div style="color: {color}; font-size: 0.9rem; font-weight: 500;">{risk_level_detail} Risk</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💼 Business Risks")
        business_risks = risk_assessment.get('business_risks', {})
        
        for risk_name, score in business_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#38a169'
            
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span><strong>{risk_name.replace('_', ' ').title()}</strong></span>
                    <span style="color: {color}; font-weight: bold;">{score:.0f}/100</span>
                </div>
                <div style="background: #f0f0f0; border-radius: 10px; height: 8px; margin: 5px 0;">
                    <div style="background: {color}; width: {score}%; height: 8px; border-radius: 10px;"></div>
                </div>
                <div style="color: {color}; font-size: 0.9rem; font-weight: 500;">{risk_level_detail} Risk</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk mitigation strategies
    st.markdown("#### 🛡️ Risk Mitigation Strategies")
    
    mitigation_strategies = risk_assessment.get('mitigation_strategies', [])
    
    if mitigation_strategies:
        for i, strategy in enumerate(mitigation_strategies, 1):
            with st.expander(f"🎯 Strategy {i}: {strategy.get('risk', 'Migration Risk')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Strategy:** {strategy.get('strategy', 'Not specified')}")
                    
                with col2:
                    st.markdown(f"**Timeline:** {strategy.get('timeline', 'TBD')}")
                    cost_impact = strategy.get('cost_impact', 'Medium')
                    st.markdown(f"**Cost Impact:** {cost_impact}")

def show_environment_analysis_tab():
    """Show environment analysis"""
    
    st.markdown("### 🏢 Environment Analysis")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Environment analysis not available. Please run the migration analysis first.")
        return
    
    # Show environment specifications
    if hasattr(st.session_state, 'environment_specs') and st.session_state.environment_specs:
        st.markdown("#### 📋 Environment Specifications")
        
        for env_name, specs in st.session_state.environment_specs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Specs:**")
                    st.write(f"CPU Cores: {specs.get('cpu_cores', specs.get('cpu_cores_allocated', 'N/A'))}")
                    st.write(f"RAM: {specs.get('ram_gb', specs.get('memory_allocated_gb', 'N/A'))} GB")
                    st.write(f"Storage: {specs.get('storage_gb', specs.get('storage_allocated_gb', 'N/A'))} GB")
                
                with col2:
                    st.markdown("**Workload:**")
                    st.write(f"Peak Connections: {specs.get('peak_connections', specs.get('max_database_connections', 'N/A'))}")
                    st.write(f"Application Type: {specs.get('application_type', 'N/A')}")

def show_ai_insights_tab():
    """Show AI insights if available"""
    
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        st.markdown("### 🤖 AI-Powered Insights")
        
        insights = st.session_state.ai_insights
        
        if 'error' in insights:
            st.warning(f"AI insights partially available")
            if 'fallback_insights' in insights:
                fallback = insights['fallback_insights']
                
                st.markdown("#### 💡 Migration Insights")
                st.write(fallback.get('cost_efficiency', ''))
                st.write(fallback.get('migration_strategy', ''))
                st.write(fallback.get('risk_assessment', ''))
                
                if 'recommendations' in fallback:
                    st.markdown("#### 📋 Recommendations")
                    for rec in fallback['recommendations']:
                        st.markdown(f"• {rec}")
        else:
            # Show full AI insights
            if 'ai_analysis' in insights:
                st.markdown("#### 🤖 Claude AI Analysis")
                st.write(insights['ai_analysis'])
    else:
        st.info("🤖 AI insights not available. Provide an Anthropic API key in the configuration to enable AI-powered analysis.")

def show_timeline_analysis_tab():
    """Show migration timeline analysis"""
    
    st.markdown("### 📅 Migration Timeline & Milestones")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Timeline analysis not available. Please configure migration parameters first.")
        return
    
    params = st.session_state.migration_params
    timeline_weeks = params.get('migration_timeline_weeks', 12)
    
    # Create timeline phases
    phases = [
        {"phase": "Planning & Assessment", "weeks": max(2, timeline_weeks * 0.15), "description": "Requirements gathering, current state analysis"},
        {"phase": "Schema Migration", "weeks": max(2, timeline_weeks * 0.25), "description": "Convert schema, stored procedures, functions"},
        {"phase": "Data Migration", "weeks": max(3, timeline_weeks * 0.35), "description": "Initial load, incremental sync, validation"},
        {"phase": "Testing & Validation", "weeks": max(2, timeline_weeks * 0.15), "description": "Performance testing, data validation, UAT"},
        {"phase": "Go-Live & Support", "weeks": max(1, timeline_weeks * 0.10), "description": "Cutover, monitoring, post-migration support"}
    ]
    
    # Timeline visualization
    current_week = 0
    for i, phase in enumerate(phases):
        start_week = current_week
        end_week = current_week + phase["weeks"]
        
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.markdown(f"**{phase['phase']}**")
        
        with col2:
            st.markdown(f"Weeks {int(start_week)+1}-{int(end_week)}")
        
        with col3:
            st.markdown(phase['description'])
        
        current_week = end_week

def show_reports_section():
    """Show reports and export section"""
    
    st.markdown("## 📄 Reports & Export")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        return
    
    # Report generation options
    st.markdown("### 📊 Available Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👔 Executive Summary")
        st.markdown("Perfect for stakeholders and decision makers")
        st.markdown("**Includes:**")
        st.markdown("• High-level cost analysis")
        st.markdown("• ROI and timeline overview")
        st.markdown("• Risk summary")
        st.markdown("• Key recommendations")
        
        if st.button("📄 Generate Executive PDF", key="exec_pdf", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                try:
                    pdf_buffer = generate_executive_summary_pdf()
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Executive Summary",
                            data=pdf_buffer.getvalue(),
                            file_name=f"AWS_Migration_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col2:
        st.markdown("#### 🔧 Technical Report")
        st.markdown("Detailed technical analysis")
        st.markdown("**Includes:**")
        st.markdown("• Environment specifications")
        st.markdown("• Instance recommendations")
        st.markdown("• Detailed cost breakdown")
        st.markdown("• Technical considerations")
        
        if st.button("📄 Generate Technical PDF", key="tech_pdf", use_container_width=True):
            st.info("Technical report generation coming soon!")
    
    with col3:
        st.markdown("#### 📊 Data Export")
        st.markdown("Raw data for further analysis")
        st.markdown("**Includes:**")
        st.markdown("• Cost analysis data")
        st.markdown("• Environment specifications")
        st.markdown("• Risk assessment data")
        
        if st.button("📊 Export Data (CSV)", key="csv_export", use_container_width=True):
            try:
                csv_data = prepare_csv_export_data()
                
                if csv_data:
                    st.download_button(
                        label="📥 Download CSV Data",
                        data=csv_data,
                        file_name=f"AWS_Migration_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("No data available for export")
            except Exception as e:
                st.error(f"Error preparing CSV: {str(e)}")

def generate_executive_summary_pdf():
    """Generate executive summary PDF report"""
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2d3748'),
            alignment=1
        )
        
        story = []
        
        # Title and header
        story.append(Paragraph("AWS Database Migration Analysis", title_style))
        story.append(Paragraph("Executive Summary Report", styles['Heading2']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key metrics
        results = st.session_state.analysis_results
        migration_costs = results.get('migration_costs', {})
        
        metrics_data = [
            ['Metric', 'Value', 'Impact'],
            ['Monthly AWS Cost', f"${results.get('monthly_aws_cost', 0):,.0f}", 'Operational'],
            ['Annual AWS Cost', f"${results.get('monthly_aws_cost', 0) * 12:,.0f}", 'Budget Planning'],
            ['Migration Investment', f"${migration_costs.get('total', 0):,.0f}", 'One-time'],
            ['Risk Level', st.session_state.risk_assessment.get('risk_level', {}).get('level', 'Medium') if hasattr(st.session_state, 'risk_assessment') else 'Medium', 'Manageable']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Key Recommendations", styles['Heading2']))
        
        recommendations = [
            "Proceed with phased migration approach starting with non-production environments",
            "Implement AWS DMS for initial data synchronization",
            "Plan for comprehensive testing protocols for each environment",
            "Consider Aurora for production workloads to optimize performance and cost"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating executive PDF: {e}")
        return None

def generate_executive_summary_pdf_robust(results, migration_params):
    """Generate executive summary PDF - ROBUST VERSION"""
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("AWS Database Migration - Executive Summary", styles['Title']))
        story.append(Spacer(1, 20))
        
        # Key metrics
        story.append(Paragraph("Executive Overview", styles['Heading2']))
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Migration Type', f"{migration_params.get('source_engine', 'N/A')} → {migration_params.get('target_engine', 'N/A')}"],
            ['Data Volume', f"{migration_params.get('data_size_gb', 0):,} GB"],
            ['Timeline', f"{migration_params.get('migration_timeline_weeks', 0)} weeks"],
            ['Monthly AWS Cost', f"${results.get('monthly_aws_cost', 0):,.0f}"],
            ['Annual AWS Cost', f"${results.get('annual_aws_cost', 0):,.0f}"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Recommendations summary
        story.append(Paragraph("Key Recommendations", styles['Heading2']))
        story.append(Paragraph("• Implement phased migration approach", styles['Normal']))
        story.append(Paragraph("• Start with development environments", styles['Normal']))
        story.append(Paragraph("• Use AWS DMS for data synchronization", styles['Normal']))
        story.append(Paragraph("• Plan comprehensive testing protocols", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating executive PDF: {e}")
        return None

def prepare_csv_export_data():
    """Prepare data for CSV export"""
    
    try:
        if not st.session_state.analysis_results:
            return None
        
        # Combine all analysis data
        export_data = []
        
        # Environment cost data
        env_costs = st.session_state.analysis_results.get('environment_costs', {})
        
        for env_name, costs in env_costs.items():
            export_data.append({
                'Environment': env_name,
                'Instance_Cost_Monthly': costs.get('instance_cost', 0),
                'Storage_Cost_Monthly': costs.get('storage_cost', 0),
                'Backup_Cost_Monthly': costs.get('backup_cost', 0),
                'Total_Monthly_Cost': costs.get('total_monthly', 0),
                'Annual_Cost': costs.get('total_monthly', 0) * 12
            })
        
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)
        
    except Exception as e:
        print(f"Error preparing CSV export: {e}")
        return None

def prepare_csv_export_data_enhanced(results, recommendations):
    """Enhanced CSV export with more detailed data"""
    
    try:
        export_data = []
        
        # Environment specifications
        if hasattr(st.session_state, 'environment_specs'):
            env_specs = st.session_state.environment_specs
            
            for env_name in env_specs.keys():
                # Combine specs with recommendations and costs
                specs = env_specs.get(env_name, {})
                rec = recommendations.get(env_name, {}) if recommendations else {}
                costs = results.get('environment_costs', {}).get(env_name, {}) if results else {}
                
                export_data.append({
                    'Environment': env_name,
                    'Current_CPU_Cores': specs.get('cpu_cores', specs.get('cpu_cores_allocated', 'N/A')),
                    'Current_RAM_GB': specs.get('ram_gb', specs.get('memory_allocated_gb', 'N/A')),
                    'Current_Storage_GB': specs.get('storage_gb', specs.get('storage_allocated_gb', 'N/A')),
                    'Recommended_Instance': rec.get('instance_class', 'N/A'),
                    'Multi_AZ': rec.get('multi_az', 'N/A'),
                    'Monthly_Instance_Cost': costs.get('instance_cost', 0),
                    'Monthly_Storage_Cost': costs.get('storage_cost', 0),
                    'Total_Monthly_Cost': costs.get('total_monthly', 0),
                    'Annual_Cost': costs.get('total_monthly', 0) * 12,
                    'Environment_Type': rec.get('environment_type', 'N/A')
                })
        
        df = pd.DataFrame(export_data)
        return df
        
    except Exception as e:
        print(f"Error preparing enhanced CSV export: {e}")
        return None

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="enterprise-header">
        <h1>🚀 Enterprise AWS Database Migration Tool</h1>
        <p style="font-size: 1.2rem; margin: 10px 0;">Comprehensive analysis and planning for AWS database migrations with vROps integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Section",
        [
            "🔧 Migration Configuration",
            "📊 Environment Setup", 
            "🌐 Network Transfer Analysis",
            "🚀 Analysis & Recommendations",
            "📊 Results Dashboard",
            "📄 Reports & Export"
        ]
    )
    
    # Progress indicator
    show_progress_indicator()
    
    # Main content based on selected page
    if page == "🔧 Migration Configuration":
        show_migration_configuration()
    elif page == "📊 Environment Setup":
        show_environment_setup()
    elif page == "🌐 Network Transfer Analysis":
        show_network_transfer_analysis()
    elif page == "🚀 Analysis & Recommendations":
        show_analysis_section()
    elif page == "📊 Results Dashboard":
        show_results_dashboard()
    elif page == "📄 Reports & Export":
        show_reports_section()
    
    # Footer
    show_footer()

def show_progress_indicator():
    """Show progress indicator in sidebar"""
    
    st.sidebar.markdown("### 📈 Progress")
    
    # Calculate progress
    progress_items = [
        ("Migration Config", bool(st.session_state.migration_params)),
        ("Environment Setup", bool(st.session_state.environment_specs)),
        ("Analysis Complete", bool(st.session_state.analysis_results)),
    ]
    
    completed = sum(1 for _, status in progress_items if status)
    progress_percent = int((completed / len(progress_items)) * 100)
    
    st.sidebar.progress(progress_percent)
    st.sidebar.write(f"**{completed}/{len(progress_items)} steps completed**")
    
    # Progress details
    for item, status in progress_items:
        emoji = "✅" if status else "⏳"
        st.sidebar.write(f"{emoji} {item}")

def show_footer():
    """Show application footer"""
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🚀 Enterprise AWS Database Migration Tool v2.0</p>
        <p>Built with Streamlit • Enhanced with vROps Analytics • Powered by AI</p>
        <p>For enterprise database migration planning and analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Additional helper functions for compatibility

def create_bulk_reports_zip_enhanced(results, recommendations, migration_params):
    """Create enhanced ZIP file with all reports"""
    
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Executive summary
            exec_pdf = generate_executive_summary_pdf_robust(results, migration_params)
            if exec_pdf:
                zip_file.writestr(
                    f"Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                    exec_pdf.getvalue()
                )
            
            # CSV data
            csv_data = prepare_csv_export_data_enhanced(results, recommendations)
            if csv_data is not None:
                zip_file.writestr(
                    f"Migration_Analysis_Data_{datetime.now().strftime('%Y%m%d')}.csv",
                    csv_data.to_csv(index=False)
                )
        
        zip_buffer.seek(0)
        return zip_buffer
        
    except Exception as e:
        print(f"Error creating enhanced ZIP file: {e}")
        return None

def validate_session_state():
    """Validate session state for proper application flow"""
    
    required_keys = [
        'environment_specs',
        'migration_params', 
        'analysis_results',
        'recommendations'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in st.session_state:
            missing_keys.append(key)
    
    if missing_keys:
        st.warning(f"⚠️ Missing session state keys: {', '.join(missing_keys)}")
        return False
    
    return True

def reset_analysis():
    """Reset analysis results"""
    
    if st.sidebar.button("🔄 Reset Analysis"):
        # Clear analysis results
        keys_to_clear = [
            'analysis_results',
            'recommendations', 
            'vrops_analysis',
            'network_analysis',
            'transfer_analysis',
            'risk_assessment',
            'ai_insights'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Analysis reset successfully!")
        st.experimental_rerun()

def show_debug_info():
    """Show debug information (for development)"""
    
    if st.sidebar.checkbox("🐛 Debug Mode"):
        st.sidebar.markdown("### Debug Info")
        st.sidebar.write("**Session State Keys:**")
        for key in st.session_state.keys():
            value_preview = str(st.session_state[key])[:50] + "..." if len(str(st.session_state[key])) > 50 else str(st.session_state[key])
            st.sidebar.write(f"• {key}: {value_preview}")

# Performance optimization functions

@st.cache_data
def load_static_data():
    """Load static configuration data"""
    
    return {
        'aws_regions': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
        'instance_families': ['t3', 'r5', 'm5', 'c5'],
        'storage_types': ['gp3', 'io2', 'gp2']
    }

@st.cache_data
def calculate_cost_projections(monthly_cost: float, years: int = 3):
    """Calculate cost projections with caching"""
    
    projections = []
    for year in range(1, years + 1):
        # Assume 5% annual increase
        annual_cost = monthly_cost * 12 * (1.05 ** (year - 1))
        projections.append({
            'year': year,
            'annual_cost': annual_cost,
            'monthly_cost': annual_cost / 12
        })
    
    return projections

# Error handling and logging

def log_error(error_message: str, context: str = ""):
    """Log errors for debugging"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ERROR in {context}: {error_message}"
    
    # In a production environment, you might want to write to a file or external logging service
    print(log_entry)

def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(str(e), func.__name__)
        st.error(f"An error occurred in {func.__name__}: {str(e)}")
        return None

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 