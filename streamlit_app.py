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

# Page Configuration
st.set_page_config(
    page_title="Enterprise AWS Database Migration Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Enhanced Enterprise CSS
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
            
            # M5 instances (General Purpose)
            'db.m5.large': {'vcpu': 2, 'memory_gb': 8, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.xlarge': {'vcpu': 4, 'memory_gb': 16, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.2xlarge': {'vcpu': 8, 'memory_gb': 32, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.4xlarge': {'vcpu': 16, 'memory_gb': 64, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.8xlarge': {'vcpu': 32, 'memory_gb': 128, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.12xlarge': {'vcpu': 48, 'memory_gb': 192, 'network_gbps': 12, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.16xlarge': {'vcpu': 64, 'memory_gb': 256, 'network_gbps': 20, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.m5.24xlarge': {'vcpu': 96, 'memory_gb': 384, 'network_gbps': 25, 'ebs_optimized': True, 'baseline_cpu': 100},
            
            # R5 instances (Memory Optimized)
            'db.r5.large': {'vcpu': 2, 'memory_gb': 16, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.xlarge': {'vcpu': 4, 'memory_gb': 32, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.2xlarge': {'vcpu': 8, 'memory_gb': 64, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.4xlarge': {'vcpu': 16, 'memory_gb': 128, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.8xlarge': {'vcpu': 32, 'memory_gb': 256, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.12xlarge': {'vcpu': 48, 'memory_gb': 384, 'network_gbps': 12, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.16xlarge': {'vcpu': 64, 'memory_gb': 512, 'network_gbps': 20, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.r5.24xlarge': {'vcpu': 96, 'memory_gb': 768, 'network_gbps': 25, 'ebs_optimized': True, 'baseline_cpu': 100},
            
            # C5 instances (Compute Optimized)
            'db.c5.large': {'vcpu': 2, 'memory_gb': 4, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.xlarge': {'vcpu': 4, 'memory_gb': 8, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.2xlarge': {'vcpu': 8, 'memory_gb': 16, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.4xlarge': {'vcpu': 16, 'memory_gb': 32, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.9xlarge': {'vcpu': 36, 'memory_gb': 72, 'network_gbps': 10, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.12xlarge': {'vcpu': 48, 'memory_gb': 96, 'network_gbps': 12, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.18xlarge': {'vcpu': 72, 'memory_gb': 144, 'network_gbps': 25, 'ebs_optimized': True, 'baseline_cpu': 100},
            'db.c5.24xlarge': {'vcpu': 96, 'memory_gb': 192, 'network_gbps': 25, 'ebs_optimized': True, 'baseline_cpu': 100}
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
        elif instance_family == 'm5':
            reasons.append("Balanced compute and memory")
        elif instance_family == 'r5':
            reasons.append("Memory-optimized for database workloads")
        elif instance_family == 'c5':
            reasons.append("Compute-optimized for CPU-intensive workloads")
        
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
# ENHANCED STREAMLIT INTERFACE
# ===========================

def show_enhanced_environment_setup():
    """Enhanced environment setup with comprehensive vROps metrics"""
    
    st.markdown("## 📊 Enhanced Environment Configuration with vROps Metrics")
    
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
            "🔄 Simple Configuration (Legacy)"
        ],
        horizontal=True
    )
    
    if config_method == "📊 vROps Metrics Import":
        show_vrops_import_interface(analyzer)
    elif config_method == "📝 Manual Detailed Entry":
        show_manual_detailed_entry(analyzer)
    elif config_method == "📁 Bulk CSV Upload":
        show_enhanced_bulk_upload(analyzer)
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
            st.code(str(e))

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

def process_vrops_data(df: pd.DataFrame, analyzer: VRopsMetricsAnalyzer) -> Dict:
    """Process uploaded vROps data into environment specifications"""
    
    st.markdown("##### 🔗 Column Mapping")
    
    # Get required metrics
    required_metrics = analyzer.required_metrics
    
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
        
        st.markdown("**Memory Metrics**")
        mappings['max_memory_usage_percent'] = st.selectbox("Max Memory Usage %", available_columns, key="max_mem_col")
        mappings['avg_memory_usage_percent'] = st.selectbox("Avg Memory Usage %", available_columns, key="avg_mem_col")
        mappings['memory_allocated_gb'] = st.selectbox("Memory Allocated GB", available_columns, key="mem_alloc_col")
    
    with col2:
        st.markdown("**Storage Metrics**")
        mappings['max_iops_total'] = st.selectbox("Max IOPS", available_columns, key="max_iops_col")
        mappings['avg_iops_total'] = st.selectbox("Avg IOPS", available_columns, key="avg_iops_col")
        mappings['max_disk_latency_ms'] = st.selectbox("Max Disk Latency ms", available_columns, key="max_lat_col")
        mappings['avg_disk_latency_ms'] = st.selectbox("Avg Disk Latency ms", available_columns, key="avg_lat_col")
        mappings['storage_allocated_gb'] = st.selectbox("Storage Allocated GB", available_columns, key="storage_col")
        
        st.markdown("**Database Metrics**")
        mappings['database_size_gb'] = st.selectbox("Database Size GB", available_columns, key="db_size_col")
        mappings['max_database_connections'] = st.selectbox("Max DB Connections", available_columns, key="max_conn_col")
        
        st.markdown("**Optional Metrics**")
        mappings['observation_period_days'] = st.selectbox("Observation Period Days", available_columns, key="obs_period_col")
    
    if st.button("🔄 Process vROps Data", type="primary"):
        
        # Validate required mappings
        required_fields = ['vm_name', 'max_cpu_usage_percent', 'avg_cpu_usage_percent', 
                          'cpu_cores_allocated', 'max_memory_usage_percent', 'memory_allocated_gb']
        
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
                
                # Add default values for missing metrics
                env_metrics.setdefault('observation_period_days', 30)
                env_metrics.setdefault('application_type', 'Mixed')
                env_metrics.setdefault('peak_hours_start', 9)
                env_metrics.setdefault('peak_hours_end', 17)
                
                # Fix: Ensure the required keys exist for compatibility with other functions
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
            'Max Memory %': f"{metrics.get('max_memory_usage_percent', 0):.1f}%",
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
            
            # Show quick insights
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
                st.markdown(f"**{env_name}:** {top_rec['instance_type']} - {top_rec['recommendation_reason']}")

def show_manual_detailed_entry(analyzer: VRopsMetricsAnalyzer):
    """Show manual detailed entry interface"""
    
    st.markdown("### 📝 Manual Detailed Entry")
    
    # Number of environments
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
        env_metrics['memory_compressed_gb'] = st.number_input(
            "Memory Compressed (GB)", min_value=0.0, max_value=100.0, value=0.0, key=f"mem_compressed_{env_index}"
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
        env_metrics['disk_queue_depth'] = st.number_input(
            "Disk Queue Depth", min_value=1, max_value=100, value=5, key=f"queue_depth_{env_index}"
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
        env_metrics['observation_period_days'] = st.number_input(
            "Observation Period (days)", min_value=7, max_value=365, value=30, key=f"obs_period_{env_index}"
        )

def show_enhanced_bulk_upload(analyzer: VRopsMetricsAnalyzer):
    """Show enhanced bulk upload with comprehensive template"""
    
    st.markdown("### 📁 Enhanced Bulk Upload")
    
    # Comprehensive template
    with st.expander("📋 Download Comprehensive Template", expanded=False):
        
        st.markdown("""
        **Enhanced Template includes:**
        - All vROps performance metrics
        - Database-specific metrics
        - Workload pattern information
        - Application characteristics
        """)
        
        template_data = create_comprehensive_template()
        csv_data = template_data.to_csv(index=False)
        
        st.dataframe(template_data, use_container_width=True)
        
        st.download_button(
            label="📥 Download Comprehensive Template",
            data=csv_data,
            file_name="comprehensive_environment_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload with enhanced processing
    uploaded_file = st.file_uploader(
        "Upload Environment Data",
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file with comprehensive environment metrics"
    )
    
    if uploaded_file is not None:
        process_enhanced_bulk_upload(uploaded_file, analyzer)

def create_comprehensive_template() -> pd.DataFrame:
    """Create comprehensive template with all metrics"""
    
    template_data = {
        # Basic Info
        'Environment_Name': ['Production-DB1', 'Staging-DB1', 'QA-DB1', 'Dev-DB1'],
        'VM_Name': ['PROD-SQL-01', 'STAGE-SQL-01', 'QA-SQL-01', 'DEV-SQL-01'],
        'Environment_Type': ['Production', 'Staging', 'QA', 'Development'],
        
        # CPU Metrics
        'CPU_Cores_Allocated': [16, 8, 4, 2],
        'Max_CPU_Usage_Percent': [85.2, 65.4, 45.6, 35.2],
        'Avg_CPU_Usage_Percent': [65.4, 45.7, 28.9, 22.1],
        'CPU_Ready_Time_ms': [2500, 1200, 500, 200],
        'CPU_CoStop_ms': [150, 80, 20, 0],
        
        # Memory Metrics
        'Memory_Allocated_GB': [64, 32, 16, 8],
        'Max_Memory_Usage_Percent': [78.9, 68.5, 52.3, 45.1],
        'Avg_Memory_Usage_Percent': [68.5, 55.2, 38.7, 32.8],
        'Memory_Balloon_GB': [0, 0.5, 0, 0],
        'Memory_Swapped_GB': [0, 0, 0, 0],
        'Memory_Compressed_GB': [0.2, 0.1, 0, 0],
        
        # Storage Metrics
        'Storage_Allocated_GB': [2000, 1000, 500, 200],
        'Storage_Used_GB': [1600, 750, 350, 120],
        'Max_IOPS_Total': [8500, 4200, 2100, 800],
        'Avg_IOPS_Total': [5200, 2800, 1200, 450],
        'Max_IOPS_Read': [6000, 3000, 1500, 600],
        'Max_IOPS_Write': [2500, 1200, 600, 200],
        'Max_Disk_Latency_ms': [12.5, 8.9, 6.2, 4.1],
        'Avg_Disk_Latency_ms': [8.9, 6.1, 4.2, 2.8],
        'Max_Disk_Throughput_MBps': [250, 150, 85, 35],
        'Avg_Disk_Throughput_MBps': [180, 105, 60, 22],
        'Disk_Queue_Depth': [6, 4, 2, 1],
        
        # Network Metrics
        'Max_Network_Throughput_Mbps': [500, 200, 100, 50],
        'Avg_Network_Throughput_Mbps': [300, 120, 60, 25],
        'Max_Network_Packets_Per_Sec': [50000, 20000, 10000, 5000],
        'Network_Latency_ms': [2.5, 3.1, 4.2, 5.1],
        'Network_Packet_Loss_Percent': [0, 0, 0, 0],
        
        # Database Metrics
        'Database_Size_GB': [1200, 600, 250, 80],
        'Max_Database_Connections': [450, 200, 100, 50],
        'Avg_Database_Connections': [320, 140, 70, 25],
        'Max_Transaction_Rate_Per_Sec': [2500, 1200, 500, 200],
        'Avg_Query_Response_Time_ms': [50, 75, 100, 150],
        'Buffer_Cache_Hit_Ratio_Percent': [98, 96, 94, 92],
        'Lock_Wait_Time_ms': [5, 8, 12, 20],
        'Log_File_Size_GB': [50, 25, 10, 5],
        'TempDB_Usage_GB': [20, 10, 5, 2],
        
        # Workload Patterns
        'Application_Type': ['OLTP', 'OLTP', 'Mixed', 'OLTP'],
        'Peak_Hours_Start': [8, 9, 9, 9],
        'Peak_Hours_End': [18, 17, 17, 17],
        'Weekend_Usage_Factor': [0.3, 0.1, 0.2, 0.1],
        'Seasonal_Peak_Factor': [1.5, 1.2, 1.1, 1.0],
        'Growth_Rate_Percent_Annual': [15, 10, 5, 5],
        'Observation_Period_Days': [90, 60, 45, 30],
        'Availability_Percent': [99.9, 99.5, 99.0, 98.0],
        
        # Application Metrics
        'Concurrent_Users_Max': [500, 200, 100, 25],
        'Concurrent_Users_Avg': [350, 120, 60, 15],
        'Response_Time_SLA_ms': [200, 500, 1000, 2000],
        'Downtime_Tolerance_Minutes': [5, 30, 60, 240],
        'Backup_Window_Hours': [2, 3, 4, 6],
        'Maintenance_Window_Hours': [4, 6, 8, 12]
    }
    
    return pd.DataFrame(template_data)

def process_enhanced_bulk_upload(uploaded_file, analyzer: VRopsMetricsAnalyzer):
    """Process enhanced bulk upload file"""
    
    try:
        # Load file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Auto-detect column mappings
        auto_mappings = auto_detect_column_mappings(df.columns.tolist())
        
        st.markdown("#### 🔍 Auto-detected Column Mappings")
        
        # Show detected mappings
        mapping_df = pd.DataFrame([
            {'Required Field': k, 'Detected Column': v, 'Confidence': 'High' if v else 'Not Found'}
            for k, v in auto_mappings.items()
        ])
        
        st.dataframe(mapping_df, use_container_width=True)
        
        # Allow manual override
        with st.expander("🔧 Override Mappings (Optional)"):
            st.info("Only modify if auto-detection is incorrect")
            # Add manual mapping interface here if needed
        
        if st.button("🚀 Process Enhanced Data", type="primary"):
            
            environments = process_enhanced_data(df, auto_mappings)
            
            if environments:
                st.session_state.environment_specs = environments
                st.success(f"✅ Processed {len(environments)} environments!")
                
                # Run comprehensive analysis
                with st.spinner("🔄 Running comprehensive analysis..."):
                    analysis_results = analyzer.analyze_vrops_metrics(environments)
                    st.session_state.vrops_analysis = analysis_results
                
                st.success("✅ Analysis complete!")
                show_vrops_analysis_summary(analysis_results)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

def auto_detect_column_mappings(columns: List[str]) -> Dict[str, str]:
    """Auto-detect column mappings based on common naming patterns"""
    
    mappings = {}
    columns_lower = [col.lower() for col in columns]
    
    # Mapping patterns
    patterns = {
        'environment_name': ['environment', 'env', 'environment_name'],
        'vm_name': ['vm', 'server', 'host', 'vm_name', 'server_name'],
        'cpu_cores_allocated': ['cpu_cores', 'cores', 'vcpu', 'cpu_count'],
        'max_cpu_usage_percent': ['max_cpu', 'cpu_max', 'peak_cpu'],
        'avg_cpu_usage_percent': ['avg_cpu', 'cpu_avg', 'cpu_average'],
        'memory_allocated_gb': ['memory_gb', 'mem_gb', 'memory_allocated'],
        'max_memory_usage_percent': ['max_memory', 'mem_max', 'peak_memory'],
        'avg_memory_usage_percent': ['avg_memory', 'mem_avg', 'memory_average'],
        'max_iops_total': ['max_iops', 'iops_max', 'peak_iops'],
        'avg_iops_total': ['avg_iops', 'iops_avg', 'iops_average'],
        'max_disk_latency_ms': ['max_latency', 'latency_max', 'peak_latency'],
        'avg_disk_latency_ms': ['avg_latency', 'latency_avg', 'latency_average'],
        'storage_allocated_gb': ['storage_gb', 'disk_gb', 'storage_allocated'],
        'database_size_gb': ['db_size', 'database_size', 'db_gb']
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
            'growth_rate_percent_annual': 10
            # Add these missing keys that are expected by other parts of the code
            'cpu_cores': env_metrics.get('cpu_cores_allocated', 4),  # Map from allocated if available
            'ram_gb': env_metrics.get('memory_allocated_gb', 16),    # Map from allocated if available
            'storage_gb': env_metrics.get('storage_allocated_gb', 500)  # Map from allocated if available
        
        }
        
        for key, default_value in defaults.items():
            env_metrics.setdefault(key, default_value)
        
        environments[env_name] = env_metrics
    
    return environments

def show_simple_configuration():
    """Show simple configuration for backward compatibility"""
    
    st.markdown("### 🔄 Simple Configuration (Legacy)")
    st.info("This is the simplified configuration mode. For better AWS sizing accuracy, consider using the vROps metrics import.")
    
    # Use the original simple interface
    show_manual_environment_setup()

# Add this to your main navigation
def integrate_enhanced_environment_module():
    """Integration instructions for the enhanced environment module"""
    
    # Replace the existing environment setup function with:
    # show_enhanced_environment_setup()
    
    # Add to session state initialization:
    # 'vrops_analysis': None,
    # 'vrops_analyzer': None
    
    pass

# ===========================
# CORE CLASSES AND FUNCTIONS
# ===========================

class AWSPricingAPI:
    """AWS Pricing API integration for real-time database pricing"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        
    def get_rds_pricing(self, region: str, engine: str, instance_class: str) -> Dict:
        """Get RDS pricing for specific instance"""
        cache_key = f"{region}_{engine}_{instance_class}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simplified pricing data (in production, use actual AWS Pricing API)
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'storage_gb': 0.115},
                    'db.t3.small': {'hourly': 0.051, 'storage_gb': 0.115},
                    'db.t3.medium': {'hourly': 0.102, 'storage_gb': 0.115},
                    'db.t3.large': {'hourly': 0.204, 'storage_gb': 0.115},
                    'db.r5.large': {'hourly': 0.24, 'storage_gb': 0.115},
                    'db.r5.xlarge': {'hourly': 0.48, 'storage_gb': 0.115},
                    'db.r5.2xlarge': {'hourly': 0.96, 'storage_gb': 0.115},
                    'db.r5.4xlarge': {'hourly': 1.92, 'storage_gb': 0.115},
                },
                'oracle-ee': {
                    'db.t3.medium': {'hourly': 0.408, 'storage_gb': 0.115},
                    'db.r5.large': {'hourly': 0.96, 'storage_gb': 0.115},
                    'db.r5.xlarge': {'hourly': 1.92, 'storage_gb': 0.115},
                    'db.r5.2xlarge': {'hourly': 3.84, 'storage_gb': 0.115},
                    'db.r5.4xlarge': {'hourly': 7.68, 'storage_gb': 0.115},
                },
                'aurora-postgresql': {
                    'db.r5.large': {'hourly': 0.29, 'storage_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.58, 'storage_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 1.16, 'storage_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 2.32, 'storage_gb': 0.10},
                    'db.r5.8xlarge': {'hourly': 4.64, 'storage_gb': 0.10},
                }
            }
        }
        
        result = pricing_data.get(region, {}).get(engine, {}).get(instance_class, 
                                                                 {'hourly': 0.5, 'storage_gb': 0.115})
        self.cache[cache_key] = result
        return result

class DatabaseEngine:
    """Database engine configuration and compatibility"""
    
    ENGINES = {
        'oracle-ee': {
            'name': 'Oracle Enterprise Edition',
            'aws_targets': ['oracle-ee', 'postgres', 'aurora-postgresql'],
            'migration_complexity': {'oracle-ee': 1.0, 'postgres': 2.5, 'aurora-postgresql': 2.0},
            'features': ['ACID', 'Stored Procedures', 'Triggers', 'Views', 'Indexes']
        },
        'oracle-se': {
            'name': 'Oracle Standard Edition',
            'aws_targets': ['oracle-se2', 'postgres', 'aurora-postgresql'],
            'migration_complexity': {'oracle-se2': 1.2, 'postgres': 2.3, 'aurora-postgresql': 1.8},
            'features': ['ACID', 'Stored Procedures', 'Triggers', 'Views']
        },
        'postgres': {
            'name': 'PostgreSQL',
            'aws_targets': ['postgres', 'aurora-postgresql'],
            'migration_complexity': {'postgres': 1.0, 'aurora-postgresql': 1.2},
            'features': ['ACID', 'Extensions', 'JSON', 'Full-text Search']
        },
        'mysql': {
            'name': 'MySQL',
            'aws_targets': ['mysql', 'aurora-mysql'],
            'migration_complexity': {'mysql': 1.0, 'aurora-mysql': 1.1},
            'features': ['ACID', 'Replication', 'Partitioning']
        }
    }
    
    @classmethod
    def get_migration_type(cls, source: str, target: str) -> str:
        """Determine if migration is homogeneous or heterogeneous"""
        if source == target:
            return "homogeneous"
        elif source.split('-')[0] == target.split('-')[0]:
            return "homogeneous"
        else:
            return "heterogeneous"
    
    @classmethod
    def get_complexity_multiplier(cls, source: str, target: str) -> float:
        """Get complexity multiplier for migration"""
        if source not in cls.ENGINES:
            return 2.0
        return cls.ENGINES[source]['migration_complexity'].get(target, 2.5)

class MigrationAnalyzer:
    """Core migration analysis engine with AI capabilities"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = AWSPricingAPI()
        self.anthropic_client = None
        
        if anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            except Exception as e:
                st.warning(f"AI client initialization failed: {e}")
    
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations based on current specs"""
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            environment_type = self._categorize_environment(env_name)
            
            # Instance sizing logic
            if cpu_cores <= 2 and ram_gb <= 8:
                instance_class = 'db.t3.medium'
            elif cpu_cores <= 4 and ram_gb <= 16:
                instance_class = 'db.t3.large'
            elif cpu_cores <= 8 and ram_gb <= 32:
                instance_class = 'db.r5.large'
            elif cpu_cores <= 16 and ram_gb <= 64:
                instance_class = 'db.r5.xlarge'
            elif cpu_cores <= 32 and ram_gb <= 128:
                instance_class = 'db.r5.2xlarge'
            else:
                instance_class = 'db.r5.4xlarge'
            
            # Environment-specific adjustments
            if environment_type == 'development':
                instance_class = self._downsize_instance(instance_class)
            elif environment_type == 'production':
                instance_class = self._ensure_production_sizing(instance_class, cpu_cores, ram_gb)
            
            recommendations[env_name] = {
                'instance_class': instance_class,
                'environment_type': environment_type,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'multi_az': environment_type in ['production', 'staging']
            }
        
        return recommendations
    
    def _categorize_environment(self, env_name: str) -> str:
        """Categorize environment type"""
        env_lower = env_name.lower()
        if any(term in env_lower for term in ['prod', 'production', 'prd']):
            return 'production'
        elif any(term in env_lower for term in ['stag', 'staging', 'preprod']):
            return 'staging'
        elif any(term in env_lower for term in ['qa', 'test', 'uat', 'sqa']):
            return 'testing'
        elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
            return 'development'
        return 'other'
    
    def _downsize_instance(self, instance_class: str) -> str:
        """Downsize instance for development environments"""
        size_mapping = {
            'db.r5.4xlarge': 'db.r5.2xlarge',
            'db.r5.2xlarge': 'db.r5.xlarge',
            'db.r5.xlarge': 'db.r5.large',
            'db.r5.large': 'db.t3.large',
            'db.t3.large': 'db.t3.medium'
        }
        return size_mapping.get(instance_class, instance_class)
    
    def _ensure_production_sizing(self, instance_class: str, cpu_cores: int, ram_gb: int) -> str:
        """Ensure adequate sizing for production"""
        if cpu_cores >= 16 or ram_gb >= 64:
            if instance_class in ['db.t3.medium', 'db.t3.large']:
                return 'db.r5.xlarge'
        return instance_class
    
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate comprehensive migration costs"""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        # AWS infrastructure costs
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            pricing = self.pricing_api.get_rds_pricing(region, target_engine, rec['instance_class'])
            
            # Calculate monthly costs
            instance_hours = 24 * 30  # 30 days
            instance_cost = pricing['hourly'] * instance_hours
            
            # Multi-AZ doubles the instance cost
            if rec['multi_az']:
                instance_cost *= 2
            
            # Storage costs
            storage_cost = rec['storage_gb'] * pricing['storage_gb']
            
            # Backup costs (estimated)
            backup_cost = storage_cost * 0.5
            
            # Additional costs
            monitoring_cost = 30 if rec['environment_type'] == 'production' else 10
            
            env_total = instance_cost + storage_cost + backup_cost + monitoring_cost
            environment_costs[env_name] = {
                'instance_cost': instance_cost,
                'storage_cost': storage_cost,
                'backup_cost': backup_cost,
                'monitoring_cost': monitoring_cost,
                'total_monthly': env_total
            }
            
            total_monthly_cost += env_total
        
        # Migration service costs
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks  # t3.medium for migration
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000  # $8k per week for team
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs['total'],
            'professional_services': ps_cost,
            'contingency': (dms_instance_cost + transfer_costs['total'] + ps_cost) * 0.2,
            'total': 0
        }
        migration_costs['total'] = sum(migration_costs.values()) - migration_costs['contingency']
        migration_costs['total'] += migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs for different methods"""
        
        use_direct_connect = migration_params.get('use_direct_connect', False)
        bandwidth_mbps = migration_params.get('bandwidth_mbps', 1000)
        
        # Internet transfer
        internet_cost = data_size_gb * 0.09  # $0.09 per GB
        internet_time_hours = (data_size_gb * 8192) / (bandwidth_mbps * 3600)  # Convert to hours
        
        # Direct Connect transfer
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02  # $0.02 per GB
            dx_time_hours = internet_time_hours * 0.3  # 3x faster
        else:
            dx_cost = internet_cost
            dx_time_hours = internet_time_hours
        
        # DataSync costs
        datasync_cost = data_size_gb * 0.0125  # $0.0125 per GB
        
        return {
            'internet': {'cost': internet_cost, 'time_hours': internet_time_hours},
            'direct_connect': {'cost': dx_cost, 'time_hours': dx_time_hours},
            'datasync': {'cost': datasync_cost, 'time_hours': dx_time_hours},
            'total': min(internet_cost, dx_cost) + datasync_cost
        }
    
    async def generate_ai_insights(self, analysis_results: Dict, migration_params: Dict) -> Dict:
        """Generate AI-powered insights and recommendations"""
        
        if not self.anthropic_client:
            return {'error': 'AI client not available'}
        
        try:
            # Prepare analysis context
            context = {
                'source_engine': migration_params.get('source_engine'),
                'target_engine': migration_params.get('target_engine'),
                'environments': len(analysis_results.get('environment_costs', {})),
                'total_monthly_cost': analysis_results.get('monthly_aws_cost', 0),
                'migration_cost': analysis_results.get('migration_costs', {}).get('total', 0),
                'data_size_gb': migration_params.get('data_size_gb', 0)
            }
            
            prompt = f"""
            As an AWS database migration expert, analyze this migration scenario and provide insights:
            
            Migration Details:
            - Source: {context['source_engine']} → Target: {context['target_engine']}
            - Environments: {context['environments']}
            - Monthly AWS Cost: ${context['total_monthly_cost']:,.2f}
            - Migration Cost: ${context['migration_cost']:,.2f}
            - Data Size: {context['data_size_gb']:,} GB
            
            Provide a comprehensive analysis covering:
            1. Migration complexity assessment
            2. Cost optimization opportunities
            3. Risk factors and mitigation strategies
            4. Timeline recommendations
            5. Best practices for this specific migration path
            
            Format as JSON with keys: complexity, cost_optimization, risks, timeline, best_practices
            """
            
            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_response = message.content[0].text
            
            # Try to parse as JSON, fallback to text
            try:
                insights = json.loads(ai_response)
            except:
                insights = {
                    'complexity': 'Medium',
                    'cost_optimization': 'Review instance sizing',
                    'risks': 'Standard migration risks apply',
                    'timeline': 'Plan for 12-16 weeks',
                    'best_practices': ai_response
                }
            
            return insights
            
        except Exception as e:
            return {'error': f'AI analysis failed: {str(e)}'}

# ===========================
# NETWORK TRANSFER ANALYSIS MODULE
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json

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
        security_req = migration_params.get('security_requirements', 'standard')
        
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
        recommendations = {
            'primary_recommendation': self._get_primary_recommendation(sorted_patterns, migration_params),
            'alternative_options': self._get_alternative_options(sorted_patterns[:3], migration_params),
            'cost_optimization': self._get_cost_optimization_tips(sorted_patterns, migration_params),
            'risk_considerations': self._get_risk_considerations(sorted_patterns, migration_params),
            'timeline_impact': self._get_timeline_impact(sorted_patterns, migration_params)
        }
        
        return recommendations
    
    def _score_cost(self, total_cost: float, budget_level: str) -> float:
        """Score based on cost relative to budget constraints"""
        budget_thresholds = {
            'low': 10000,
            'medium': 50000,
            'high': 200000
        }
        
        threshold = budget_thresholds.get(budget_level, 50000)
        
        if total_cost <= threshold * 0.5:
            return 100
        elif total_cost <= threshold:
            return 80
        elif total_cost <= threshold * 1.5:
            return 60
        elif total_cost <= threshold * 2:
            return 40
        else:
            return 20
    
    def _score_time(self, transfer_days: float, timeline_weeks: int) -> float:
        """Score based on time relative to migration timeline"""
        available_days = timeline_weeks * 7 * 0.3  # 30% of timeline for data transfer
        
        if transfer_days <= available_days * 0.3:
            return 100
        elif transfer_days <= available_days * 0.5:
            return 90
        elif transfer_days <= available_days * 0.7:
            return 75
        elif transfer_days <= available_days:
            return 60
        else:
            return 30
    
    def _get_primary_recommendation(self, sorted_patterns: List, migration_params: Dict) -> Dict:
        """Get primary recommendation with reasoning"""
        
        best_pattern_id, best_scores = sorted_patterns[0]
        pattern_info = self.transfer_patterns[best_pattern_id]
        
        reasoning = self._generate_recommendation_reasoning(
            best_pattern_id, best_scores, migration_params
        )
        
        return {
            'pattern_id': best_pattern_id,
            'pattern_name': pattern_info['name'],
            'description': pattern_info['description'],
            'score': best_scores['composite_score'],
            'reasoning': reasoning,
            'implementation_steps': self._get_implementation_steps(best_pattern_id),
            'estimated_timeline': self._get_implementation_timeline(best_pattern_id),
            'key_considerations': pattern_info['pros'][:3]
        }
    
    def _get_alternative_options(self, top_patterns: List, migration_params: Dict) -> List:
        """Get alternative options with brief explanations"""
        
        alternatives = []
        
        for pattern_id, scores in top_patterns[1:]:  # Skip the primary recommendation
            pattern_info = self.transfer_patterns[pattern_id]
            
            alternatives.append({
                'pattern_name': pattern_info['name'],
                'score': scores['composite_score'],
                'best_for': pattern_info['use_cases'][0],
                'trade_off': self._get_trade_off_analysis(pattern_id, top_patterns[0][0])
            })
        
        return alternatives
    
    def _generate_recommendation_reasoning(self, pattern_id: str, scores: Dict, migration_params: Dict) -> str:
        """Generate human-readable reasoning for recommendation"""
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        reasoning_parts = []
        
        # Data size reasoning
        if data_size_gb < 500:
            reasoning_parts.append("Given the moderate data size")
        elif data_size_gb < 5000:
            reasoning_parts.append("Given the substantial data volume")
        else:
            reasoning_parts.append("Given the large-scale data migration requirements")
        
        # Cost reasoning
        if scores['cost_score'] > 80:
            reasoning_parts.append("this pattern offers excellent cost efficiency")
        elif scores['cost_score'] > 60:
            reasoning_parts.append("this pattern provides balanced cost considerations")
        else:
            reasoning_parts.append("while this pattern has higher costs, it provides superior capabilities")
        
        # Time reasoning
        if scores['time_score'] > 80:
            reasoning_parts.append("and fits well within your migration timeline")
        elif scores['time_score'] > 60:
            reasoning_parts.append("and aligns reasonably with your timeline constraints")
        else:
            reasoning_parts.append("though it may require timeline adjustments")
        
        return ". ".join(reasoning_parts) + "."
    
    def _get_implementation_steps(self, pattern_id: str) -> List[str]:
        """Get implementation steps for a pattern"""
        
        steps_map = {
            'internet_dms': [
                "Set up DMS replication instance in target AWS region",
                "Configure source and target database endpoints",
                "Create and configure replication task",
                "Test connectivity and run initial assessment",
                "Execute full load and ongoing replication"
            ],
            'dx_dms': [
                "Establish AWS Direct Connect connection",
                "Configure virtual interfaces and routing",
                "Set up DMS replication instance",
                "Configure database endpoints over DX",
                "Test connectivity and execute migration"
            ],
            'dx_datasync_vpc': [
                "Establish AWS Direct Connect connection",
                "Configure VPC endpoints for DataSync and S3",
                "Deploy DataSync agent in on-premises environment",
                "Create DataSync tasks for data transfer",
                "Monitor and validate data transfer completion"
            ],
            'vpn_dms': [
                "Set up VPN Gateway and Customer Gateway",
                "Establish site-to-site VPN connection",
                "Configure DMS replication instance",
                "Set up database endpoints over VPN",
                "Execute migration with monitoring"
            ],
            'hybrid_snowball_dms': [
                "Order and configure AWS Snowball devices",
                "Perform initial data extraction to Snowball",
                "Ship devices and import to S3",
                "Set up DMS for ongoing synchronization",
                "Coordinate final cutover timing"
            ],
            'multipath_redundant': [
                "Establish both Direct Connect and VPN connections",
                "Configure BGP routing with path preferences",
                "Set up redundant DMS instances",
                "Test failover scenarios",
                "Execute migration with active monitoring"
            ]
        }
        
        return steps_map.get(pattern_id, ["Pattern-specific steps not defined"])
    
    def _get_implementation_timeline(self, pattern_id: str) -> str:
        """Get estimated implementation timeline"""
        
        timeline_map = {
            'internet_dms': "1-2 weeks",
            'dx_dms': "4-6 weeks (including DX provisioning)",
            'dx_datasync_vpc': "6-8 weeks (complex VPC setup)",
            'vpn_dms': "2-3 weeks",
            'hybrid_snowball_dms': "8-12 weeks (includes shipping)",
            'multipath_redundant': "8-10 weeks (dual-path complexity)"
        }
        
        return timeline_map.get(pattern_id, "Timeline varies")
    
    def _get_trade_off_analysis(self, pattern_id: str, primary_pattern_id: str) -> str:
        """Get trade-off analysis between patterns"""
        
        # This would be more sophisticated in production
        trade_offs = {
            'internet_dms': "Lower setup cost but higher data transfer fees",
            'dx_dms': "Higher setup cost but better reliability and performance",
            'dx_datasync_vpc': "Maximum security but highest complexity",
            'vpn_dms': "Good security balance with moderate setup",
            'hybrid_snowball_dms': "Fastest for large data but complex coordination",
            'multipath_redundant': "Maximum reliability but highest cost"
        }
        
        return trade_offs.get(pattern_id, "Different cost/performance trade-offs")
    
    def _get_cost_optimization_tips(self, sorted_patterns: List, migration_params: Dict) -> List[str]:
        """Generate cost optimization recommendations"""
        
        tips = []
        
        # Analyze if Direct Connect is cost-effective
        dx_patterns = [p for p in sorted_patterns if 'dx' in p[0]]
        if dx_patterns and migration_params.get('data_size_gb', 1000) < 1000:
            tips.append("Consider internet-based transfer for smaller datasets to avoid Direct Connect setup costs")
        
        # Timeline optimization
        if migration_params.get('migration_timeline_weeks', 12) > 16:
            tips.append("Extended timeline allows for more cost-effective transfer methods")
        
        # Hybrid approach suggestion
        data_size = migration_params.get('data_size_gb', 1000)
        if data_size > 10000:
            tips.append("Consider hybrid Snowball approach for very large datasets to minimize bandwidth costs")
        
        tips.append("Implement data compression and deduplication to reduce transfer volumes")
        tips.append("Schedule transfers during off-peak hours to optimize bandwidth utilization")
        
        return tips
    
    def _get_risk_considerations(self, sorted_patterns: List, migration_params: Dict) -> List[str]:
        """Generate risk considerations"""
        
        risks = []
        
        primary_pattern = sorted_patterns[0][0]
        
        if 'internet' in primary_pattern:
            risks.append("Internet dependency may cause variable transfer speeds")
            risks.append("Consider backup connectivity options for critical migrations")
        
        if 'dx' in primary_pattern:
            risks.append("Direct Connect provisioning time may impact project timeline")
            risks.append("Single point of failure without redundant connectivity")
        
        if 'snowball' in primary_pattern:
            risks.append("Physical device logistics and coordination complexity")
            risks.append("Potential delays due to shipping and customs processes")
        
        if migration_params.get('data_size_gb', 1000) > 5000:
            risks.append("Large data volumes require careful bandwidth planning")
            risks.append("Consider phased approach to minimize business impact")
        
        return risks
    
    def _get_timeline_impact(self, sorted_patterns: List, migration_params: Dict) -> Dict:
        """Analyze timeline impact of different patterns"""
        
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        fastest_pattern = min(sorted_patterns, key=lambda x: x[1]['metrics']['transfer_time_days'])
        slowest_pattern = max(sorted_patterns, key=lambda x: x[1]['metrics']['transfer_time_days'])
        
        return {
            'fastest_option': {
                'pattern': self.transfer_patterns[fastest_pattern[0]]['name'],
                'duration_days': fastest_pattern[1]['metrics']['transfer_time_days'],
                'timeline_utilization': f"{(fastest_pattern[1]['metrics']['transfer_time_days'] / (timeline_weeks * 7)) * 100:.1f}%"
            },
            'slowest_option': {
                'pattern': self.transfer_patterns[slowest_pattern[0]]['name'],
                'duration_days': slowest_pattern[1]['metrics']['transfer_time_days'],
                'timeline_utilization': f"{(slowest_pattern[1]['metrics']['transfer_time_days'] / (timeline_weeks * 7)) * 100:.1f}%"
            },
            'recommendation': "Plan for 20-30% buffer time beyond calculated transfer duration for testing and validation"
        }

# ===========================
# NETWORK VISUALIZATION FUNCTIONS
# ===========================

def create_network_comparison_chart(transfer_analysis: Dict) -> go.Figure:
    """Create comprehensive network pattern comparison chart"""
    
    patterns = []
    costs = []
    durations = []
    reliability = []
    security = []
    complexity = []
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = NetworkTransferAnalyzer().transfer_patterns[pattern_id]
        patterns.append(pattern_info['name'])
        costs.append(metrics['total_cost'])
        durations.append(metrics['transfer_time_days'])
        reliability.append(metrics['reliability_score'])
        security.append(metrics['security_score'])
        complexity.append(metrics['complexity_score'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Cost Comparison', 'Transfer Duration', 
                       'Reliability vs Security', 'Complexity Assessment'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cost comparison
    fig.add_trace(
        go.Bar(x=patterns, y=costs, name='Total Cost ($)', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Duration comparison
    fig.add_trace(
        go.Bar(x=patterns, y=durations, name='Duration (days)', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Reliability vs Security scatter
    fig.add_trace(
        go.Scatter(x=reliability, y=security, mode='markers+text',
                  text=patterns, textposition="top center",
                  marker=dict(size=15, color=complexity, colorscale='Viridis',
                            showscale=True, colorbar=dict(title="Complexity Score")),
                  name='Reliability vs Security'),
        row=2, col=1
    )
    
    # Complexity radar-style (simplified as bar)
    fig.add_trace(
        go.Bar(x=patterns, y=complexity, name='Complexity Score', marker_color='coral'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Network Transfer Pattern Analysis",
        showlegend=False
    )
    
    # Update x-axis labels to be rotated
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    return fig

def create_cost_duration_optimization_chart(transfer_analysis: Dict) -> go.Figure:
    """Create cost vs duration optimization chart"""
    
    patterns = []
    costs = []
    durations = []
    scores = []
    
    analyzer = NetworkTransferAnalyzer()
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = analyzer.transfer_patterns[pattern_id]
        patterns.append(pattern_info['name'])
        costs.append(metrics['total_cost'])
        durations.append(metrics['transfer_time_days'])
        
        # Calculate efficiency score (inverse of cost and duration)
        efficiency = 1000000 / (metrics['total_cost'] * metrics['transfer_time_days'])
        scores.append(efficiency)
    
    fig = go.Figure()
    
    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=durations,
        y=costs,
        mode='markers+text',
        text=patterns,
        textposition="top center",
        marker=dict(
            size=[score/max(scores)*50 + 10 for score in scores],  # Size based on efficiency
            color=scores,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Efficiency Score")
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Duration: %{x:.1f} days<br>' +
                      'Cost: $%{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Cost vs Duration Optimization Matrix',
        xaxis_title='Transfer Duration (days)',
        yaxis_title='Total Cost ($)',
        height=600,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text="🎯 Ideal: Lower-left quadrant (Low cost, Low duration)",
                showarrow=False,
                font=dict(size=12, color="green"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
        ]
    )
    
    return fig

def create_network_architecture_diagram(selected_pattern: str) -> go.Figure:
    """Create network architecture diagram for selected pattern"""
    
    # Simplified architecture representation using plotly
    # In production, you might use more sophisticated diagramming tools
    
    analyzer = NetworkTransferAnalyzer()
    pattern_info = analyzer.transfer_patterns.get(selected_pattern, {})
    
    # Create a simple flow diagram
    fig = go.Figure()
    
    # Define positions for different components
    positions = {
        'on_premises': (1, 3),
        'internet': (3, 4),
        'dx': (3, 2),
        'vpn': (3, 3),
        'aws_services': (5, 3),
        'target_db': (7, 3)
    }
    
    # Add nodes
    for component, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            text=[component.replace('_', ' ').title()],
            textposition="middle center",
            marker=dict(size=60, color='lightblue'),
            showlegend=False
        ))
    
    # Add connections based on pattern
    connections = []
    if 'internet' in selected_pattern:
        connections.extend([('on_premises', 'internet'), ('internet', 'aws_services')])
    if 'dx' in selected_pattern:
        connections.extend([('on_premises', 'dx'), ('dx', 'aws_services')])
    if 'vpn' in selected_pattern:
        connections.extend([('on_premises', 'vpn'), ('vpn', 'aws_services')])
    
    connections.append(('aws_services', 'target_db'))
    
    # Draw connections
    for start, end in connections:
        start_pos = positions[start]
        end_pos = positions[end]
        
        fig.add_trace(go.Scatter(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            mode='lines',
            line=dict(width=3, color='gray'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Network Architecture: {pattern_info.get("name", selected_pattern)}',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

# ===========================
# STREAMLIT INTERFACE FUNCTIONS
# ===========================

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
    if hasattr(st.session_state, 'transfer_analysis'):
        show_network_analysis_results()

def show_network_analysis_results():
    """Display network analysis results"""
    
    # Fix: Check if transfer_analysis exists and is not None
    if not hasattr(st.session_state, 'transfer_analysis') or st.session_state.transfer_analysis is None:
        st.error("No network analysis results available. Please run the network analysis first.")
        return
    
    transfer_analysis = st.session_state.transfer_analysis
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Recommendations",
        "📊 Pattern Comparison", 
        "💰 Cost Analysis",
        "⏱️ Timeline Analysis",
        "🏗️ Architecture"
    ])
    
    with tab1:
        show_network_recommendations(transfer_analysis)
    
    with tab2:
        show_pattern_comparison(transfer_analysis)
    
    with tab3:
        show_network_cost_analysis(transfer_analysis)
    
    with tab4:
        show_network_timeline_analysis(transfer_analysis)
    
    with tab5:
        show_network_architecture(transfer_analysis)

def show_network_recommendations(transfer_analysis: Dict):
    """Show network recommendations"""
    
    st.markdown("### 🎯 AI-Powered Network Recommendations")
    
    # Fix: Add None check before accessing
    if transfer_analysis is None:
        st.error("No transfer analysis data available")
        return
    
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
            st.markdown(f"**Estimated Timeline:** {primary.get('estimated_timeline', 'TBD')}")
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
                st.markdown(f"**Trade-off:** {alt['trade_off']}")
    
    # Cost optimization tips
    cost_tips = recommendations.get('cost_optimization', [])
    
    if cost_tips:
        st.markdown("#### 💡 Cost Optimization Tips")
        for tip in cost_tips:
            st.markdown(f"• {tip}")
    
    # Risk considerations
    risks = recommendations.get('risk_considerations', [])
    
    if risks:
        st.markdown("#### ⚠️ Risk Considerations")
        for risk in risks:
            st.markdown(f"• {risk}")

def show_pattern_comparison(transfer_analysis: Dict):
    """Show pattern comparison visualizations"""
    
    st.markdown("### 📊 Network Pattern Comparison")
    
    # Comparison chart
    comparison_fig = create_network_comparison_chart(transfer_analysis)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Cost vs Duration optimization
    optimization_fig = create_cost_duration_optimization_chart(transfer_analysis)
    st.plotly_chart(optimization_fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("#### 📋 Detailed Metrics Comparison")
    
    analyzer = NetworkTransferAnalyzer()
    comparison_data = []
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        comparison_data.append({
            'Pattern': pattern_info['name'],
            'Total Cost': f"${metrics['total_cost']:,.0f}",
            'Transfer Duration': f"{metrics['transfer_time_days']:.1f} days",
            'Setup Cost': f"${metrics['setup_cost']:,.0f}",
            'Data Transfer Cost': f"${metrics['data_transfer_cost']:,.0f}",
            'Reliability Score': f"{metrics['reliability_score']}/100",
            'Security Score': f"{metrics['security_score']}/100",
            'Complexity': pattern_info['complexity']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def show_network_cost_analysis(transfer_analysis: Dict):
    """Show detailed network cost analysis"""
    
    st.markdown("### 💰 Network Cost Analysis")
    
    # Cost breakdown for each pattern
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
                st.metric("Cost per GB", f"${metrics['total_cost']/st.session_state.migration_params.get('data_size_gb', 1000):.2f}")
                
                # ROI calculation
                migration_params = st.session_state.migration_params
                data_size = migration_params.get('data_size_gb', 1000)
                
                # Estimate ongoing monthly savings (simplified)
                monthly_savings = data_size * 0.05  # Assume $0.05/GB monthly savings
                roi_months = metrics['total_cost'] / monthly_savings if monthly_savings > 0 else float('inf')
                
                if roi_months < 100:
                    st.metric("ROI Timeline", f"{roi_months:.1f} months")
                else:
                    st.metric("ROI Timeline", "Not applicable")

def show_network_timeline_analysis(transfer_analysis: Dict):
    """Show network timeline analysis"""
    
    st.markdown("### ⏱️ Timeline Analysis")
    
    recommendations = transfer_analysis.get('recommendations', {})
    timeline_impact = recommendations.get('timeline_impact', {})
    
    if timeline_impact:
        col1, col2 = st.columns(2)
        
        with col1:
            fastest = timeline_impact.get('fastest_option', {})
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #38a169;">
                <div class="metric-value" style="color: #38a169;">
                    ⚡ Fastest Option
                </div>
                <div class="metric-label">{fastest.get('pattern', 'N/A')}</div>
                <div style="margin-top: 10px;">
                    Duration: {fastest.get('duration_days', 0):.1f} days<br>
                    Timeline Usage: {fastest.get('timeline_utilization', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            slowest = timeline_impact.get('slowest_option', {})
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e53e3e;">
                <div class="metric-value" style="color: #e53e3e;">
                    🐌 Slowest Option
                </div>
                <div class="metric-label">{slowest.get('pattern', 'N/A')}</div>
                <div style="margin-top: 10px;">
                    Duration: {slowest.get('duration_days', 0):.1f} days<br>
                    Timeline Usage: {slowest.get('timeline_utilization', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"💡 {timeline_impact.get('recommendation', '')}")
    
    # Timeline comparison chart
    patterns = []
    durations = []
    colors = []
    
    migration_timeline_days = st.session_state.migration_params.get('migration_timeline_weeks', 12) * 7
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        analyzer = NetworkTransferAnalyzer()
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        patterns.append(pattern_info['name'])
        durations.append(metrics['transfer_time_days'])
        
        # Color based on timeline fit
        if metrics['transfer_time_days'] <= migration_timeline_days * 0.3:
            colors.append('green')
        elif metrics['transfer_time_days'] <= migration_timeline_days * 0.5:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=patterns,
        y=durations,
        marker_color=colors,
        text=[f"{d:.1f} days" for d in durations],
        textposition='auto'
    ))
    
    # Add timeline constraint line
    fig.add_hline(
        y=migration_timeline_days * 0.3,
        line_dash="dash",
        line_color="green",
        annotation_text="Recommended (30% of timeline)"
    )
    
    fig.update_layout(
        title='Transfer Duration vs Timeline Constraints',
        xaxis_title='Network Pattern',
        yaxis_title='Duration (days)',
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_network_architecture(transfer_analysis: Dict):
    """Show network architecture diagrams"""
    
    st.markdown("### 🏗️ Network Architecture")
    
    # Pattern selector
    analyzer = NetworkTransferAnalyzer()
    pattern_options = list(analyzer.transfer_patterns.keys())
    pattern_names = [analyzer.transfer_patterns[p]['name'] for p in pattern_options]
    
    selected_pattern_name = st.selectbox(
        "Select Pattern to Visualize",
        pattern_names
    )
    
    selected_pattern_id = pattern_options[pattern_names.index(selected_pattern_name)]
    
    # Architecture diagram
    arch_fig = create_network_architecture_diagram(selected_pattern_id)
    st.plotly_chart(arch_fig, use_container_width=True)
    
    # Pattern details
    pattern_info = analyzer.transfer_patterns[selected_pattern_id]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Components")
        for component in pattern_info['components']:
            st.markdown(f"• {component}")
    
    with col2:
        st.markdown("#### 📋 Use Cases")
        for use_case in pattern_info['use_cases']:
            st.markdown(f"• {use_case}")
    
    # Pros and Cons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Advantages")
        for pro in pattern_info['pros']:
            st.markdown(f"• {pro}")
    
    with col2:
        st.markdown("#### ⚠️ Considerations")
        for con in pattern_info['cons']:
            st.markdown(f"• {con}")
    
    # Technical specifications
    if selected_pattern_id in transfer_analysis:
        metrics = transfer_analysis[selected_pattern_id]
        
        st.markdown("#### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bandwidth Utilization", f"{metrics['bandwidth_utilization']}%")
        
        with col2:
            st.metric("Reliability Score", f"{metrics['reliability_score']}/100")
        
        with col3:
            st.metric("Security Score", f"{metrics['security_score']}/100")
        
        with col4:
            st.metric("Complexity Level", pattern_info['complexity'])

# Add this to the main navigation in the original app
def add_network_module_to_main_app():
    """Instructions for adding network module to main app"""
    
    # Add this to the sidebar radio options:
    # "🌐 Network Analysis"
    
    # Add this to the main content section:
    # elif page == "🌐 Network Analysis":
    #     show_network_transfer_analysis()
    
    pass

# ===========================
# RISK ASSESSMENT FUNCTIONS
# ===========================

def calculate_migration_risks(migration_params: Dict, recommendations: Dict) -> Dict:
    """Calculate comprehensive migration risk assessment"""
    
    source_engine = migration_params.get('source_engine', '')
    target_engine = migration_params.get('target_engine', '')
    data_size_gb = migration_params.get('data_size_gb', 0)
    
    # Technical risks
    technical_risks = {
        'engine_compatibility': _assess_engine_compatibility(source_engine, target_engine),
        'data_migration_complexity': _assess_data_complexity(data_size_gb),
        'application_integration': _assess_application_risks(migration_params),
        'performance_risk': _assess_performance_risks(recommendations)
    }
    
    # Business risks
    business_risks = {
        'timeline_risk': _assess_timeline_risks(migration_params),
        'cost_overrun_risk': _assess_cost_risks(migration_params),
        'business_continuity': _assess_continuity_risks(recommendations),
        'resource_availability': _assess_resource_risks(migration_params)
    }
    
    # Calculate overall risk score
    tech_score = sum(technical_risks.values()) / len(technical_risks)
    business_score = sum(business_risks.values()) / len(business_risks)
    overall_score = (tech_score + business_score) / 2
    
    return {
        'overall_score': overall_score,
        'risk_level': _get_risk_level(overall_score),
        'technical_risks': technical_risks,
        'business_risks': business_risks,
        'mitigation_strategies': _generate_mitigation_strategies(technical_risks, business_risks)
    }

def _assess_engine_compatibility(source: str, target: str) -> float:
    """Assess engine compatibility risk (0-100)"""
    compatibility_matrix = {
        ('oracle-ee', 'postgres'): 75,
        ('oracle-ee', 'aurora-postgresql'): 65,
        ('oracle-ee', 'oracle-ee'): 15,
        ('postgres', 'aurora-postgresql'): 20,
        ('postgres', 'postgres'): 10,
        ('mysql', 'aurora-mysql'): 15
    }
    return compatibility_matrix.get((source, target), 80)

def _assess_data_complexity(data_size_gb: int) -> float:
    """Assess data migration complexity risk"""
    if data_size_gb < 100:
        return 20
    elif data_size_gb < 1000:
        return 40
    elif data_size_gb < 10000:
        return 60
    else:
        return 85

def _assess_application_risks(migration_params: Dict) -> float:
    """Assess application integration risks"""
    num_applications = migration_params.get('num_applications', 1)
    return min(90, 20 + (num_applications * 15))

def _assess_performance_risks(recommendations: Dict) -> float:
    """Assess performance-related risks"""
    prod_envs = [env for env, rec in recommendations.items() 
                if rec['environment_type'] == 'production']
    
    if not prod_envs:
        return 30
    
    # Check if production environments are adequately sized
    prod_rec = recommendations[prod_envs[0]]
    if 'xlarge' in prod_rec['instance_class']:
        return 25
    elif 'large' in prod_rec['instance_class']:
        return 45
    else:
        return 70

def _assess_timeline_risks(migration_params: Dict) -> float:
    """Assess timeline-related risks"""
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    data_size_gb = migration_params.get('data_size_gb', 1000)
    
    # Risk increases with larger data and shorter timelines
    complexity_factor = data_size_gb / 1000
    time_pressure = max(0, (16 - timeline_weeks) * 5)
    
    return min(95, 20 + complexity_factor + time_pressure)

def _assess_cost_risks(migration_params: Dict) -> float:
    """Assess cost overrun risks"""
    migration_budget = migration_params.get('migration_budget', 500000)
    estimated_cost = migration_params.get('estimated_migration_cost', 300000)
    
    if migration_budget <= 0:
        return 80
    
    budget_ratio = estimated_cost / migration_budget
    if budget_ratio < 0.7:
        return 20
    elif budget_ratio < 0.9:
        return 40
    elif budget_ratio < 1.1:
        return 60
    else:
        return 90

def _assess_continuity_risks(recommendations: Dict) -> float:
    """Assess business continuity risks"""
    prod_count = len([env for env, rec in recommendations.items() 
                     if rec['environment_type'] == 'production'])
    
    if prod_count == 0:
        return 10
    elif prod_count == 1:
        return 45
    else:
        return 60  # Multiple production environments increase coordination complexity

def _assess_resource_risks(migration_params: Dict) -> float:
    """Assess resource availability risks"""
    team_size = migration_params.get('team_size', 5)
    expertise_level = migration_params.get('team_expertise', 'medium')
    
    base_risk = 50
    
    if team_size < 3:
        base_risk += 20
    elif team_size > 8:
        base_risk -= 10
    
    if expertise_level == 'high':
        base_risk -= 20
    elif expertise_level == 'low':
        base_risk += 25
    
    return max(10, min(90, base_risk))

def _get_risk_level(score: float) -> Dict:
    """Get risk level description"""
    if score < 30:
        return {'level': 'Low', 'color': '#38a169', 'action': 'Standard monitoring'}
    elif score < 50:
        return {'level': 'Medium', 'color': '#d69e2e', 'action': 'Active mitigation required'}
    elif score < 70:
        return {'level': 'High', 'color': '#e53e3e', 'action': 'Immediate action required'}
    else:
        return {'level': 'Critical', 'color': '#9f1239', 'action': 'Project at risk - urgent intervention needed'}

def _generate_mitigation_strategies(technical_risks: Dict, business_risks: Dict) -> List[Dict]:
    """Generate risk mitigation strategies"""
    strategies = []
    
    # Technical risk mitigations
    if technical_risks['engine_compatibility'] > 60:
        strategies.append({
            'risk': 'Engine Compatibility',
            'strategy': 'Conduct comprehensive schema assessment and implement AWS SCT',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    if technical_risks['data_migration_complexity'] > 50:
        strategies.append({
            'risk': 'Data Migration Complexity',
            'strategy': 'Implement incremental migration with AWS DMS and validation scripts',
            'timeline': '1-2 weeks setup',
            'cost_impact': 'Low'
        })
    
    # Business risk mitigations
    if business_risks['timeline_risk'] > 60:
        strategies.append({
            'risk': 'Timeline Pressure',
            'strategy': 'Add parallel migration streams and increase team capacity',
            'timeline': 'Immediate',
            'cost_impact': 'High'
        })
    
    if business_risks['cost_overrun_risk'] > 60:
        strategies.append({
            'risk': 'Cost Overrun',
            'strategy': 'Implement strict budget controls and scope management',
            'timeline': 'Ongoing',
            'cost_impact': 'Low'
        })
    
    return strategies

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def create_cost_waterfall_chart(cost_analysis: Dict) -> go.Figure:
    """Create cost transformation waterfall chart"""
    
    current_costs = cost_analysis.get('current_total_cost', 0)
    aws_costs = cost_analysis['annual_aws_cost']
    migration_costs = cost_analysis['migration_costs']['total']
    
    # Waterfall components
    categories = ['Current Infrastructure', 'Migration Investment', 'AWS Annual Cost', 'Net Position']
    values = [current_costs, -migration_costs, -aws_costs, current_costs - migration_costs - aws_costs]
    
    fig = go.Figure()
    
    # Create waterfall effect
    cumulative = [current_costs]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(values[-1])
    
    colors = ['blue', 'red', 'orange', 'green' if values[-1] > 0 else 'red']
    
    for i, (category, value, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            x=[category],
            y=[abs(value)],
            name=category,
            marker_color=color,
            text=f'${abs(value):,.0f}',
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Cost Transformation Analysis',
        xaxis_title='Cost Components',
        yaxis_title='Annual Cost ($)',
        showlegend=False,
        height=500
    )
    
    return fig

def create_risk_heatmap(risk_assessment: Dict) -> go.Figure:
    """Create risk assessment heatmap"""
    
    # Prepare risk data
    tech_risks = risk_assessment['technical_risks']
    business_risks = risk_assessment['business_risks']
    
    risk_categories = list(tech_risks.keys()) + list(business_risks.keys())
    risk_scores = list(tech_risks.values()) + list(business_risks.values())
    risk_types = ['Technical'] * len(tech_risks) + ['Business'] * len(business_risks)
    
    # Create heatmap data
    heatmap_data = []
    for i, (category, score, risk_type) in enumerate(zip(risk_categories, risk_scores, risk_types)):
        heatmap_data.append([score])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Risk Score'],
        y=[cat.replace('_', ' ').title() for cat in risk_categories],
        colorscale=[
            [0, '#38a169'],      # Green
            [0.3, '#d69e2e'],    # Yellow
            [0.6, '#e53e3e'],    # Red
            [1, '#9f1239']       # Dark red
        ],
        text=[[f'{score:.0f}' for score in row] for row in heatmap_data],
        texttemplate='%{text}',
        colorbar=dict(title="Risk Level")
    ))
    
    fig.update_layout(
        title='Migration Risk Assessment',
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_environment_comparison_chart(environment_costs: Dict) -> go.Figure:
    """Create environment cost comparison chart"""
    
    environments = list(environment_costs.keys())
    instance_costs = [env['instance_cost'] for env in environment_costs.values()]
    storage_costs = [env['storage_cost'] for env in environment_costs.values()]
    total_costs = [env['total_monthly'] for env in environment_costs.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Instance Cost',
        x=environments,
        y=instance_costs,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Storage Cost',
        x=environments,
        y=storage_costs,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Monthly Cost by Environment',
        xaxis_title='Environment',
        yaxis_title='Monthly Cost ($)',
        barmode='stack',
        height=400
    )
    
    return fig

def create_migration_timeline_gantt(migration_params: Dict) -> go.Figure:
    """Create migration timeline Gantt chart"""
    
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    
    # Define migration phases
    phases = [
        {'name': 'Assessment & Planning', 'start': 0, 'duration': 2},
        {'name': 'Environment Setup', 'start': 1, 'duration': 3},
        {'name': 'Schema Migration', 'start': 3, 'duration': 2},
        {'name': 'Data Migration', 'start': 4, 'duration': 4},
        {'name': 'Application Testing', 'start': 6, 'duration': 3},
        {'name': 'User Acceptance Testing', 'start': 8, 'duration': 2},
        {'name': 'Go-Live Preparation', 'start': 9, 'duration': 2},
        {'name': 'Production Cutover', 'start': 11, 'duration': 1}
    ]
    
    # Create simple bar chart instead of Gantt to avoid dependencies
    fig = go.Figure()
    
    for i, phase in enumerate(phases):
        fig.add_trace(go.Bar(
            name=phase['name'],
            x=[phase['name']],
            y=[phase['duration']],
            text=f"{phase['duration']} weeks",
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Migration Phase Duration',
        xaxis_title='Phase',
        yaxis_title='Duration (weeks)',
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

# ===========================
# PDF REPORT GENERATION
# ===========================

def generate_executive_summary_pdf(analysis_results: Dict, migration_params: Dict) -> io.BytesIO:
    """Generate executive summary PDF report"""
    
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
    
    # Key metrics table
    cost_analysis = analysis_results
    migration_costs = cost_analysis['migration_costs']
    
    metrics_data = [
        ['Metric', 'Value', 'Impact'],
        ['Monthly AWS Cost', f"${cost_analysis['monthly_aws_cost']:,.0f}", 'Operational'],
        ['Annual AWS Cost', f"${cost_analysis['annual_aws_cost']:,.0f}", 'Budget Planning'],
        ['Migration Investment', f"${migration_costs['total']:,.0f}", 'One-time'],
        ['ROI Timeline', '18-24 months', 'Financial'],
        ['Risk Level', 'Medium', 'Manageable']
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
        "Plan for 12-16 week migration timeline with 2-week buffer",
        "Establish comprehensive testing protocols for each environment",
        "Consider Aurora for production workloads to optimize performance and cost"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    # Next steps
    story.append(Spacer(1, 20))
    story.append(Paragraph("Next Steps", styles['Heading2']))
    
    next_steps = [
        "1. Obtain stakeholder approval and budget allocation",
        "2. Form migration team and assign roles",
        "3. Set up AWS infrastructure and migration tools",
        "4. Begin with development environment migration",
        "5. Execute production cutover after successful testing"
    ]
    
    for step in next_steps:
        story.append(Paragraph(step, styles['Normal']))
        story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_technical_report_pdf(analysis_results: Dict, recommendations: Dict, migration_params: Dict) -> io.BytesIO:
    """Generate detailed technical PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("AWS Database Migration - Technical Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    # Migration parameters
    story.append(Paragraph("Migration Configuration", styles['Heading2']))
    
    config_data = [
        ['Parameter', 'Value'],
        ['Source Engine', migration_params.get('source_engine', 'N/A')],
        ['Target Engine', migration_params.get('target_engine', 'N/A')],
        ['Data Size', f"{migration_params.get('data_size_gb', 0):,} GB"],
        ['Timeline', f"{migration_params.get('migration_timeline_weeks', 0)} weeks"],
        ['Environments', str(len(recommendations))],
        ['Migration Type', DatabaseEngine.get_migration_type(
            migration_params.get('source_engine', ''),
            migration_params.get('target_engine', '')
        )]
    ]
    
    config_table = Table(config_data, colWidths=[2.5*inch, 2*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(config_table)
    story.append(PageBreak())
    
    # Environment recommendations
    story.append(Paragraph("Environment Recommendations", styles['Heading2']))
    
    env_data = [['Environment', 'Instance Class', 'CPU/RAM', 'Storage', 'Multi-AZ']]
    
    for env_name, rec in recommendations.items():
        env_data.append([
            env_name,
            rec['instance_class'],
            f"{rec['cpu_cores']} vCPU / {rec['ram_gb']} GB",
            f"{rec['storage_gb']} GB",
            'Yes' if rec['multi_az'] else 'No'
        ])
    
    env_table = Table(env_data, colWidths=[1.2*inch, 1.3*inch, 1.2*inch, 1*inch, 0.8*inch])
    env_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    story.append(env_table)
    story.append(Spacer(1, 20))
    
    # Cost breakdown
    story.append(Paragraph("Cost Analysis", styles['Heading2']))
    
    env_costs = analysis_results['environment_costs']
    cost_data = [['Environment', 'Instance', 'Storage', 'Backup', 'Total Monthly']]
    
    for env_name, costs in env_costs.items():
        cost_data.append([
            env_name,
            f"${costs['instance_cost']:,.0f}",
            f"${costs['storage_cost']:,.0f}",
            f"${costs['backup_cost']:,.0f}",
            f"${costs['total_monthly']:,.0f}"
        ])
    
    cost_table = Table(cost_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1.2*inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e53e3e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    story.append(cost_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ===========================
# STREAMLIT APPLICATION
# ===========================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'environment_specs': {},
        'migration_params': {},
        'network_analysis': None,        # <-- ADD THIS LINE
        'transfer_analysis': None,       # <-- ADD THIS LINE
        'vrops_analysis': None,        # ADD THIS
        'vrops_analyzer': None,       # ADD THIS
        'analysis_results': None,
        'recommendations': None,
        'risk_assessment': None,
        'ai_insights': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    """Main Streamlit application"""
    
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="enterprise-header">
        <h1>🚀 Enterprise AWS Database Migration Tool</h1>
        <p>AI-Powered Analysis • Real-time AWS Pricing • Comprehensive Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Select Section:",
            [
                "🔧 Migration Configuration",
                "📊 Environment Setup",
                "🌐 Network Analysis",
                "🚀 Analysis & Recommendations",
                "📈 Results Dashboard",
                "📄 Reports & Export"
            ]
        )
        
        # Status indicators
        st.markdown("### 📋 Status")
        
        if st.session_state.environment_specs:
            st.success(f"✅ {len(st.session_state.environment_specs)} environments configured")
        else:
            st.warning("⚠️ Configure environments")
        
        if st.session_state.migration_params:
            st.success("✅ Migration parameters set")
        else:
            st.warning("⚠️ Set migration parameters")
        
        if st.session_state.analysis_results:
            st.success("✅ Analysis complete")
            # Quick metrics
            results = st.session_state.analysis_results
            st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
            st.metric("Migration Cost", f"${results['migration_costs']['total']:,.0f}")
        else:
            st.info("ℹ️ Analysis pending")
        
        # In the sidebar status section, add:
        if st.session_state.transfer_analysis:
            st.success("✅ Network analysis complete")
            # Quick network metrics
            recommendations = st.session_state.transfer_analysis.get('recommendations', {})
            primary = recommendations.get('primary_recommendation', {})
            if primary:
                st.metric("Recommended Pattern", primary.get('pattern_name', 'N/A'))
        else:
            st.info("ℹ️ Network analysis pending")
        
        if st.session_state.vrops_analysis:
            st.success("✅ vROps analysis complete")
            
            # Show health scores
            health_scores = []
            for env_name, analysis in st.session_state.vrops_analysis.items():
                if isinstance(analysis, dict) and 'performance_scores' in analysis:
                    health_scores.append(analysis['performance_scores'].get('overall_health', 0))
            
            if health_scores:
                avg_health = sum(health_scores) / len(health_scores)
                st.metric("Avg Health Score", f"{avg_health:.1f}/100")
        else:
            st.info("ℹ️ vROps analysis pending")
        
    # Main content
    if page == "🔧 Migration Configuration":
        show_migration_configuration()
    elif page == "📊 Environment Setup":
        show_enhanced_environment_setup()
    elif page == "🌐 Network Analysis":          # <-- ADD THIS SECTION
        show_network_transfer_analysis()
    elif page == "🚀 Analysis & Recommendations":
        show_analysis_section()
    elif page == "📈 Results Dashboard":
        show_results_dashboard()
    elif page == "📄 Reports & Export":
        show_reports_section()

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
                complexity = DatabaseEngine.get_complexity_multiplier(source_engine, target_engine)
                
                st.markdown(f"""
                **Migration Type:** {migration_type.title()}  
                **Complexity Factor:** {complexity:.1f}x  
                **Estimated Effort:** {'Low' if complexity < 1.5 else 'Medium' if complexity < 2.0 else 'High'}
                """)
    
    # Migration parameters
    st.markdown("### ⚙️ Migration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💾 Data Configuration")
        data_size_gb = st.number_input("Total Data Size (GB)", min_value=1, max_value=100000, value=1000)
        num_applications = st.number_input("Connected Applications", min_value=1, max_value=50, value=3)
        num_stored_procedures = st.number_input("Stored Procedures/Functions", min_value=0, max_value=10000, value=50)
    
    with col2:
        st.markdown("#### ⏱️ Timeline & Resources")
        migration_timeline_weeks = st.slider("Migration Timeline (weeks)", min_value=4, max_value=52, value=12)
        team_size = st.number_input("Team Size", min_value=2, max_value=20, value=5)
        team_expertise = st.selectbox("Team Expertise Level", ["low", "medium", "high"], index=1)
    
    with col3:
        st.markdown("#### 🌐 Infrastructure")
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"], index=0)
        use_direct_connect = st.checkbox("Use AWS Direct Connect", value=True)
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
            'num_stored_procedures': num_stored_procedures,
            'migration_timeline_weeks': migration_timeline_weeks,
            'team_size': team_size,
            'team_expertise': team_expertise,
            'region': region,
            'use_direct_connect': use_direct_connect,
            'bandwidth_mbps': bandwidth_mbps,
            'migration_budget': migration_budget,
            'anthropic_api_key': anthropic_api_key,
            'estimated_migration_cost': 0  # Will be calculated
        }
        
        st.success("✅ Configuration saved! Proceed to Environment Setup.")
        st.balloons()

def show_environment_setup():
    """Show environment setup interface"""
    
    st.markdown("## 📊 Environment Configuration")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Environment configuration options
    config_method = st.radio(
        "Configuration Method:",
        ["📝 Manual Entry", "📁 Bulk Upload"],
        horizontal=True
    )
    
    if config_method == "📁 Bulk Upload":
        show_bulk_upload_interface()
    else:
        show_manual_environment_setup()

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

def show_manual_environment_setup():
    """Show manual environment setup interface"""
    
    st.markdown("### 📝 Manual Environment Configuration")
    
    # Number of environments
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    # Environment configuration
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

def show_analysis_section():
    """Show analysis and recommendations section"""
    
    st.markdown("## 🚀 Migration Analysis & Recommendations")
    
    # Check prerequisites
    if not st.session_state.migration_params:
        st.error("❌ Migration configuration required")
        return
    
    if not st.session_state.environment_specs:
        st.error("❌ Environment configuration required")
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
        for env_name, specs in envs.items():
            cpu_cores = specs.get('cpu_cores', 'N/A')
            ram_gb = specs.get('ram_gb', 'N/A')
            st.markdown(f"• **{env_name}:** {specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM")
    
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
        
        # Update migration params with estimated cost
        st.session_state.migration_params['estimated_migration_cost'] = cost_analysis['migration_costs']['total']
        
        st.session_state.analysis_results = cost_analysis
        
        # Step 3: Risk assessment
        st.write("⚠️ Assessing risks...")
        risk_assessment = calculate_migration_risks(st.session_state.migration_params, recommendations)
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: AI insights (if available)
        if anthropic_api_key:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = asyncio.run(analyzer.generate_ai_insights(cost_analysis, st.session_state.migration_params))
                st.session_state.ai_insights = ai_insights
            except Exception as e:
                st.warning(f"AI insights generation failed: {str(e)}")
                st.session_state.ai_insights = {'error': str(e)}
        
        st.success("✅ Analysis complete! Check the Results Dashboard.")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.code(str(e))

def show_results_dashboard():
    """Show comprehensive results dashboard"""
    
    st.markdown("## 📈 Migration Analysis Results")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Please run the analysis first.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Cost Summary",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "📊 Visualizations",
        "🤖 AI Insights",
        "📅 Timeline"
    ])
    
    with tab1:
        show_cost_summary()
    
    with tab2:
        show_risk_assessment()
    
    with tab3:
        show_environment_analysis()
    
    with tab4:
        show_visualizations()
    
    with tab5:
        show_ai_insights()
    
    with tab6:
        show_timeline_analysis()

def show_cost_summary():
    """Show cost summary dashboard"""
    
    st.markdown("### 💰 Cost Analysis Summary")
    
    results = st.session_state.analysis_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Monthly AWS Cost",
            f"${results['monthly_aws_cost']:,.0f}",
            delta=f"${results['monthly_aws_cost']*12:,.0f}/year"
        )
    
    with col2:
        st.metric(
            "Migration Investment",
            f"${results['migration_costs']['total']:,.0f}",
            delta="One-time cost"
        )
    
    with col3:
        # Calculate simple ROI based on current vs new costs
        current_estimated = results['monthly_aws_cost'] * 1.8  # Assume 80% higher current costs
        annual_savings = (current_estimated - results['monthly_aws_cost']) * 12
        roi_years = results['migration_costs']['total'] / max(annual_savings, 1)
        
        st.metric(
            "ROI Timeline",
            f"{roi_years:.1f} years",
            delta=f"${annual_savings:,.0f}/year savings"
        )
    
    with col4:
        total_3_year = results['migration_costs']['total'] + (results['annual_aws_cost'] * 3)
        st.metric(
            "3-Year Total Investment",
            f"${total_3_year:,.0f}",
            delta="Including migration"
        )
    
    # Environment cost breakdown
    st.markdown("### 🏢 Environment Cost Breakdown")
    
    env_costs = results['environment_costs']
    env_data = []
    
    for env_name, costs in env_costs.items():
        env_data.append({
            'Environment': env_name,
            'Instance Cost': f"${costs['instance_cost']:,.0f}",
            'Storage Cost': f"${costs['storage_cost']:,.0f}",
            'Backup Cost': f"${costs['backup_cost']:,.0f}",
            'Monthly Total': f"${costs['total_monthly']:,.0f}",
            'Annual Total': f"${costs['total_monthly']*12:,.0f}"
        })
    
    env_df = pd.DataFrame(env_data)
    st.dataframe(env_df, use_container_width=True)
    
    # Migration cost breakdown
    st.markdown("### 🚀 Migration Cost Breakdown")
    
    migration_costs = results['migration_costs']
    migration_data = {
        'Cost Component': [
            'DMS Instance',
            'Data Transfer',
            'Professional Services',
            'Contingency (20%)',
            'Total Migration Cost'
        ],
        'Amount': [
            f"${migration_costs['dms_instance']:,.0f}",
            f"${migration_costs['data_transfer']:,.0f}",
            f"${migration_costs['professional_services']:,.0f}",
            f"${migration_costs['contingency']:,.0f}",
            f"${migration_costs['total']:,.0f}"
        ],
        'Description': [
            'DMS replication instance for data migration',
            'Network costs for data transfer to AWS',
            'Migration team and project management',
            'Risk buffer for unexpected costs',
            'Total one-time migration investment'
        ]
    }
    
    migration_df = pd.DataFrame(migration_data)
    st.dataframe(migration_df, use_container_width=True)

def show_risk_assessment():
    """Show risk assessment dashboard"""
    
    st.markdown("### ⚠️ Migration Risk Assessment")
    
    if not st.session_state.risk_assessment:
        st.warning("Risk assessment not available")
        return
    
    risk_assessment = st.session_state.risk_assessment
    
    # Overall risk level
    risk_level = risk_assessment['risk_level']
    
    st.markdown(f"""
    <div class="metric-card risk-{risk_level['level'].lower()}">
        <div class="metric-value" style="color: {risk_level['color']};">
            {risk_level['level']} Risk
        </div>
        <div class="metric-label">Overall Migration Risk</div>
        <div style="margin-top: 10px; font-weight: 500;">
            Action Required: {risk_level['action']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Technical Risks")
        tech_risks = risk_assessment['technical_risks']
        
        for risk_name, score in tech_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#38a169'
            
            st.markdown(f"""
            **{risk_name.replace('_', ' ').title()}**  
            <div style="background: {color}; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block;">
                {score:.0f}/100 - {risk_level_detail}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
    
    with col2:
        st.markdown("#### 💼 Business Risks")
        business_risks = risk_assessment['business_risks']
        
        for risk_name, score in business_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#38a169'
            
            st.markdown(f"""
            **{risk_name.replace('_', ' ').title()}**  
            <div style="background: {color}; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block;">
                {score:.0f}/100 - {risk_level_detail}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
    
    # Risk mitigation strategies
    st.markdown("#### 🛡️ Risk Mitigation Strategies")
    
    mitigation_strategies = risk_assessment['mitigation_strategies']
    
    if mitigation_strategies:
        for strategy in mitigation_strategies:
            with st.expander(f"🎯 {strategy['risk']} Mitigation"):
                st.markdown(f"**Strategy:** {strategy['strategy']}")
                st.markdown(f"**Timeline:** {strategy['timeline']}")
                st.markdown(f"**Cost Impact:** {strategy['cost_impact']}")
    else:
        st.info("No specific mitigation strategies required - risk levels are manageable with standard best practices.")

def show_environment_analysis():
    """Show environment analysis dashboard"""
    
    st.markdown("### 🏢 Environment Analysis")
    
    if not st.session_state.recommendations:
        st.warning("Environment analysis not available")
        return
    
    recommendations = st.session_state.recommendations
    environment_specs = st.session_state.environment_specs
    
    # Environment comparison
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs[env_name]
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec['environment_type'].title(),
            'Current Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Recommended Instance': rec['instance_class'],
            'Storage': f"{specs['storage_gb']} GB",
            'Multi-AZ': 'Yes' if rec['multi_az'] else 'No',
            'Daily Usage': f"{specs['daily_usage_hours']} hours"
        })
    
    env_df = pd.DataFrame(env_comparison_data)
    st.dataframe(env_df, use_container_width=True)
    
    # Environment-specific insights
    st.markdown("#### 💡 Environment Insights")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🔍 {env_name} Environment Details"):
            specs = environment_specs[env_name]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Configuration**")
                st.write(f"CPU Cores: {specs['cpu_cores']}")
                st.write(f"RAM: {specs['ram_gb']} GB")
                st.write(f"Storage: {specs['storage_gb']} GB")
                st.write(f"Daily Usage: {specs['daily_usage_hours']} hours")
            
            with col2:
                st.markdown("**AWS Recommendation**")
                st.write(f"Instance: {rec['instance_class']}")
                st.write(f"Environment Type: {rec['environment_type'].title()}")
                st.write(f"Multi-AZ: {'Yes' if rec['multi_az'] else 'No'}")
                
                # Get instance specs (simplified)
                if 'xlarge' in rec['instance_class']:
                    aws_cpu = 16 if '4xlarge' in rec['instance_class'] else 8 if '2xlarge' in rec['instance_class'] else 4
                    aws_ram = aws_cpu * 8
                elif 'large' in rec['instance_class']:
                    aws_cpu = 2
                    aws_ram = 8
                else:
                    aws_cpu = 1
                    aws_ram = 4
                
                st.write(f"AWS vCPUs: {aws_cpu}")
                st.write(f"AWS RAM: {aws_ram} GB")
            
            with col3:
                st.markdown("**Optimization Notes**")
                
                # Generate optimization suggestions
                if rec['environment_type'] == 'production':
                    st.write("✅ Production-grade configuration")
                    st.write("✅ Multi-AZ for high availability")
                elif rec['environment_type'] == 'development':
                    st.write("💡 Cost-optimized for development")
                    st.write("💡 Single-AZ to reduce costs")
                
                if specs['daily_usage_hours'] < 12:
                    st.write("⚡ Consider Aurora Serverless for variable workloads")

def show_visualizations():
    """Show visualization dashboard"""
    
    st.markdown("### 📊 Migration Analysis Visualizations")
    
    results = st.session_state.analysis_results
    
    # Cost waterfall chart
    st.markdown("#### 💧 Cost Transformation Analysis")
    
    # Create mock current costs for comparison
    current_total_cost = results['annual_aws_cost'] * 1.8  # Assume 80% higher current costs
    results['current_total_cost'] = current_total_cost
    
    waterfall_fig = create_cost_waterfall_chart(results)
    st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Environment cost comparison
    st.markdown("#### 🏢 Environment Cost Comparison")
    
    env_comparison_fig = create_environment_comparison_chart(results['environment_costs'])
    st.plotly_chart(env_comparison_fig, use_container_width=True)
    
    # Risk heatmap
    if st.session_state.risk_assessment:
        st.markdown("#### 🔥 Risk Assessment Heatmap")
        
        risk_heatmap_fig = create_risk_heatmap(st.session_state.risk_assessment)
        st.plotly_chart(risk_heatmap_fig, use_container_width=True)

def show_ai_insights():
    """Show AI insights dashboard"""
    
    st.markdown("### 🤖 AI-Powered Insights")
    
    ai_insights = st.session_state.ai_insights
    
    if not ai_insights:
        st.info("💡 AI insights not available. Provide an Anthropic API key in the configuration to enable AI analysis.")
        return
    
    if 'error' in ai_insights:
        st.error(f"❌ AI analysis failed: {ai_insights['error']}")
        return
    
    # Display AI insights
    st.markdown("""
    <div class="ai-insight-card">
        <h3>🤖 AI Migration Analysis</h3>
        <p>Powered by Claude AI for intelligent migration insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract insights
    for key, value in ai_insights.items():
        if key != 'error':
            st.markdown(f"#### {key.replace('_', ' ').title()}")
            
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    st.markdown(f"**{sub_key.replace('_', ' ').title()}:** {sub_value}")
            else:
                st.markdown(str(value))
            
            st.markdown("---")

def show_timeline_analysis():
    """Show timeline analysis dashboard"""
    
    st.markdown("### 📅 Migration Timeline Analysis")
    
    migration_params = st.session_state.migration_params
    
    # Timeline Gantt chart
    gantt_fig = create_migration_timeline_gantt(migration_params)
    st.plotly_chart(gantt_fig, use_container_width=True)
    
    # Timeline summary
    timeline_weeks = migration_params['migration_timeline_weeks']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Timeline", f"{timeline_weeks} weeks")
        st.metric("Planning Phase", "2-3 weeks")
    
    with col2:
        st.metric("Migration Phase", f"{timeline_weeks//2} weeks")
        st.metric("Testing Phase", "3-4 weeks")
    
    with col3:
        st.metric("Go-Live Phase", "1-2 weeks")
        st.metric("Team Size", f"{migration_params['team_size']} members")
    
    # Critical path analysis
    st.markdown("#### 🎯 Critical Path & Dependencies")
    
    critical_activities = [
        "Environment setup and infrastructure provisioning",
        "Schema migration and data validation",
        "Application code refactoring and testing",
        "Production cutover and go-live",
        "Post-migration optimization and monitoring"
    ]
    
    for i, activity in enumerate(critical_activities, 1):
        st.markdown(f"**{i}.** {activity}")

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
                pdf_buffer = generate_executive_summary_pdf(
                    st.session_state.analysis_results,
                    st.session_state.migration_params
                )
                
            st.download_button(
                label="📥 Download Executive Summary",
                data=pdf_buffer.getvalue(),
                file_name=f"AWS_Migration_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col2:
        st.markdown("#### 🔧 Technical Report")
        st.markdown("Detailed technical analysis for architects and engineers")
        st.markdown("**Includes:**")
        st.markdown("• Environment specifications")
        st.markdown("• Instance recommendations")
        st.markdown("• Detailed cost breakdown")
        st.markdown("• Technical considerations")
        
        if st.button("📄 Generate Technical PDF", key="tech_pdf", use_container_width=True):
            with st.spinner("Generating technical report..."):
                pdf_buffer = generate_technical_report_pdf(
                    st.session_state.analysis_results,
                    st.session_state.recommendations,
                    st.session_state.migration_params
                )
                
            st.download_button(
                label="📥 Download Technical Report",
                data=pdf_buffer.getvalue(),
                file_name=f"AWS_Migration_Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col3:
        st.markdown("#### 📊 Data Export")
        st.markdown("Raw data for further analysis")
        st.markdown("**Includes:**")
        st.markdown("• Cost analysis data")
        st.markdown("• Environment specifications")
        st.markdown("• Risk assessment data")
        st.markdown("• Recommendations")
        
        if st.button("📊 Export Data (CSV)", key="csv_export", use_container_width=True):
            # Prepare CSV data
            env_costs = st.session_state.analysis_results['environment_costs']
            
            csv_data = []
            for env_name, costs in env_costs.items():
                csv_data.append({
                    'Environment': env_name,
                    'Instance_Cost': costs['instance_cost'],
                    'Storage_Cost': costs['storage_cost'],
                    'Backup_Cost': costs['backup_cost'],
                    'Total_Monthly': costs['total_monthly'],
                    'Total_Annual': costs['total_monthly'] * 12
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Data",
                data=csv_string,
                file_name=f"AWS_Migration_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Bulk download option
    st.markdown("---")
    st.markdown("### 📦 Bulk Download")
    
    if st.button("📊 Generate All Reports", key="bulk_reports", use_container_width=True):
        with st.spinner("Generating all reports... This may take a moment..."):
            # Create ZIP file with all reports
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Executive summary
                exec_pdf = generate_executive_summary_pdf(
                    st.session_state.analysis_results,
                    st.session_state.migration_params
                )
                zip_file.writestr(
                    f"Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                    exec_pdf.getvalue()
                )
                
                # Technical report
                tech_pdf = generate_technical_report_pdf(
                    st.session_state.analysis_results,
                    st.session_state.recommendations,
                    st.session_state.migration_params
                )
                zip_file.writestr(
                    f"Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    tech_pdf.getvalue()
                )
                
                # CSV data
                env_costs = st.session_state.analysis_results['environment_costs']
                csv_data = []
                for env_name, costs in env_costs.items():
                    csv_data.append({
                        'Environment': env_name,
                        'Instance_Cost': costs['instance_cost'],
                        'Storage_Cost': costs['storage_cost'],
                        'Backup_Cost': costs['backup_cost'],
                        'Total_Monthly': costs['total_monthly'],
                        'Total_Annual': costs['total_monthly'] * 12
                    })
                
                csv_df = pd.DataFrame(csv_data)
                zip_file.writestr(
                    f"Migration_Analysis_Data_{datetime.now().strftime('%Y%m%d')}.csv",
                    csv_df.to_csv(index=False)
                )
            
            zip_buffer.seek(0)
            
        st.download_button(
            label="📥 Download All Reports (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"AWS_Migration_Complete_Analysis_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True
        )

if __name__ == "__main__":
    main()