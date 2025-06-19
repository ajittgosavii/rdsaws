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
import anthropic 
import streamlit as st
import pandas as pd

# PDF Generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from vrops_ui import show_enhanced_environment_setup_with_vrops, show_vrops_results_tab
#from vrops_ui import VRopsMetricsAnalyzer
#from reader_writer_optimizer import OptimizedReaderWriterAnalyzer, display_optimization_results
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

class StorageConfigurationManager:
    """Centralized storage configuration management"""
    
    def __init__(self):
        self.default_environment_multipliers = {
            'production': 1.5,      # Production needs more storage for safety
            'staging': 1.0,         # Staging matches production data
            'testing': 0.7,         # Testing with subset of data
            'qa': 0.6,              # QA with sample data  
            'development': 0.3      # Development with minimal data
        }
        self.growth_buffer = 0.3    # 30% growth buffer
        self.replication_overhead = 0.1  # 10% for transaction logs, backups
    
    def calculate_recommended_storage(self, migration_data_gb: int, environment_type: str, 
                                    environment_count: int = 1) -> Dict:
        """Calculate recommended storage for an environment"""
        
        env_type = environment_type.lower()
        base_multiplier = self.default_environment_multipliers.get(env_type, 1.0)
        
        # Base storage requirement
        base_storage = int(migration_data_gb * base_multiplier)
        
        # Add growth buffer
        growth_storage = int(base_storage * self.growth_buffer)
        
        # Add replication overhead for writer/reader setups
        replication_storage = int(base_storage * self.replication_overhead)
        
        # Total recommended storage
        total_recommended = base_storage + growth_storage + replication_storage
        
        return {
            'base_storage_gb': base_storage,
            'growth_buffer_gb': growth_storage,
            'replication_overhead_gb': replication_storage,
            'total_recommended_gb': total_recommended,
            'environment_type': environment_type,
            'multiplier_used': base_multiplier
        }
    
    def validate_storage_configuration(self, migration_data_gb: int, 
                                     environment_specs: Dict) -> Dict:
        """Validate storage configuration across migration and environments"""
        
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'total_env_storage': 0,
            'migration_to_env_ratio': 0,
            'environment_analysis': {}
        }
        
        if not environment_specs:
            validation_result['errors'].append("No environment specifications found")
            validation_result['is_valid'] = False
            return validation_result
        
        total_env_storage = 0
        
        for env_name, specs in environment_specs.items():
            env_storage = specs.get('storage_gb', 0)
            env_type = specs.get('environment_type', env_name).lower()
            
            total_env_storage += env_storage
            
            # Calculate recommended storage for this environment
            recommended = self.calculate_recommended_storage(
                migration_data_gb, env_type
            )
            
            # Analysis for this environment
            env_analysis = {
                'current_storage_gb': env_storage,
                'recommended_storage_gb': recommended['total_recommended_gb'],
                'difference_gb': env_storage - recommended['total_recommended_gb'],
                'difference_percent': ((env_storage / recommended['total_recommended_gb']) - 1) * 100 if recommended['total_recommended_gb'] > 0 else 0,
                'status': 'good'
            }
            
            # Determine status and generate warnings
            if env_storage < recommended['base_storage_gb']:
                env_analysis['status'] = 'critical'
                validation_result['errors'].append(
                    f"{env_name}: Storage ({env_storage:,} GB) is below minimum recommended ({recommended['base_storage_gb']:,} GB)"
                )
                validation_result['is_valid'] = False
            elif env_storage < recommended['total_recommended_gb'] * 0.9:
                env_analysis['status'] = 'warning'
                validation_result['warnings'].append(
                    f"{env_name}: Storage ({env_storage:,} GB) is below recommended ({recommended['total_recommended_gb']:,} GB)"
                )
            elif env_storage > recommended['total_recommended_gb'] * 2:
                env_analysis['status'] = 'over_provisioned'
                validation_result['warnings'].append(
                    f"{env_name}: Storage ({env_storage:,} GB) is significantly over-provisioned (recommended: {recommended['total_recommended_gb']:,} GB)"
                )
            
            validation_result['environment_analysis'][env_name] = env_analysis
        
        validation_result['total_env_storage'] = total_env_storage
        
        if migration_data_gb > 0:
            validation_result['migration_to_env_ratio'] = total_env_storage / migration_data_gb
            
            # Overall validation
            if validation_result['migration_to_env_ratio'] < 0.8:
                validation_result['errors'].append(
                    f"Total environment storage ({total_env_storage:,} GB) is significantly less than migration data ({migration_data_gb:,} GB)"
                )
                validation_result['is_valid'] = False
            elif validation_result['migration_to_env_ratio'] > 5:
                validation_result['warnings'].append(
                    f"Total environment storage ({total_env_storage:,} GB) is much larger than migration data ({migration_data_gb:,} GB). Consider optimizing."
                )
        
        # Generate recommendations
        if validation_result['warnings'] or validation_result['errors']:
            validation_result['recommendations'].append("Use the Storage Auto-Calculator to optimize storage allocation")
            validation_result['recommendations'].append("Consider environment-specific data requirements")
            validation_result['recommendations'].append("Plan for 30% growth buffer in production environments")
        
        return validation_result
   

def show_storage_validation_widget():
    """Show storage validation widget in environment setup"""
    
    # Early return if required data is missing
    if not st.session_state.migration_params or not st.session_state.environment_specs:
        return
    
    # Get migration data size
    migration_data_gb = st.session_state.migration_params.get('data_size_gb', 0)
    storage_manager = StorageConfigurationManager()
    
    st.markdown("#### 🔍 Storage Configuration Validation")
    
    # Validate storage configuration
    validation = storage_manager.validate_storage_configuration(
        migration_data_gb, 
        st.session_state.environment_specs
    )
    
    # Status indicator section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_icon = "✅" if validation['is_valid'] else "❌"
        status_text = 'Valid' if validation['is_valid'] else 'Invalid'
        st.metric("Configuration Status", f"{status_icon} {status_text}")
    
    with col2:
        total_storage = validation['total_env_storage']
        st.metric("Total Environment Storage", f"{total_storage:,} GB")
    
    with col3:
        if validation['migration_to_env_ratio'] > 0:
            ratio = validation['migration_to_env_ratio']
            st.metric("Storage Ratio", f"{ratio:.1f}x")
    
    # Display errors if any
    if validation['errors']:
        st.error("🚨 **Critical Issues:**")
        for error in validation['errors']:
            st.error(f"• {error}")
    
    # Display warnings if any
    if validation['warnings']:
        st.warning("⚠️ **Warnings:**")
        for warning in validation['warnings']:
            st.warning(f"• {warning}")
    
    # Display recommendations if any
    if validation['recommendations']:
        st.info("💡 **Recommendations:**")
        for rec in validation['recommendations']:
            st.info(f"• {rec}")
    
    # Environment-specific analysis section
    if validation['environment_analysis']:
        with st.expander("📊 Environment Analysis Details"):
            analysis_data = []
            
            # Status icon mapping
            status_icons = {
                'good': '✅',
                'warning': '⚠️',
                'critical': '🚨',
                'over_provisioned': '💰'
            }
            
            # Process each environment analysis
            for env_name, analysis in validation['environment_analysis'].items():
                status_icon = status_icons.get(analysis['status'], '❓')
                
                analysis_data.append({
                    'Environment': env_name,
                    'Current (GB)': f"{analysis['current_storage_gb']:,}",
                    'Recommended (GB)': f"{analysis['recommended_storage_gb']:,}",
                    'Difference': f"{analysis['difference_gb']:+,} GB ({analysis['difference_percent']:+.1f}%)",
                    'Status': f"{status_icon} {analysis['status'].title()}"
                })
            
            # Create and display dataframe
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
            else:
                st.info("No environment analysis data available.")

class VRopsMetricsAnalyzer:
    def __init__(self):
        # Initialize your analyzer
        pass
    
    def analyze_metrics(self, metrics_data):
        # Your analysis logic here
        analysis_results = {}
        return analysis_results

@dataclass
class InstanceSpecs:
    """Instance specifications with performance metrics"""
    instance_class: str
    vcpu: int
    memory_gb: float
    network_performance: str
    ebs_bandwidth_mbps: int
    hourly_cost: float
    suitable_for: List[str]
    max_connections: int
    iops_capability: int

class OptimizedReaderWriterAnalyzer:
    """Advanced Reader/Writer optimization with intelligent recommendations"""
    
    def __init__(self):
        self.instance_specs = self._initialize_instance_specs()
        self.pricing_data = self._initialize_pricing_data()
        
    def _initialize_instance_specs(self) -> Dict[str, InstanceSpecs]:
        """Initialize comprehensive instance specifications"""
        return {
            # T3 Series - Burstable Performance
            'db.t3.micro': InstanceSpecs('db.t3.micro', 2, 1, 'Low to Moderate', 87, 0.0255, ['development', 'testing'], 87, 1000),
            'db.t3.small': InstanceSpecs('db.t3.small', 2, 2, 'Low to Moderate', 174, 0.051, ['development', 'testing'], 174, 2000),
            'db.t3.medium': InstanceSpecs('db.t3.medium', 2, 4, 'Low to Moderate', 347, 0.102, ['development', 'small_production'], 347, 3000),
            'db.t3.large': InstanceSpecs('db.t3.large', 2, 8, 'Low to Moderate', 695, 0.204, ['development', 'small_production'], 695, 4000),
            'db.t3.xlarge': InstanceSpecs('db.t3.xlarge', 4, 16, 'Low to Moderate', 695, 0.408, ['staging', 'medium_production'], 1000, 5000),
            'db.t3.2xlarge': InstanceSpecs('db.t3.2xlarge', 8, 32, 'Low to Moderate', 695, 0.816, ['staging', 'medium_production'], 1500, 6000),
            
            # R5 Series - Memory Optimized
            'db.r5.large': InstanceSpecs('db.r5.large', 2, 16, 'Up to 10 Gbps', 693, 0.24, ['memory_intensive', 'analytics'], 1000, 7500),
            'db.r5.xlarge': InstanceSpecs('db.r5.xlarge', 4, 32, 'Up to 10 Gbps', 1387, 0.48, ['memory_intensive', 'production'], 2000, 10000),
            'db.r5.2xlarge': InstanceSpecs('db.r5.2xlarge', 8, 64, 'Up to 10 Gbps', 2775, 0.96, ['large_production', 'analytics'], 3000, 15000),
            'db.r5.4xlarge': InstanceSpecs('db.r5.4xlarge', 16, 128, '10 Gbps', 4750, 1.92, ['large_production', 'high_memory'], 5000, 25000),
            'db.r5.8xlarge': InstanceSpecs('db.r5.8xlarge', 32, 256, '10 Gbps', 6800, 3.84, ['enterprise', 'high_performance'], 8000, 40000),
            'db.r5.12xlarge': InstanceSpecs('db.r5.12xlarge', 48, 384, '12 Gbps', 9500, 5.76, ['enterprise', 'very_high_memory'], 12000, 60000),
            'db.r5.16xlarge': InstanceSpecs('db.r5.16xlarge', 64, 512, '20 Gbps', 13600, 7.68, ['enterprise', 'extreme_performance'], 16000, 80000),
            'db.r5.24xlarge': InstanceSpecs('db.r5.24xlarge', 96, 768, '25 Gbps', 19000, 11.52, ['enterprise', 'maximum_performance'], 24000, 120000),
            
            # R6i Series - Latest Generation Memory Optimized
            'db.r6i.large': InstanceSpecs('db.r6i.large', 2, 16, 'Up to 12.5 Gbps', 1000, 0.252, ['memory_intensive', 'latest_gen'], 1200, 8000),
            'db.r6i.xlarge': InstanceSpecs('db.r6i.xlarge', 4, 32, 'Up to 12.5 Gbps', 2000, 0.504, ['production', 'latest_gen'], 2400, 12000),
            'db.r6i.2xlarge': InstanceSpecs('db.r6i.2xlarge', 8, 64, 'Up to 12.5 Gbps', 4000, 1.008, ['large_production', 'latest_gen'], 4000, 18000),
            'db.r6i.4xlarge': InstanceSpecs('db.r6i.4xlarge', 16, 128, '12.5 Gbps', 8000, 2.016, ['enterprise', 'latest_gen'], 6000, 30000),
            
            # C5 Series - Compute Optimized
            'db.c5.large': InstanceSpecs('db.c5.large', 2, 4, 'Up to 10 Gbps', 693, 0.192, ['compute_intensive', 'low_memory'], 1000, 5000),
            'db.c5.xlarge': InstanceSpecs('db.c5.xlarge', 4, 8, 'Up to 10 Gbps', 1387, 0.384, ['compute_intensive', 'oltp'], 2000, 8000),
            'db.c5.2xlarge': InstanceSpecs('db.c5.2xlarge', 8, 16, 'Up to 10 Gbps', 2775, 0.768, ['high_cpu', 'oltp'], 3000, 12000),
            'db.c5.4xlarge': InstanceSpecs('db.c5.4xlarge', 16, 32, '10 Gbps', 4750, 1.536, ['very_high_cpu', 'oltp'], 5000, 20000),
        }
    
    def _initialize_pricing_data(self) -> Dict:
        """Initialize comprehensive pricing data for different regions and deployment options"""
        return {
            'us-east-1': {
                'reserved_1_year': {'discount': 0.35, 'upfront_ratio': 0.0},
                'reserved_3_year': {'discount': 0.55, 'upfront_ratio': 0.0},
                'spot': {'discount': 0.70, 'availability': 0.85},
                'multi_az_multiplier': 2.0,
                'cross_az_transfer_gb': 0.01,
                'backup_storage_gb': 0.095,
                'snapshot_storage_gb': 0.095
            }
        }
    
    def optimize_cluster_configuration(self, environment_specs: Dict) -> Dict:
        """Generate optimized Reader/Writer recommendations with detailed cost analysis"""
        
        optimized_recommendations = {}
        
        for env_name, specs in environment_specs.items():
            optimization = self._optimize_single_environment(env_name, specs)
            optimized_recommendations[env_name] = optimization
        
        return optimized_recommendations
    
    def _optimize_single_environment(self, env_name: str, specs: Dict) -> Dict:
        """Optimize configuration for a single environment"""
        
        # Extract environment characteristics
        cpu_cores = specs.get('cpu_cores', 4)
        ram_gb = specs.get('ram_gb', 16)
        storage_gb = specs.get('storage_gb', 500)
        iops_requirement = specs.get('iops_requirement', 3000)
        peak_connections = specs.get('peak_connections', 100)
        workload_pattern = specs.get('workload_pattern', 'balanced')
        read_write_ratio = specs.get('read_write_ratio', 70)
        environment_type = specs.get('environment_type', 'production')
        daily_usage_hours = specs.get('daily_usage_hours', 24)
        
        # Analyze workload characteristics
        workload_analysis = self._analyze_workload_characteristics(
            cpu_cores, ram_gb, iops_requirement, peak_connections, 
            workload_pattern, read_write_ratio
        )
        
        # Optimize Writer configuration
        writer_optimization = self._optimize_writer_instance(workload_analysis, environment_type)
        
        # Optimize Reader configuration
        reader_optimization = self._optimize_reader_configuration(
            writer_optimization, workload_analysis, environment_type
        )
        
        # Calculate comprehensive costs
        cost_analysis = self._calculate_comprehensive_costs(
            writer_optimization, reader_optimization, storage_gb, 
            iops_requirement, environment_type, daily_usage_hours
        )
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            workload_analysis, writer_optimization, reader_optimization, cost_analysis
        )
        
        return {
            'environment_name': env_name,
            'environment_type': environment_type,
            'workload_analysis': workload_analysis,
            'writer_optimization': writer_optimization,
            'reader_optimization': reader_optimization,
            'cost_analysis': cost_analysis,
            'recommendations': recommendations,
            'optimization_score': self._calculate_optimization_score(
                workload_analysis, writer_optimization, reader_optimization, cost_analysis
            )
        }
    
    def _analyze_workload_characteristics(self, cpu_cores: int, ram_gb: int, 
                                        iops_requirement: int, peak_connections: int,
                                        workload_pattern: str, read_write_ratio: int) -> Dict:
        """Analyze workload characteristics to determine optimal configuration"""
        
        # Calculate workload intensity
        cpu_intensity = min(100, (cpu_cores / 64) * 100)  # Normalize to 64 cores max
        memory_intensity = min(100, (ram_gb / 768) * 100)  # Normalize to 768 GB max
        io_intensity = min(100, (iops_requirement / 80000) * 100)  # Normalize to 80K IOPS max
        connection_intensity = min(100, (peak_connections / 16000) * 100)  # Normalize to 16K connections max
        
        # Determine workload type
        if workload_pattern == 'read_heavy' or read_write_ratio >= 80:
            workload_type = 'read_heavy'
            read_scaling_factor = 2.0
        elif workload_pattern == 'write_heavy' or read_write_ratio <= 30:
            workload_type = 'write_heavy'
            read_scaling_factor = 0.5
        elif workload_pattern == 'analytics':
            workload_type = 'analytics'
            read_scaling_factor = 3.0
        else:
            workload_type = 'balanced'
            read_scaling_factor = 1.2
        
        # Calculate complexity score
        complexity_score = (cpu_intensity + memory_intensity + io_intensity + connection_intensity) / 4
        
        return {
            'cpu_intensity': cpu_intensity,
            'memory_intensity': memory_intensity,
            'io_intensity': io_intensity,
            'connection_intensity': connection_intensity,
            'complexity_score': complexity_score,
            'workload_type': workload_type,
            'read_scaling_factor': read_scaling_factor,
            'recommended_reader_count': self._calculate_optimal_reader_count(
                workload_type, complexity_score, peak_connections
            ),
            'performance_requirements': self._determine_performance_requirements(
                cpu_intensity, memory_intensity, io_intensity
            )
        }
    
    def _calculate_optimal_reader_count(self, workload_type: str, complexity_score: float, peak_connections: int) -> int:
        """Calculate optimal number of read replicas"""
        
        base_readers = {
            'read_heavy': 3,
            'analytics': 2,
            'balanced': 1,
            'write_heavy': 0
        }
        
        # Adjust based on complexity and connections
        complexity_adjustment = int(complexity_score / 30)  # Add reader every 30 points of complexity
        connection_adjustment = int(peak_connections / 2000)  # Add reader every 2000 connections
        
        optimal_count = base_readers.get(workload_type, 1) + complexity_adjustment + connection_adjustment
        
        return min(5, max(0, optimal_count))  # Cap at 5 readers
    
    def _determine_performance_requirements(self, cpu_intensity: float, memory_intensity: float, io_intensity: float) -> str:
        """Determine performance tier requirements"""
        
        avg_intensity = (cpu_intensity + memory_intensity + io_intensity) / 3
        
        if avg_intensity >= 80:
            return 'high_performance'
        elif avg_intensity >= 60:
            return 'medium_performance'
        elif avg_intensity >= 40:
            return 'standard_performance'
        else:
            return 'basic_performance'
    
    def _optimize_writer_instance(self, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Writer instance selection"""
        
        performance_req = workload_analysis['performance_requirements']
        complexity_score = workload_analysis['complexity_score']
        
        # Filter suitable instances based on performance requirements
        suitable_instances = []
        
        for instance_class, specs in self.instance_specs.items():
            if self._is_instance_suitable_for_writer(specs, performance_req, environment_type, complexity_score):
                suitable_instances.append((instance_class, specs))
        
        # Score instances based on multiple criteria
        scored_instances = []
        for instance_class, specs in suitable_instances:
            score = self._score_writer_instance(specs, workload_analysis, environment_type)
            scored_instances.append((score, instance_class, specs))
        
        # Sort by score (higher is better)
        scored_instances.sort(reverse=True)
        
        # Get top recommendations
        primary_recommendation = scored_instances[0] if scored_instances else None
        alternatives = scored_instances[1:3] if len(scored_instances) > 1 else []
        
        if primary_recommendation:
            score, instance_class, specs = primary_recommendation
            
            return {
                'instance_class': instance_class,
                'specs': specs,
                'score': score,
                'multi_az': environment_type in ['production', 'staging'],
                'reasoning': self._generate_writer_reasoning(specs, workload_analysis, environment_type),
                'alternatives': [
                    {'instance_class': alt[1], 'score': alt[0], 'specs': alt[2]}
                    for alt in alternatives
                ],
                'monthly_cost': self._calculate_instance_monthly_cost(specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(specs, environment_type) * 12
            }
        else:
            # Fallback if no suitable instances found
            fallback_instance = 'db.r5.large'
            fallback_specs = self.instance_specs[fallback_instance]
            
            return {
                'instance_class': fallback_instance,
                'specs': fallback_specs,
                'score': 50.0,
                'multi_az': True,
                'reasoning': 'Fallback recommendation due to no optimal matches',
                'alternatives': [],
                'monthly_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type) * 12
            }
    
    def _is_instance_suitable_for_writer(self, specs: InstanceSpecs, performance_req: str, 
                                       environment_type: str, complexity_score: float) -> bool:
        """Check if instance is suitable for writer role"""
        
        # Performance tier requirements
        performance_minimums = {
            'high_performance': {'min_vcpu': 16, 'min_memory': 64, 'min_iops': 20000},
            'medium_performance': {'min_vcpu': 8, 'min_memory': 32, 'min_iops': 10000},
            'standard_performance': {'min_vcpu': 4, 'min_memory': 16, 'min_iops': 5000},
            'basic_performance': {'min_vcpu': 2, 'min_memory': 8, 'min_iops': 2000}
        }
        
        minimums = performance_minimums.get(performance_req, performance_minimums['standard_performance'])
        
        # Check basic requirements
        if (specs.vcpu < minimums['min_vcpu'] or 
            specs.memory_gb < minimums['min_memory'] or 
            specs.iops_capability < minimums['min_iops']):
            return False
        
        # Environment-specific filters
        if environment_type == 'production':
            # Production requires more robust instances
            if specs.vcpu < 4 or specs.memory_gb < 16:
                return False
            # Avoid burstable instances for production unless complexity is low
            if 't3' in specs.instance_class and complexity_score > 50:
                return False
        
        return True
    
    def _score_writer_instance(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> float:
        """Score writer instance based on multiple criteria"""
        
        score = 0.0
        
        # Performance scoring (40% weight)
        performance_score = min(100, (specs.vcpu * 10 + specs.memory_gb / 4 + specs.iops_capability / 500))
        score += performance_score * 0.4
        
        # Cost efficiency scoring (30% weight)
        cost_per_vcpu = specs.hourly_cost / specs.vcpu
        cost_per_gb_memory = specs.hourly_cost / specs.memory_gb
        cost_efficiency = max(0, 100 - (cost_per_vcpu * 100 + cost_per_gb_memory * 10))
        score += cost_efficiency * 0.3
        
        # Suitability scoring (20% weight)
        suitability_score = 0
        if environment_type in specs.suitable_for:
            suitability_score = 100
        elif any(env in specs.suitable_for for env in ['production', 'large_production', 'enterprise']):
            suitability_score = 80
        else:
            suitability_score = 60
        score += suitability_score * 0.2
        
        # Network performance scoring (10% weight)
        network_score = 60  # Base score
        if '25 Gbps' in specs.network_performance:
            network_score = 100
        elif '20 Gbps' in specs.network_performance:
            network_score = 90
        elif '12' in specs.network_performance:
            network_score = 80
        elif '10 Gbps' in specs.network_performance:
            network_score = 70
        score += network_score * 0.1
        
        return min(100, score)
    
    def _optimize_reader_configuration(self, writer_optimization: Dict, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Reader configuration based on writer and workload"""
    
    recommended_count = workload_analysis['recommended_reader_count']
    
    if recommended_count == 0:
        return {
            'count': 0,
            'instance_class': None,
            'specs': None,
            'single_reader_monthly_cost': 0,
            'total_monthly_cost': 0,
            'total_annual_cost': 0,
            'multi_az': False,
            'reasoning': 'No read replicas needed for this workload pattern',
            'scaling_recommendations': []
        }
    
    # Determine optimal reader instance size
    writer_specs = writer_optimization['specs']
    reader_instance_class = self._calculate_optimal_reader_size(writer_specs, workload_analysis, environment_type)
    reader_specs = self.instance_specs[reader_instance_class]
    
    # Calculate costs
    single_reader_monthly_cost = self._calculate_instance_monthly_cost(reader_specs, environment_type, is_reader=True)
    total_monthly_cost = single_reader_monthly_cost * recommended_count
    total_annual_cost = total_monthly_cost * 12
    
    return {
        'count': recommended_count,
        'instance_class': reader_instance_class,
        'specs': reader_specs,
        'single_reader_monthly_cost': single_reader_monthly_cost,
        'total_monthly_cost': total_monthly_cost,
        'total_annual_cost': total_annual_cost,
        'multi_az': environment_type == 'production',
        'reasoning': self._generate_reader_reasoning(recommended_count, reader_specs, workload_analysis),
        'scaling_recommendations': self._generate_reader_scaling_recommendations(workload_analysis)
    }

def _calculate_optimal_reader_size(self, writer_specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
    """Calculate optimal reader instance size"""
    
    workload_type = workload_analysis['workload_type']
    read_scaling_factor = workload_analysis['read_scaling_factor']
    
    # Base strategy: readers are typically same size or smaller than writer
    if workload_type == 'read_heavy' or workload_type == 'analytics':
        # For read-heavy workloads, readers should match or exceed writer capacity
        target_vcpu = int(writer_specs.vcpu * read_scaling_factor)
        target_memory = writer_specs.memory_gb * read_scaling_factor
    else:
        # For balanced/write-heavy workloads, readers can be smaller
        target_vcpu = max(2, int(writer_specs.vcpu * 0.7))
        target_memory = writer_specs.memory_gb * 0.7
    
    # Find best matching instance
    best_match = None
    best_score = 0
    
    for instance_class, specs in self.instance_specs.items():
        if specs.vcpu >= target_vcpu * 0.8 and specs.memory_gb >= target_memory * 0.8:
            # Score based on how close it matches our target and cost efficiency
            size_match_score = 100 - abs(specs.vcpu - target_vcpu) * 5 - abs(specs.memory_gb - target_memory) * 2
            cost_efficiency_score = 100 - (specs.hourly_cost * 10)  # Lower cost is better
            overall_score = size_match_score * 0.7 + cost_efficiency_score * 0.3
            
            if overall_score > best_score:
                best_score = overall_score
                best_match = instance_class
    
    return best_match or 'db.r5.large'  # Fallback

def _calculate_instance_monthly_cost(self, specs: InstanceSpecs, environment_type: str, is_reader: bool = False) -> float:
    """Calculate monthly cost for an instance"""
    
    base_hourly_cost = specs.hourly_cost
    
    # Apply Multi-AZ multiplier if needed
    if environment_type in ['production', 'staging'] and not is_reader:
        base_hourly_cost *= self.pricing_data['us-east-1']['multi_az_multiplier']
    
    # Calculate monthly cost (730 hours per month)
    monthly_cost = base_hourly_cost * 730
    
    return monthly_cost

def _calculate_comprehensive_costs(self, writer_optimization: Dict, reader_optimization: Dict,
                                 storage_gb: int, iops_requirement: int, environment_type: str,
                                 daily_usage_hours: int) -> Dict:
    """Calculate comprehensive cost analysis"""
    
    # Instance costs
    writer_monthly_cost = writer_optimization['monthly_cost']
    reader_monthly_cost = reader_optimization['total_monthly_cost']
    total_instance_monthly_cost = writer_monthly_cost + reader_monthly_cost
    
    # Storage costs
    storage_costs = self._calculate_storage_costs(storage_gb, iops_requirement, environment_type)
    
    # Additional costs
    backup_monthly_cost = storage_gb * 0.095  # $0.095 per GB per month
    monitoring_monthly_cost = 50 if environment_type == 'production' else 20  # CloudWatch Enhanced Monitoring
    
    # Cross-AZ transfer costs (estimated)
    cross_az_monthly_cost = 0
    if reader_optimization['count'] > 0:
        estimated_cross_az_gb = storage_gb * 0.1  # Estimate 10% of storage as cross-AZ traffic
        cross_az_monthly_cost = estimated_cross_az_gb * 0.01
    
    total_monthly_cost = (total_instance_monthly_cost + storage_costs['total_monthly'] + 
                        backup_monthly_cost + monitoring_monthly_cost + cross_az_monthly_cost)
    
    # Reserved Instance calculations
    reserved_1_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 1)
    reserved_3_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 3)
    
    return {
        'monthly_breakdown': {
            'writer_instance': writer_monthly_cost,
            'reader_instances': reader_monthly_cost,
            'storage': storage_costs['total_monthly'],
            'backup': backup_monthly_cost,
            'monitoring': monitoring_monthly_cost,
            'cross_az_transfer': cross_az_monthly_cost,
            'total': total_monthly_cost
        },
        'annual_breakdown': {
            'writer_instance': writer_monthly_cost * 12,
            'reader_instances': reader_monthly_cost * 12,
            'storage': storage_costs['total_monthly'] * 12,
            'backup': backup_monthly_cost * 12,
            'monitoring': monitoring_monthly_cost * 12,
            'cross_az_transfer': cross_az_monthly_cost * 12,
            'total': total_monthly_cost * 12
        },
        'storage_details': storage_costs,
        'reserved_instance_options': {
            '1_year': reserved_1_year,
            '3_year': reserved_3_year
        },
        'cost_optimization_opportunities': self._identify_cost_optimization_opportunities(
            writer_optimization, reader_optimization, storage_costs, environment_type
        )
    }

def _calculate_storage_costs(self, storage_gb: int, iops_requirement: int, environment_type: str) -> Dict:
    """Calculate detailed storage costs"""
    
    # Determine optimal storage type
    if iops_requirement > 16000:
        storage_type = 'io2'
        base_cost_per_gb = 0.125
        iops_cost_per_iops = 0.065
        additional_iops = iops_requirement
    elif iops_requirement > 3000:
        storage_type = 'gp3'
        base_cost_per_gb = 0.08
        # GP3 includes 3000 IOPS free, charge for additional
        additional_iops = max(0, iops_requirement - 3000)
        iops_cost_per_iops = 0.005 if additional_iops > 0 else 0
    else:
        storage_type = 'gp3'
        base_cost_per_gb = 0.08
        iops_cost_per_iops = 0
        additional_iops = 0
    
    base_storage_cost = storage_gb * base_cost_per_gb
    iops_cost = additional_iops * iops_cost_per_iops
    total_monthly_cost = base_storage_cost + iops_cost
    
    return {
        'storage_type': storage_type,
        'storage_gb': storage_gb,
        'iops_provisioned': iops_requirement,
        'base_storage_cost': base_storage_cost,
        'iops_cost': iops_cost,
        'total_monthly': total_monthly_cost,
        'cost_per_gb': base_cost_per_gb,
        'cost_per_iops': iops_cost_per_iops
    }

def _calculate_reserved_instance_savings(self, monthly_instance_cost: float, years: int) -> Dict:
    """Calculate Reserved Instance savings"""
    
    if years == 1:
        discount = self.pricing_data['us-east-1']['reserved_1_year']['discount']
    else:
        discount = self.pricing_data['us-east-1']['reserved_3_year']['discount']
    
    annual_on_demand = monthly_instance_cost * 12
    annual_reserved = annual_on_demand * (1 - discount)
    total_on_demand = annual_on_demand * years
    total_reserved = annual_reserved * years
    total_savings = total_on_demand - total_reserved
    
    return {
        'term_years': years,
        'discount_percentage': discount * 100,
        'annual_cost': annual_reserved,
        'total_cost': total_reserved,
        'total_savings': total_savings,
        'monthly_cost': annual_reserved / 12
    }

def _identify_cost_optimization_opportunities(self, writer_optimization: Dict, 
                                            reader_optimization: Dict, storage_costs: Dict,
                                            environment_type: str) -> List[Dict]:
    """Identify cost optimization opportunities"""
    
    opportunities = []
    
    # Reserved Instance opportunity
    if writer_optimization['monthly_cost'] > 200:  # Threshold for RI consideration
        opportunities.append({
            'type': 'Reserved Instances',
            'description': 'Consider 1-year or 3-year Reserved Instances for predictable workloads',
            'potential_savings': writer_optimization['monthly_cost'] * 12 * 0.35,
            'impact': 'High',
            'implementation_effort': 'Low'
        })
    
    # Reader optimization
    if reader_optimization['count'] > 2:
        opportunities.append({
            'type': 'Reader Instance Optimization',
            'description': 'Consider using Aurora Auto Scaling for variable read workloads',
            'potential_savings': reader_optimization['total_monthly_cost'] * 0.2,
            'impact': 'Medium',
            'implementation_effort': 'Medium'
        })
    
    # Storage optimization
    if storage_costs['storage_type'] == 'io2' and storage_costs['iops_provisioned'] < 10000:
        opportunities.append({
            'type': 'Storage Type Optimization',
            'description': 'Consider gp3 storage for lower IOPS requirements',
            'potential_savings': storage_costs['total_monthly'] * 0.3,
            'impact': 'Medium',
            'implementation_effort': 'Low'
        })
    
    # Environment-specific optimizations
    if environment_type in ['development', 'testing']:
        opportunities.append({
            'type': 'Development Environment Optimization',
            'description': 'Consider scheduled start/stop or spot instances for non-production',
            'potential_savings': (writer_optimization['monthly_cost'] + reader_optimization['total_monthly_cost']) * 0.5,
            'impact': 'High',
            'implementation_effort': 'Medium'
        })
    
    # Default strategy if none generated
    if not opportunities:
        opportunities.append({
            'type': 'General Migration Management',
            'strategy': 'Follow AWS migration best practices and maintain regular checkpoints',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    return opportunities

def _generate_optimization_recommendations(self, workload_analysis: Dict, writer_optimization: Dict,
                                         reader_optimization: Dict, cost_analysis: Dict) -> List[str]:
    """Generate comprehensive optimization recommendations"""
    
    recommendations = []
    
    # Performance recommendations
    if workload_analysis['complexity_score'] > 80:
        recommendations.append("Consider upgrading to latest generation instances (R6i) for better price/performance")
    
    # Cost recommendations
    total_annual_cost = cost_analysis['annual_breakdown']['total']
    if total_annual_cost > 50000:
        recommendations.append("Evaluate Reserved Instances for significant cost savings on predictable workloads")
    
    # Scaling recommendations
    if reader_optimization['count'] > 1:
        recommendations.append("Implement Aurora Auto Scaling to optimize reader count based on actual demand")
    
    # Monitoring recommendations
    recommendations.append("Set up Enhanced Monitoring and Performance Insights for optimization opportunities")
    recommendations.append("Configure CloudWatch alarms for CPU, memory, and connection metrics")
    
    return recommendations

def _calculate_optimization_score(self, workload_analysis: Dict, writer_optimization: Dict,
                                reader_optimization: Dict, cost_analysis: Dict) -> float:
    """Calculate overall optimization score"""
    
    # Performance score (0-100)
    performance_score = min(100, writer_optimization['score'])
    
    # Cost efficiency score (0-100)
    cost_per_vcpu = cost_analysis['monthly_breakdown']['total'] / writer_optimization['specs'].vcpu
    cost_efficiency_score = max(0, 100 - (cost_per_vcpu / 10))  # Lower cost per vCPU = higher score
    
    # Configuration appropriateness (0-100)
    config_score = 80  # Base score
    if reader_optimization['count'] > 0 and workload_analysis['workload_type'] in ['read_heavy', 'analytics']:
        config_score += 15
    elif reader_optimization['count'] == 0 and workload_analysis['workload_type'] == 'write_heavy':
        config_score += 15
    
    # Overall score (weighted average)
    overall_score = (performance_score * 0.4 + cost_efficiency_score * 0.4 + config_score * 0.2)
    
    return min(100, overall_score)

def _generate_writer_reasoning(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
    """Generate reasoning for writer instance recommendation"""
    
    reasoning_parts = []
    
    # Performance reasoning
    if workload_analysis['complexity_score'] > 70:
        reasoning_parts.append(f"High complexity workload requires {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
    else:
        reasoning_parts.append(f"Balanced configuration with {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
    
    # Environment reasoning
    if environment_type == 'production':
        reasoning_parts.append("Production-grade instance with high availability features")
    else:
        reasoning_parts.append(f"Cost-optimized for {environment_type} environment")
    
    # Network reasoning
    if '25 Gbps' in specs.network_performance:
        reasoning_parts.append("Enhanced networking for high-throughput workloads")
    
    return ". ".join(reasoning_parts) + "."

def _generate_reader_reasoning(self, count: int, specs: InstanceSpecs, workload_analysis: Dict) -> str:
    """Generate reasoning for reader configuration"""
    
    if count == 0:
        return "No read replicas needed for write-heavy or low-complexity workloads"
    
    workload_type = workload_analysis['workload_type']
    
    reasoning_parts = [
        f"{count} read replica{'s' if count > 1 else ''} recommended for {workload_type} workload",
        f"Each reader: {specs.vcpu} vCPUs, {specs.memory_gb}GB memory"
    ]
    
    if workload_type == 'read_heavy':
        reasoning_parts.append("Multiple readers will distribute read load effectively")
    elif workload_type == 'analytics':
        reasoning_parts.append("Dedicated readers for analytical queries to avoid impact on primary")
    
    return ". ".join(reasoning_parts) + "."

def _generate_reader_scaling_recommendations(self, workload_analysis: Dict) -> List[str]:
    """Generate reader scaling recommendations"""
    
    recommendations = []
    
    if workload_analysis['workload_type'] == 'read_heavy':
        recommendations.append("Consider Aurora Auto Scaling to automatically adjust reader count based on CPU/connection metrics")
        recommendations.append("Monitor read replica lag and add readers if lag consistently exceeds 1 second")
    
    if workload_analysis['complexity_score'] > 60:
        recommendations.append("For high-complexity workloads, consider reader instances same size as writer")
    
    recommendations.append("Use connection pooling (like PgBouncer) to optimize connection distribution across readers")
    
    return recommendations

def _generate_optimization_recommendations(self, workload_analysis: Dict, writer_optimization: Dict,
                                         reader_optimization: Dict, cost_analysis: Dict) -> List[str]:
    """Generate comprehensive optimization recommendations"""
    
    recommendations = []
    
    # Performance recommendations
    if workload_analysis['complexity_score'] > 80:
        recommendations.append("Consider upgrading to latest generation instances (R6i) for better price/performance")
    
    # Cost recommendations
    total_annual_cost = cost_analysis['annual_breakdown']['total']
    if total_annual_cost > 50000:
        recommendations.append("Evaluate Reserved Instances for significant cost savings on predictable workloads")
    
    # Scaling recommendations
    if reader_optimization['count'] > 1:
        recommendations.append("Implement Aurora Auto Scaling to optimize reader count based on actual demand")
    
    # Monitoring recommendations
    recommendations.append("Set up Enhanced Monitoring and Performance Insights for optimization opportunities")
    recommendations.append("Configure CloudWatch alarms for CPU, memory, and connection metrics")
    
    return recommendations

def _calculate_optimization_score(self, workload_analysis: Dict, writer_optimization: Dict,
                                reader_optimization: Dict, cost_analysis: Dict) -> float:
    """Calculate overall optimization score"""
    
    # Performance score (0-100)
    performance_score = min(100, writer_optimization['score'])
    
    # Cost efficiency score (0-100)
    cost_per_vcpu = cost_analysis['monthly_breakdown']['total'] / writer_optimization['specs'].vcpu
    cost_efficiency_score = max(0, 100 - (cost_per_vcpu / 10))  # Lower cost per vCPU = higher score
    
    # Configuration appropriateness (0-100)
    config_score = 80  # Base score
    if reader_optimization['count'] > 0 and workload_analysis['workload_type'] in ['read_heavy', 'analytics']:
        config_score += 15
    elif reader_optimization['count'] == 0 and workload_analysis['workload_type'] == 'write_heavy':
        config_score += 15
    
    # Overall score (weighted average)
    overall_score = (performance_score * 0.4 + cost_efficiency_score * 0.4 + config_score * 0.2)
    
    return min(100, overall_score)

def _generate_writer_reasoning(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
    """Generate reasoning for writer instance recommendation"""
    
    reasoning_parts = []
    
    # Performance reasoning
    if workload_analysis['complexity_score'] > 70:
        reasoning_parts.append(f"High complexity workload requires {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
    else:
        reasoning_parts.append(f"Balanced configuration with {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
    
    # Environment reasoning
    if environment_type == 'production':
        reasoning_parts.append("Production-grade instance with high availability features")
    else:
        reasoning_parts.append(f"Cost-optimized for {environment_type} environment")
    
    # Network reasoning
    if '25 Gbps' in specs.network_performance:
        reasoning_parts.append("Enhanced networking for high-throughput workloads")
    
    return ". ".join(reasoning_parts) + "."

def _generate_reader_reasoning(self, count: int, specs: InstanceSpecs, workload_analysis: Dict) -> str:
    """Generate reasoning for reader configuration"""
    
    if count == 0:
        return "No read replicas needed for write-heavy or low-complexity workloads"
    
    workload_type = workload_analysis['workload_type']
    
    reasoning_parts = [
        f"{count} read replica{'s' if count > 1 else ''} recommended for {workload_type} workload",
        f"Each reader: {specs.vcpu} vCPUs, {specs.memory_gb}GB memory"
    ]
    
    if workload_type == 'read_heavy':
        reasoning_parts.append("Multiple readers will distribute read load effectively")
    elif workload_type == 'analytics':
        reasoning_parts.append("Dedicated readers for analytical queries to avoid impact on primary")
    
    return ". ".join(reasoning_parts) + "."

# End of OptimizedReaderWriterAnalyzer class methods

def display_optimization_results(optimization_results: Dict):
    """Display comprehensive optimization results in Streamlit"""
    
    st.markdown("# 🚀 Optimized Reader/Writer Recommendations")
    
    # Overall summary
    total_monthly_cost = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                             for env in optimization_results.values()])
    total_annual_cost = total_monthly_cost * 12
    avg_optimization_score = sum([env['optimization_score'] 
                                 for env in optimization_results.values()]) / len(optimization_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Monthly Cost", f"${total_monthly_cost:,.0f}")
    
    with col2:
        st.metric("Total Annual Cost", f"${total_annual_cost:,.0f}")
    
    with col3:
        st.metric("Avg Optimization Score", f"{avg_optimization_score:.1f}/100")
    
    with col4:
        total_instances = sum([1 + env['reader_optimization']['count'] 
                              for env in optimization_results.values()])
        st.metric("Total Instances", total_instances)
    
    # Detailed results for each environment
    for env_name, optimization in optimization_results.items():
        with st.expander(f"🏢 {env_name.title()} - Optimization Score: {optimization['optimization_score']:.1f}/100", expanded=True):
            
            # Configuration overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✍️ Writer Configuration")
                writer = optimization['writer_optimization']
                st.info(f"""
                **Instance:** {writer['instance_class']}  
                **vCPUs:** {writer['specs'].vcpu}  
                **Memory:** {writer['specs'].memory_gb} GB  
                **Network:** {writer['specs'].network_performance}  
                **Multi-AZ:** {'✅ Yes' if writer['multi_az'] else '❌ No'}  
                **Monthly Cost:** ${writer['monthly_cost']:,.0f}  
                **Annual Cost:** ${writer['annual_cost']:,.0f}
                """)
                
                st.markdown("**Reasoning:**")
                st.write(writer['reasoning'])
            
            with col2:
                st.markdown("#### 📖 Reader Configuration")
                reader = optimization['reader_optimization']
                
                if reader['count'] > 0:
                    st.success(f"""
                    **Count:** {reader['count']} replicas  
                    **Instance:** {reader['instance_class']}  
                    **vCPUs:** {reader['specs'].vcpu} each  
                    **Memory:** {reader['specs'].memory_gb} GB each  
                    **Multi-AZ:** {'✅ Yes' if reader['multi_az'] else '❌ No'}  
                    **Cost per Reader:** ${reader['single_reader_monthly_cost']:,.0f}/month  
                    **Total Monthly Cost:** ${reader['total_monthly_cost']:,.0f}  
                    **Total Annual Cost:** ${reader['total_annual_cost']:,.0f}
                    """)
                else:
                    st.warning("**No read replicas recommended**")
                
                st.markdown("**Reasoning:**")
                st.write(reader['reasoning'])
            
            # Cost breakdown chart
            st.markdown("#### 💰 Cost Breakdown")
            
            cost_data = optimization['cost_analysis']['monthly_breakdown']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Writer Instance', 'Reader Instances', 'Storage', 'Backup', 'Monitoring', 'Cross-AZ Transfer'],
                values=[cost_data['writer_instance'], cost_data['reader_instances'], 
                       cost_data['storage'], cost_data['backup'], cost_data['monitoring'], cost_data['cross_az_transfer']],
                hole=0.4,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
            )])
            
            fig.update_layout(
                title=f"{env_name} Monthly Cost Distribution - Total: ${cost_data['total']:,.0f}",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"optimization_cost_chart_{env_name}")
            
            # Reserved Instance savings
            st.markdown("#### 💳 Reserved Instance Savings Opportunities")
            
            ri_1_year = optimization['cost_analysis']['reserved_instance_options']['1_year']
            ri_3_year = optimization['cost_analysis']['reserved_instance_options']['3_year']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1-Year Reserved Instances**")
                st.metric("Annual Savings", f"${ri_1_year['total_savings']:,.0f}", 
                         delta=f"{ri_1_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_1_year['monthly_cost']:,.0f}")
            
            with col2:
                st.markdown("**3-Year Reserved Instances**")
                st.metric("Total 3-Year Savings", f"${ri_3_year['total_savings']:,.0f}", 
                         delta=f"{ri_3_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_3_year['monthly_cost']:,.0f}")
            
            # Optimization opportunities
            st.markdown("#### 🎯 Cost Optimization Opportunities")
            
            opportunities = optimization['cost_analysis']['cost_optimization_opportunities']
            
            for opp in opportunities:
                impact_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[opp['impact']]
                st.markdown(f"""
                {impact_color} **{opp['type']}** - {opp['impact']} Impact  
                {opp['description']}  
                💰 Potential savings: ${opp['potential_savings']:,.0f}/year  
                🔧 Implementation effort: {opp['implementation_effort']}
                """)
            
            # Workload analysis
            st.markdown("#### 📊 Workload Analysis")
            
            workload = optimization['workload_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Intensity Metrics:**")
                st.progress(workload['cpu_intensity']/100, text=f"CPU Intensity: {workload['cpu_intensity']:.1f}%")
                st.progress(workload['memory_intensity']/100, text=f"Memory Intensity: {workload['memory_intensity']:.1f}%")
                st.progress(workload['io_intensity']/100, text=f"I/O Intensity: {workload['io_intensity']:.1f}%")
                st.progress(workload['connection_intensity']/100, text=f"Connection Intensity: {workload['connection_intensity']:.1f}%")
            
            with col2:
                st.markdown("**Workload Characteristics:**")
                st.write(f"**Type:** {workload['workload_type'].replace('_', ' ').title()}")
                st.write(f"**Complexity Score:** {workload['complexity_score']:.1f}/100")
                st.write(f"**Performance Tier:** {workload['performance_requirements'].replace('_', ' ').title()}")
                st.write(f"**Read Scaling Factor:** {workload['read_scaling_factor']:.1f}x")
            
            # Recommendations
            st.markdown("#### 💡 Optimization Recommendations")
            
            for rec in optimization['recommendations']:
                st.markdown(f"• {rec}")

# Helper functions for export and reporting
def export_recommendations_to_csv(optimization_results):
    """Export optimization results to CSV format"""
    
    export_data = []
    
    for env_name, optimization in optimization_results.items():
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        costs = optimization['cost_analysis']['monthly_breakdown']
        
        export_data.append({
            'Environment': env_name,
            'Environment_Type': optimization['environment_type'],
            'Optimization_Score': f"{optimization['optimization_score']:.1f}",
            'Writer_Instance': writer['instance_class'],
            'Writer_vCPUs': writer['specs'].vcpu,
            'Writer_Memory_GB': writer['specs'].memory_gb,
            'Writer_Monthly_Cost': f"${writer['monthly_cost']:,.0f}",
            'Writer_Annual_Cost': f"${writer['annual_cost']:,.0f}",
            'Reader_Count': reader['count'],
            'Reader_Instance': reader['instance_class'] if reader['count'] > 0 else 'None',
            'Reader_Monthly_Cost_Total': f"${reader['total_monthly_cost']:,.0f}",
            'Reader_Annual_Cost_Total': f"${reader['total_annual_cost']:,.0f}",
            'Storage_Monthly_Cost': f"${costs['storage']:,.0f}",
            'Total_Monthly_Cost': f"${costs['total']:,.0f}",
            'Total_Annual_Cost': f"${costs['total'] * 12:,.0f}",
            'Reserved_1Y_Savings': f"${optimization['cost_analysis']['reserved_instance_options']['1_year']['total_savings']:,.0f}",
            'Reserved_3Y_Savings': f"${optimization['cost_analysis']['reserved_instance_options']['3_year']['total_savings']:,.0f}",
            'Workload_Type': optimization['workload_analysis']['workload_type'],
            'Complexity_Score': f"{optimization['workload_analysis']['complexity_score']:.1f}",
            'Writer_Reasoning': writer['reasoning'],
            'Reader_Reasoning': reader['reasoning']
        })
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False)

def generate_optimization_report_pdf(optimization_results):
    """Generate detailed PDF report for optimization results"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Reader/Writer Optimization Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    total_monthly = sum([opt['cost_analysis']['monthly_breakdown']['total'] for opt in optimization_results.values()])
    avg_score = sum([opt['optimization_score'] for opt in optimization_results.values()]) / len(optimization_results)
    
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(f"Total Environments Analyzed: {len(optimization_results)}", styles['Normal']))
    story.append(Paragraph(f"Total Monthly Cost: ${total_monthly:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Total Annual Cost: ${total_monthly * 12:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Average Optimization Score: {avg_score:.1f}/100", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Detailed recommendations for each environment
    for env_name, optimization in optimization_results.items():
        story.append(Paragraph(f"{env_name} Environment", styles['Heading2']))
        
        # Create summary table
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        
        env_data = [
            ['Component', 'Configuration', 'Monthly Cost', 'Annual Cost'],
            ['Writer Instance', 
             f"{writer['instance_class']} ({writer['specs'].vcpu} vCPU, {writer['specs'].memory_gb}GB)",
             f"${writer['monthly_cost']:,.0f}",
             f"${writer['annual_cost']:,.0f}"],
            ['Read Replicas',
             f"{reader['count']} x {reader['instance_class']}" if reader['count'] > 0 else "None",
             f"${reader['total_monthly_cost']:,.0f}",
             f"${reader['total_annual_cost']:,.0f}"],
            ['Total Environment',
             f"Score: {optimization['optimization_score']:.1f}/100",
             f"${optimization['cost_analysis']['monthly_breakdown']['total']:,.0f}",
             f"${optimization['cost_analysis']['monthly_breakdown']['total'] * 12:,.0f}"]
        ]
        
        env_table = Table(env_data, colWidths=[1.5*inch, 2.5*inch, 1.2*inch, 1.2*inch])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        
        story.append(env_table)
        story.append(Spacer(1, 15))
        
        # Add reasoning
        story.append(Paragraph("Writer Reasoning:", styles['Heading3']))
        story.append(Paragraph(writer['reasoning'], styles['Normal']))
        story.append(Paragraph("Reader Reasoning:", styles['Heading3']))
        story.append(Paragraph(reader['reasoning'], styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Integration function for main app
def show_optimized_recommendations():
    """Show optimized Reader/Writer recommendations"""
    
    st.markdown("## 🧠 AI-Optimized Reader/Writer Recommendations")
    
    if not st.session_state.environment_specs:
        st.warning("⚠️ Please configure environments first.")
        st.info("👆 Go to 'Environment Setup' section to configure your database environments")
        return
    
    # Show current environment count
    env_count = len(st.session_state.environment_specs)
    is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Environments Configured", env_count)
    with col2:
        config_type = "Enhanced Cluster Config" if is_enhanced else "Basic Config"
        st.metric("Configuration Type", config_type)
    
    if not is_enhanced:
        st.info("💡 For best results, use the Enhanced Environment Setup with cluster configuration")
    
    # Configuration options
    st.markdown("### ⚙️ Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Cost Efficiency", "Performance", "Balanced"],
            index=2,
            help="Choose whether to prioritize cost savings, performance, or a balance of both"
        )
    
    with col2:
        consider_reserved_instances = st.checkbox(
            "Include Reserved Instance Analysis",
            value=True,
            help="Calculate potential savings with 1-year and 3-year Reserved Instances"
        )
    
    with col3:
        environment_priority = st.selectbox(
            "Environment Priority",
            ["Production First", "All Equal", "Cost First"],
            index=0,
            help="Prioritization strategy for resource allocation"
        )
    
    # Run optimization button
    if st.button("🧠 Generate AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing workloads and optimizing configurations..."):
            
            try:
                # Initialize optimizer
                optimizer = OptimizedReaderWriterAnalyzer()
                
                # Apply optimization settings
                optimizer.set_optimization_preferences(
                    goal=optimization_goal.lower().replace(" ", "_"),
                    consider_reserved=consider_reserved_instances,
                    environment_priority=environment_priority.lower().replace(" ", "_")
                )
                
                # Run optimization
                optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
                
                # Store results
                st.session_state.optimization_results = optimization_results
                
                st.success("✅ Optimization complete!")
                
                # Show quick summary
                total_monthly = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                                   for env in optimization_results.values()])
                avg_score = sum([env['optimization_score'] 
                               for env in optimization_results.values()]) / len(optimization_results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Monthly Cost", f"${total_monthly:,.0f}")
                with col2:
                    st.metric("Average Optimization Score", f"{avg_score:.1f}/100")
                with col3:
                    potential_ri_savings = sum([
                        env['cost_analysis']['reserved_instance_options']['3_year']['total_savings']
                        for env in optimization_results.values()
                    ])
                    st.metric("Potential 3-Year RI Savings", f"${potential_ri_savings:,.0f}")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.code(str(e))
    
    # Display results if available
    if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
        st.markdown("---")
        display_optimization_results(st.session_state.optimization_results)
        
        # Export recommendations
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv_data = export_recommendations_to_csv(st.session_state.optimization_results)
                st.download_button(
                    label="📥 Download Recommendations CSV",
                    data=csv_data,
                    file_name=f"reader_writer_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                pdf_data = generate_optimization_report_pdf(st.session_state.optimization_results)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"reader_writer_optimization_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

def is_enhanced_environment_data(environment_specs):
    """Check if environment specs contain enhanced cluster data"""
    if not environment_specs:
        return False
    
    sample_spec = next(iter(environment_specs.values()))
    enhanced_fields = ['workload_pattern', 'read_write_ratio', 'multi_az_writer']
    
    return any(field in sample_spec for field in enhanced_fields)

# Add the missing set_optimization_preferences method to OptimizedReaderWriterAnalyzer class
def set_optimization_preferences(self, goal='balanced', consider_reserved=True, environment_priority='production_first'):
    """Set optimization preferences for the analyzer"""
    self.optimization_goal = goal
    self.consider_reserved_instances = consider_reserved
    self.environment_priority = environment_priority
    
    # Adjust scoring weights based on goal
    if goal == 'cost_efficiency':
        self.performance_weight = 0.2
        self.cost_weight = 0.6
        self.suitability_weight = 0.2
    elif goal == 'performance':
        self.performance_weight = 0.7
        self.cost_weight = 0.1
        self.suitability_weight = 0.2
    else:  # balanced
        self.performance_weight = 0.4
        self.cost_weight = 0.3
        self.suitability_weight = 0.3

# Fixed vROPs section integration
def show_vrops_results_tab():
    """Show vROPs analysis results in a dedicated tab"""
    
    st.markdown("### 📊 vROps Performance Analysis")
    
    if not hasattr(st.session_state, 'vrops_analysis') or not st.session_state.vrops_analysis:
        st.info("📊 vROps analysis not available. Configure vROps metrics in the Environment Setup section.")
        
        # Show instructions for vROps setup
        with st.expander("🔧 How to Configure vROps Analysis"):
            st.markdown("""
            **To enable vROps analysis:**
            
            1. Go to the **Environment Setup** section
            2. Look for the **vROps Metrics Analysis** section
            3. Configure your current VM metrics:
               - CPU usage and allocation
               - Memory usage and allocation  
               - Storage IOPS and latency
               - Network utilization
            4. Run the vROps analysis
            5. Return here to view detailed results
            
            **Benefits of vROps Integration:**
            - Right-size AWS instances based on actual usage
            - Identify over/under-provisioned resources
            - Performance baseline establishment
            - Cost optimization opportunities
            """)
        return
    
    # Display vROps analysis results
    vrops_data = st.session_state.vrops_analysis
    
    # Overall health summary
    st.markdown("#### 🏥 Overall Health Score")
    
    health_scores = []
    for env_name, analysis in vrops_data.items():
        if isinstance(analysis, dict) and 'performance_scores' in analysis:
            health_scores.append(analysis['performance_scores'].get('overall_health', 0))
    
    if health_scores:
        avg_health = sum(health_scores) / len(health_scores)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Health Score", f"{avg_health:.1f}/100")
        
        with col2:
            healthy_envs = sum(1 for score in health_scores if score >= 80)
            st.metric("Healthy Environments", f"{healthy_envs}/{len(health_scores)}")
        
        with col3:
            critical_envs = sum(1 for score in health_scores if score < 60)
            st.metric("Critical Environments", critical_envs)
        
        with col4:
            st.metric("Total Environments", len(health_scores))
    
    # Environment-specific analysis
    for env_name, analysis in vrops_data.items():
        if 'error' in analysis:
            st.error(f"❌ {env_name}: {analysis['error']}")
            continue
        
        with st.expander(f"🖥️ {env_name} - Health: {analysis['performance_scores']['overall_health']:.1f}/100"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🖥️ CPU Analysis")
                cpu_analysis = analysis['cpu_analysis']
                st.metric("Max Usage", f"{cpu_analysis['max_usage_percent']:.1f}%")
                st.metric("Required Capacity", f"{cpu_analysis['required_capacity_percent']:.1f}%")
                st.write(f"**Recommendation:** {cpu_analysis['scaling_recommendation']}")
            
            with col2:
                st.markdown("#### 💾 Memory Analysis")
                memory_analysis = analysis['memory_analysis']
                st.metric("Max Usage", f"{memory_analysis['max_usage_percent']:.1f}%")
                st.metric("Required Capacity", f"{memory_analysis['required_capacity_percent']:.1f}%")
                st.write(f"**Recommendation:** {memory_analysis['scaling_recommendation']}")
            
            with col3:
                st.markdown("#### 📊 Performance Scores")
                scores = analysis['performance_scores']
                st.metric("CPU Health", f"{scores['cpu_health']}/100")
                st.metric("Memory Health", f"{scores['memory_health']}/100")
                st.metric("Storage Health", f"{scores['storage_health']}/100")
            
            # Instance recommendations
            st.markdown("#### 🎯 AWS Instance Recommendations")
            
            recommendations = analysis.get('instance_recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations[:3]):  # Show top 3
                    st.markdown(f"""
                    **{i+1}. {rec['instance_type']}** (Fit Score: {rec['fit_score']:.1f}/100)  
                    vCPUs: {rec['vcpu']}, Memory: {rec['memory_gb']}GB  
                    💡 {rec['recommendation_reason']}
                    """)
            else:
                st.info("No specific instance recommendations available")

# Fixed show_storage_validation_widget function
def show_storage_validation_widget():
    """Show storage validation widget in environment setup"""
    
    # Early return if required data is missing
    if not st.session_state.migration_params or not st.session_state.environment_specs:
        return
    
    # Get migration data size
    migration_data_gb = st.session_state.migration_params.get('data_size_gb', 0)
    storage_manager = StorageConfigurationManager()
    
    st.markdown("#### 🔍 Storage Configuration Validation")
    
    # Validate storage configuration
    validation = storage_manager.validate_storage_configuration(
        migration_data_gb, 
        st.session_state.environment_specs
    )
    
    # Status indicator section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_icon = "✅" if validation['is_valid'] else "❌"
        status_text = 'Valid' if validation['is_valid'] else 'Invalid'
        st.metric("Configuration Status", f"{status_icon} {status_text}")
    
    with col2:
        total_storage = validation['total_env_storage']
        st.metric("Total Environment Storage", f"{total_storage:,} GB")
    
    with col3:
        if validation['migration_to_env_ratio'] > 0:
            ratio = validation['migration_to_env_ratio']
            st.metric("Storage Ratio", f"{ratio:.1f}x")
    
    # Display errors if any
    if validation['errors']:
        st.error("🚨 **Critical Issues:**")
        for error in validation['errors']:
            st.error(f"• {error}")
    
    # Display warnings if any
    if validation['warnings']:
        st.warning("⚠️ **Warnings:**")
        for warning in validation['warnings']:
            st.warning(f"• {warning}")
    
    # Display recommendations if any
    if validation['recommendations']:
        st.info("💡 **Recommendations:**")
        for rec in validation['recommendations']:
            st.info(f"• {rec}")
    
    # Environment-specific analysis section
    if validation['environment_analysis']:
        with st.expander("📊 Environment Analysis Details"):
            analysis_data = []
            
            # Status icon mapping
            status_icons = {
                'good': '✅',
                'warning': '⚠️',
                'critical': '🚨',
                'over_provisioned': '💰'
            }
            
            # Process each environment analysis
            for env_name, analysis in validation['environment_analysis'].items():
                status_icon = status_icons.get(analysis['status'], '❓')
                
                analysis_data.append({
                    'Environment': env_name,
                    'Current (GB)': f"{analysis['current_storage_gb']:,}",
                    'Recommended (GB)': f"{analysis['recommended_storage_gb']:,}",
                    'Difference': f"{analysis['difference_gb']:+,} GB ({analysis['difference_percent']:+.1f}%)",
                    'Status': f"{status_icon} {analysis['status'].title()}"
                })
            
            # Create and display dataframe
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
            else:
                st.info("No environment analysis data available.")

# Fixed show_enhanced_environment_setup_with_cluster_config function
def show_enhanced_environment_setup_with_cluster_config():
    """Enhanced environment setup with Writer/Reader configuration"""
    st.markdown("## 📊 Database Cluster Configuration")
    
    # Add storage auto-calculator right after the title
    if st.session_state.migration_params:
        with st.expander("🤖 Storage Auto-Calculator (Optional)", expanded=False):
            show_storage_auto_calculator()
        st.markdown("---")  # Add a separator
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Configuration method selection
    config_method = st.radio(
        "Choose configuration method:",
        [
            "📝 Manual Cluster Configuration", 
            "📁 Bulk Upload with Cluster Details",
            "🔄 Simple Configuration (Legacy)"
        ],
        horizontal=True
    )
    
    if config_method == "📝 Manual Cluster Configuration":
        show_manual_cluster_configuration()
    elif config_method == "📁 Bulk Upload with Cluster Details":
        show_bulk_cluster_upload()
    else:
        show_simple_configuration()
    
    # Add vROps section
    st.markdown("---")
    show_vrops_section()
    
    # Add storage validation at the end
    if st.session_state.migration_params and st.session_state.environment_specs:
        st.markdown("---")  # Add a separator
        show_storage_validation_widget()

# Fixed show_vrops_section function
def show_vrops_section():
    """Show vROps metrics configuration section"""
    st.markdown("## 📊 vROps Metrics Analysis")
    
    with st.expander("Configure vROPs Metrics", expanded=False):
        st.markdown("### CPU Metrics")
        col1, col2 = st.columns(2)
        with col1:
            max_cpu = st.number_input("Max CPU Usage (%)", min_value=0, max_value=100, value=80)
            avg_cpu = st.number_input("Avg CPU Usage (%)", min_value=0, max_value=100, value=45)
        with col2:
            cpu_cores = st.number_input("CPU Cores Allocated", min_value=1, value=8)
            cpu_ready = st.number_input("CPU Ready Time (ms)", min_value=0, value=500)
        
        st.markdown("### Memory Metrics")
        col1, col2 = st.columns(2)
        with col1:
            max_mem = st.number_input("Max Memory Usage (%)", min_value=0, max_value=100, value=75)
            avg_mem = st.number_input("Avg Memory Usage (%)", min_value=0, max_value=100, value=50)
        with col2:
            mem_alloc = st.number_input("Memory Allocated (GB)", min_value=1, value=32)
            mem_balloon = st.number_input("Ballooned Memory (GB)", min_value=0, value=0)
        
        st.markdown("### Storage Metrics")
        col1, col2 = st.columns(2)
        with col1:
            max_iops = st.number_input("Max IOPS", min_value=0, value=5000)
            max_latency = st.number_input("Max Disk Latency (ms)", min_value=0, value=10)
        with col2:
            storage_used = st.number_input("Storage Used (GB)", min_value=0, value=500)
            storage_alloc = st.number_input("Storage Allocated (GB)", min_value=0, value=600)
        
        # Store metrics in session state
        st.session_state.vrops_metrics = {
            'max_cpu_usage_percent': max_cpu,
            'avg_cpu_usage_percent': avg_cpu,
            'cpu_cores_allocated': cpu_cores,
            'cpu_ready_time_ms': cpu_ready,
            'max_memory_usage_percent': max_mem,
            'avg_memory_usage_percent': avg_mem,
            'memory_allocated_gb': mem_alloc,
            'memory_balloon_gb': mem_balloon,
            'max_iops_total': max_iops,
            'max_disk_latency_ms': max_latency,
            'storage_used_gb': storage_used,
            'storage_allocated_gb': storage_alloc
        }
    
    if st.button("Run vROPS Analysis", type="primary"):
        if hasattr(st.session_state, 'vrops_metrics'):
            analyzer = VRopsMetricsAnalyzer()
            st.session_state.vrops_analysis = analyzer.analyze_metrics(
                {"PROD": st.session_state.vrops_metrics}
            )
            st.success("vROPS analysis completed!")
        else:
            st.error("Please configure vROPs metrics first")

# Fixed show_simple_configuration function
def show_simple_configuration():
    """Show simple/legacy configuration interface"""
    
    st.markdown("### 🔄 Simple Configuration (Legacy Mode)")
    st.info("This is the simplified configuration mode. For advanced cluster features, use Manual Cluster Configuration.")
    
    # Number of environments
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
                    key=f"simple_env_name_{i}"
                )
                
                cpu_cores = st.number_input(
                    "CPU Cores",
                    min_value=1, max_value=128,
                    value=[4, 8, 16, 32][min(i, 3)],
                    key=f"simple_cpu_{i}"
                )
                
                ram_gb = st.number_input(
                    "RAM (GB)",
                    min_value=4, max_value=1024,
                    value=[16, 32, 64, 128][min(i, 3)],
                    key=f"simple_ram_{i}"
                )
                
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"simple_storage_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"simple_usage_{i}"
                )
                
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"simple_connections_{i}"
                )
                
                environment_specs[env_name] = {
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'daily_usage_hours': daily_usage_hours,
                    'peak_connections': peak_connections,
                    # Add minimal enhanced fields for compatibility
                    'environment_type': 'production' if 'prod' in env_name.lower() else 'development',
                    'workload_pattern': 'balanced',
                    'read_write_ratio': 70
                }
    
    if st.button("💾 Save Simple Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Simple configuration saved!")
        
        # Display summary
        st.markdown("#### 📊 Configuration Summary")
        summary_df = pd.DataFrame.from_dict(environment_specs, orient='index')
        st.dataframe(summary_df, use_container_width=True)

# Fixed show_storage_auto_calculator function
def show_storage_auto_calculator():
    """Helper to auto-calculate environment storage"""
    
    if not st.session_state.migration_params:
        st.warning("Please configure migration parameters first")
        return
    
    migration_data_size = st.session_state.migration_params.get('data_size_gb', 1000)
    
    st.markdown("### 🤖 Storage Auto-Calculator")
    st.info(f"Base migration data size: {migration_data_size:,} GB")
    
    # Environment multipliers
    st.markdown("#### Environment Size Multipliers")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prod_mult = st.slider("Production", 0.8, 2.0, 1.2, 0.1, key="prod_mult_calc")
    with col2:
        stage_mult = st.slider("Staging", 0.5, 1.5, 0.8, 0.1, key="stage_mult_calc")
    with col3:
        test_mult = st.slider("Testing/QA", 0.3, 1.0, 0.6, 0.1, key="test_mult_calc")
    with col4:
        dev_mult = st.slider("Development", 0.1, 0.8, 0.3, 0.1, key="dev_mult_calc")
    
    # Calculate and show recommendations
    st.markdown("#### 📊 Recommended Storage Sizes")
    
    recommendations = {
        'Production': int(migration_data_size * prod_mult),
        'Staging': int(migration_data_size * stage_mult),
        'Testing': int(migration_data_size * test_mult),
        'QA': int(migration_data_size * test_mult),
        'Development': int(migration_data_size * dev_mult)
    }
    
    rec_data = []
    for env_type, recommended_gb in recommendations.items():
        growth_buffer = int(recommended_gb * 0.3)  # 30% growth buffer
        final_size = recommended_gb + growth_buffer
        
        rec_data.append({
            'Environment Type': env_type,
            'Base Size (GB)': f"{recommended_gb:,}",
            'Growth Buffer (GB)': f"{growth_buffer:,}",
            'Recommended Total (GB)': f"{final_size:,}"
        })
    
    rec_df = pd.DataFrame(rec_data)
    st.dataframe(rec_df, use_container_width=True)
    
    # Option to apply to current environments
    if st.session_state.environment_specs:
        st.markdown("#### ⚙️ Apply to Current Environments")
        
        if st.button("🎯 Auto-Calculate Storage for Current Environments", key="apply_storage_calc"):
            updated_specs = {}
            
            for env_name, specs in st.session_state.environment_specs.items():
                env_type = specs.get('environment_type', env_name).lower()
                
                # Match environment type to multiplier
                if 'prod' in env_type:
                    multiplier = prod_mult
                elif 'stag' in env_type:
                    multiplier = stage_mult
                elif any(x in env_type for x in ['test', 'qa', 'uat']):
                    multiplier = test_mult
                elif 'dev' in env_type:
                    multiplier = dev_mult
                else:
                    multiplier = 0.5  # Default
                
                # Calculate new storage
                base_storage = int(migration_data_size * multiplier)
                recommended_storage = int(base_storage * 1.3)  # Add 30% buffer
                
                # Update specs
                updated_specs[env_name] = {
                    **specs,  # Keep existing configuration
                    'storage_gb': recommended_storage,
                    'original_storage_gb': specs.get('storage_gb', 0),
                    'auto_calculated': True
                }
            
            # Apply changes
            st.session_state.environment_specs = updated_specs
            st.success("✅ Storage auto-calculated and applied!")
            st.balloons()
            
            # Show what changed
            st.markdown("**Changes Applied:**")
            for env_name, specs in updated_specs.items():
                old_size = specs.get('original_storage_gb', 0)
                new_size = specs.get('storage_gb', 0)
                if old_size != new_size:
                    st.write(f"• {env_name}: {old_size:,} GB → {new_size:,} GB")

# Fixed VRopsMetricsAnalyzer class
class VRopsMetricsAnalyzer:
    """Comprehensive vROps metrics analyzer for VM to AWS instance sizing"""
    
    def __init__(self):
        self.instance_specs = self._initialize_aws_instance_specs()
        self.performance_thresholds = self._initialize_performance_thresholds()
    
    def _initialize_aws_instance_specs(self):
        """Initialize AWS instance specifications for comparison"""
        return {
            'db.t3.micro': {'vcpu': 2, 'memory_gb': 1, 'suitable_for': ['dev', 'test']},
            'db.t3.small': {'vcpu': 2, 'memory_gb': 2, 'suitable_for': ['dev', 'test']},
            'db.t3.medium': {'vcpu': 2, 'memory_gb': 4, 'suitable_for': ['dev', 'small_prod']},
            'db.t3.large': {'vcpu': 2, 'memory_gb': 8, 'suitable_for': ['dev', 'small_prod']},
            'db.t3.xlarge': {'vcpu': 4, 'memory_gb': 16, 'suitable_for': ['staging', 'medium_prod']},
            'db.t3.2xlarge': {'vcpu': 8, 'memory_gb': 32, 'suitable_for': ['staging', 'medium_prod']},
            'db.r5.large': {'vcpu': 2, 'memory_gb': 16, 'suitable_for': ['memory_intensive', 'analytics']},
            'db.r5.xlarge': {'vcpu': 4, 'memory_gb': 32, 'suitable_for': ['production', 'memory_intensive']},
            'db.r5.2xlarge': {'vcpu': 8, 'memory_gb': 64, 'suitable_for': ['large_production', 'analytics']},
            'db.r5.4xlarge': {'vcpu': 16, 'memory_gb': 128, 'suitable_for': ['large_production', 'high_memory']},
            'db.r5.8xlarge': {'vcpu': 32, 'memory_gb': 256, 'suitable_for': ['enterprise', 'high_performance']},
            'db.r5.12xlarge': {'vcpu': 48, 'memory_gb': 384, 'suitable_for': ['enterprise', 'very_high_memory']},
            'db.r5.16xlarge': {'vcpu': 64, 'memory_gb': 512, 'suitable_for': ['enterprise', 'extreme_performance']},
            'db.r5.24xlarge': {'vcpu': 96, 'memory_gb': 768, 'suitable_for': ['enterprise', 'maximum_performance']},
            'db.r6i.large': {'vcpu': 2, 'memory_gb': 16, 'suitable_for': ['memory_intensive', 'latest_gen']},
            'db.r6i.xlarge': {'vcpu': 4, 'memory_gb': 32, 'suitable_for': ['production', 'latest_gen']},
            'db.r6i.2xlarge': {'vcpu': 8, 'memory_gb': 64, 'suitable_for': ['large_production', 'latest_gen']},
            'db.r6i.4xlarge': {'vcpu': 16, 'memory_gb': 128, 'suitable_for': ['enterprise', 'latest_gen']},
            'db.c5.large': {'vcpu': 2, 'memory_gb': 4, 'suitable_for': ['compute_intensive', 'oltp']},
            'db.c5.xlarge': {'vcpu': 4, 'memory_gb': 8, 'suitable_for': ['compute_intensive', 'oltp']},
            'db.c5.2xlarge': {'vcpu': 8, 'memory_gb': 16, 'suitable_for': ['high_cpu', 'oltp']},
            'db.c5.4xlarge': {'vcpu': 16, 'memory_gb': 32, 'suitable_for': ['very_high_cpu', 'oltp']}
        }
    
    def _initialize_performance_thresholds(self):
        """Initialize performance thresholds for health scoring"""
        return {
            'cpu': {
                'excellent': 70,
                'good': 80,
                'warning': 90,
                'critical': 95
            },
            'memory': {
                'excellent': 70,
                'good': 80,
                'warning': 90,
                'critical': 95
            },
            'storage': {
                'latency_excellent': 5,
                'latency_good': 10,
                'latency_warning': 20,
                'latency_critical': 50,
                'iops_utilization_good': 70,
                'iops_utilization_warning': 85,
                'iops_utilization_critical': 95
            }
        }
    
    def analyze_metrics(self, metrics_data: Dict) -> Dict:
        """Analyze vROPs metrics and provide AWS instance recommendations"""
        
        results = {}
        
        for env_name, metrics in metrics_data.items():
            try:
                env_analysis = self._analyze_single_environment(env_name, metrics)
                results[env_name] = env_analysis
            except Exception as e:
                results[env_name] = {'error': f'Analysis failed: {str(e)}'}
        
        return results
    
    def _analyze_single_environment(self, env_name: str, metrics: Dict) -> Dict:
        """Analyze metrics for a single environment"""
        
        # Extract metrics
        max_cpu_usage = metrics.get('max_cpu_usage_percent', 0)
        avg_cpu_usage = metrics.get('avg_cpu_usage_percent', 0)
        cpu_cores = metrics.get('cpu_cores_allocated', 4)
        cpu_ready_time = metrics.get('cpu_ready_time_ms', 0)
        
        max_memory_usage = metrics.get('max_memory_usage_percent', 0)
        avg_memory_usage = metrics.get('avg_memory_usage_percent', 0)
        memory_allocated = metrics.get('memory_allocated_gb', 16)
        memory_balloon = metrics.get('memory_balloon_gb', 0)
        
        max_iops = metrics.get('max_iops_total', 0)
        max_latency = metrics.get('max_disk_latency_ms', 0)
        storage_used = metrics.get('storage_used_gb', 0)
        storage_allocated = metrics.get('storage_allocated_gb', 0)
        
        # Analyze CPU performance
        cpu_analysis = self._analyze_cpu_performance(
            max_cpu_usage, avg_cpu_usage, cpu_cores, cpu_ready_time
        )
        
        # Analyze memory performance
        memory_analysis = self._analyze_memory_performance(
            max_memory_usage, avg_memory_usage, memory_allocated, memory_balloon
        )
        
        # Analyze storage performance
        storage_analysis = self._analyze_storage_performance(
            max_iops, max_latency, storage_used, storage_allocated
        )
        
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(
            cpu_analysis, memory_analysis, storage_analysis
        )
        
        # Generate AWS instance recommendations
        instance_recommendations = self._generate_instance_recommendations(
            cpu_analysis, memory_analysis, storage_analysis, env_name
        )
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(
            cpu_analysis, memory_analysis, storage_analysis
        )
        
        return {
            'cpu_analysis': cpu_analysis,
            'memory_analysis': memory_analysis,
            'storage_analysis': storage_analysis,
            'performance_scores': performance_scores,
            'instance_recommendations': instance_recommendations,
            'optimization_recommendations': optimization_recommendations,
            'environment_name': env_name
        }
    
    def _analyze_cpu_performance(self, max_cpu: float, avg_cpu: float, 
                                cpu_cores: int, cpu_ready: float) -> Dict:
        """Analyze CPU performance metrics"""
        
        # Calculate required capacity (max usage + buffer)
        cpu_buffer = 20  # 20% buffer for peaks
        required_capacity = max_cpu + cpu_buffer
        
        # Determine scaling recommendation
        if max_cpu > 90:
            scaling_recommendation = "Critical: Immediate CPU scaling required"
            scaling_action = "scale_up_immediately"
        elif max_cpu > 80:
            scaling_recommendation = "Warning: Consider CPU scaling"
            scaling_action = "scale_up_soon"
        elif max_cpu < 40:
            scaling_recommendation = "Optimization: CPU resources may be over-provisioned"
            scaling_action = "consider_scale_down"
        else:
            scaling_recommendation = "Good: CPU utilization is within optimal range"
            scaling_action = "maintain"
        
        # CPU Ready Time analysis
        cpu_ready_status = "Good"
        if cpu_ready > 1000:
            cpu_ready_status = "Critical - High CPU contention"
        elif cpu_ready > 500:
            cpu_ready_status = "Warning - Moderate CPU contention"
        
        return {
            'max_usage_percent': max_cpu,
            'avg_usage_percent': avg_cpu,
            'allocated_cores': cpu_cores,
            'required_capacity_percent': min(100, required_capacity),
            'scaling_recommendation': scaling_recommendation,
            'scaling_action': scaling_action,
            'cpu_ready_time_ms': cpu_ready,
            'cpu_ready_status': cpu_ready_status,
            'efficiency_score': max(0, 100 - abs(70 - max_cpu))  # Optimal around 70%
        }
    
    def _analyze_memory_performance(self, max_memory: float, avg_memory: float,
                                   memory_allocated: float, memory_balloon: float) -> Dict:
        """Analyze memory performance metrics"""
        
        # Calculate required capacity
        memory_buffer = 15  # 15% buffer for memory
        required_capacity = max_memory + memory_buffer
        
        # Determine scaling recommendation
        if max_memory > 90 or memory_balloon > 0:
            scaling_recommendation = "Critical: Memory scaling required"
            scaling_action = "scale_up_immediately"
        elif max_memory > 80:
            scaling_recommendation = "Warning: Consider memory scaling"
            scaling_action = "scale_up_soon"
        elif max_memory < 50 and memory_balloon == 0:
            scaling_recommendation = "Optimization: Memory may be over-provisioned"
            scaling_action = "consider_scale_down"
        else:
            scaling_recommendation = "Good: Memory utilization is optimal"
            scaling_action = "maintain"
        
        # Memory pressure analysis
        memory_pressure = "None"
        if memory_balloon > memory_allocated * 0.1:
            memory_pressure = "High - Significant ballooning detected"
        elif memory_balloon > 0:
            memory_pressure = "Moderate - Some ballooning detected"
        
        return {
            'max_usage_percent': max_memory,
            'avg_usage_percent': avg_memory,
            'allocated_gb': memory_allocated,
            'required_capacity_percent': min(100, required_capacity),
            'scaling_recommendation': scaling_recommendation,
            'scaling_action': scaling_action,
            'balloon_memory_gb': memory_balloon,
            'memory_pressure': memory_pressure,
            'efficiency_score': max(0, 100 - abs(75 - max_memory))  # Optimal around 75%
        }
    
    def _analyze_storage_performance(self, max_iops: float, max_latency: float,
                                   storage_used: float, storage_allocated: float) -> Dict:
        """Analyze storage performance metrics"""
        
        # Storage utilization
        storage_utilization = (storage_used / storage_allocated * 100) if storage_allocated > 0 else 0
        
        # Latency analysis
        latency_status = "Excellent"
        if max_latency > 50:
            latency_status = "Critical - Very high latency"
        elif max_latency > 20:
            latency_status = "Warning - High latency"
        elif max_latency > 10:
            latency_status = "Good - Moderate latency"
        
        # IOPS analysis
        iops_status = "Unknown"
        if max_iops > 0:
            if max_iops > 10000:
                iops_status = "High performance workload"
            elif max_iops > 5000:
                iops_status = "Medium performance workload"
            elif max_iops > 1000:
                iops_status = "Standard workload"
            else:
                iops_status = "Light workload"
        
        # Storage recommendations
        storage_recommendations = []
        if storage_utilization > 80:
            storage_recommendations.append("Consider increasing storage capacity")
        if max_latency > 20:
            storage_recommendations.append("Consider higher performance storage (SSD/NVMe)")
        if max_iops > 10000:
            storage_recommendations.append("Consider provisioned IOPS storage")
        
        return {
            'max_iops': max_iops,
            'max_latency_ms': max_latency,
            'storage_used_gb': storage_used,
            'storage_allocated_gb': storage_allocated,
            'storage_utilization_percent': storage_utilization,
            'latency_status': latency_status,
            'iops_status': iops_status,
            'recommendations': storage_recommendations,
            'efficiency_score': max(0, 100 - max_latency)  # Lower latency = higher score
        }
    
    def _calculate_performance_scores(self, cpu_analysis: Dict, memory_analysis: Dict,
                                    storage_analysis: Dict) -> Dict:
        """Calculate overall performance health scores"""
        
        # CPU Health Score
        cpu_health = 100
        if cpu_analysis['max_usage_percent'] > 90:
            cpu_health = 30
        elif cpu_analysis['max_usage_percent'] > 80:
            cpu_health = 60
        elif cpu_analysis['max_usage_percent'] < 40:
            cpu_health = 70  # Over-provisioned but not critical
        else:
            cpu_health = 90
        
        # Adjust for CPU ready time
        if cpu_analysis['cpu_ready_time_ms'] > 1000:
            cpu_health = min(cpu_health, 40)
        elif cpu_analysis['cpu_ready_time_ms'] > 500:
            cpu_health = min(cpu_health, 70)
        
        # Memory Health Score
        memory_health = 100
        if memory_analysis['max_usage_percent'] > 90 or memory_analysis['balloon_memory_gb'] > 0:
            memory_health = 30
        elif memory_analysis['max_usage_percent'] > 80:
            memory_health = 60
        elif memory_analysis['max_usage_percent'] < 50:
            memory_health = 75  # Over-provisioned
        else:
            memory_health = 90
        
        # Storage Health Score
        storage_health = 100
        if storage_analysis['max_latency_ms'] > 50:
            storage_health = 30
        elif storage_analysis['max_latency_ms'] > 20:
            storage_health = 60
        elif storage_analysis['max_latency_ms'] > 10:
            storage_health = 80
        else:
            storage_health = 95
        
        # Overall Health Score (weighted average)
        overall_health = (cpu_health * 0.4 + memory_health * 0.4 + storage_health * 0.2)
        
        return {
            'cpu_health': int(cpu_health),
            'memory_health': int(memory_health),
            'storage_health': int(storage_health),
            'overall_health': int(overall_health)
        }
    
    def _generate_instance_recommendations(self, cpu_analysis: Dict, memory_analysis: Dict,
                                         storage_analysis: Dict, env_name: str) -> List[Dict]:
        """Generate AWS instance recommendations based on performance analysis"""
        
        required_vcpu = max(2, int(cpu_analysis['allocated_cores'] * (cpu_analysis['required_capacity_percent'] / 100)))
        required_memory = max(4, int(memory_analysis['allocated_gb'] * (memory_analysis['required_capacity_percent'] / 100)))
        
        # Find suitable instances
        suitable_instances = []
        
        for instance_type, specs in self.instance_specs.items():
            if specs['vcpu'] >= required_vcpu and specs['memory_gb'] >= required_memory:
                
                # Calculate fit score
                vcpu_efficiency = min(100, (required_vcpu / specs['vcpu']) * 100)
                memory_efficiency = min(100, (required_memory / specs['memory_gb']) * 100)
                
                # Bonus for not over-provisioning
                if specs['vcpu'] <= required_vcpu * 1.5 and specs['memory_gb'] <= required_memory * 1.5:
                    size_bonus = 20
                else:
                    size_bonus = 0
                
                fit_score = (vcpu_efficiency + memory_efficiency) / 2 + size_bonus
                
                # Generate recommendation reason
                recommendation_reason = self._generate_recommendation_reason(
                    instance_type, specs, cpu_analysis, memory_analysis, storage_analysis, env_name
                )
                
                suitable_instances.append({
                    'instance_type': instance_type,
                    'vcpu': specs['vcpu'],
                    'memory_gb': specs['memory_gb'],
                    'fit_score': min(100, fit_score),
                    'required_vcpu': required_vcpu,
                    'required_memory_gb': required_memory,
                    'recommendation_reason': recommendation_reason
                })
        
        # Sort by fit score and return top 5
        suitable_instances.sort(key=lambda x: x['fit_score'], reverse=True)
        return suitable_instances[:5]
    
    def _generate_recommendation_reason(self, instance_type: str, specs: Dict,
                                      cpu_analysis: Dict, memory_analysis: Dict,
                                      storage_analysis: Dict, env_name: str) -> str:
        """Generate human-readable recommendation reasoning"""
        
        reasons = []
        
        # CPU reasoning
        if cpu_analysis['max_usage_percent'] > 80:
            reasons.append(f"Provides {specs['vcpu']} vCPUs to handle high CPU utilization")
        elif cpu_analysis['max_usage_percent'] < 40:
            reasons.append(f"Right-sized with {specs['vcpu']} vCPUs to avoid over-provisioning")
        else:
            reasons.append(f"Well-balanced {specs['vcpu']} vCPUs for current workload")
        
        # Memory reasoning
        if memory_analysis['balloon_memory_gb'] > 0:
            reasons.append(f"{specs['memory_gb']}GB memory eliminates memory pressure")
        elif memory_analysis['max_usage_percent'] > 80:
            reasons.append(f"{specs['memory_gb']}GB memory provides adequate headroom")
        else:
            reasons.append(f"{specs['memory_gb']}GB memory matches current requirements")
        
        # Instance family reasoning
        if 'r5' in instance_type or 'r6i' in instance_type:
            reasons.append("Memory-optimized for database workloads")
        elif 'c5' in instance_type:
            reasons.append("Compute-optimized for CPU-intensive workloads")
        elif 't3' in instance_type:
            reasons.append("Burstable performance suitable for variable workloads")
        
        return "; ".join(reasons)
    
    def _generate_optimization_recommendations(self, cpu_analysis: Dict, memory_analysis: Dict,
                                             storage_analysis: Dict) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # CPU optimizations
        if cpu_analysis['scaling_action'] == 'scale_up_immediately':
            recommendations.append("🔴 CRITICAL: Increase CPU allocation immediately to prevent performance degradation")
        elif cpu_analysis['scaling_action'] == 'scale_up_soon':
            recommendations.append("🟡 WARNING: Plan CPU capacity increase within next maintenance window")
        elif cpu_analysis['scaling_action'] == 'consider_scale_down':
            recommendations.append("💡 OPTIMIZATION: Consider reducing CPU allocation to optimize costs")
        
        # Memory optimizations
        if memory_analysis['scaling_action'] == 'scale_up_immediately':
            recommendations.append("🔴 CRITICAL: Increase memory allocation immediately")
            if memory_analysis['balloon_memory_gb'] > 0:
                recommendations.append(f"Memory ballooning detected ({memory_analysis['balloon_memory_gb']:.1f}GB) - clear sign of memory pressure")
        elif memory_analysis['scaling_action'] == 'scale_up_soon':
            recommendations.append("🟡 WARNING: Plan memory capacity increase")
        elif memory_analysis['scaling_action'] == 'consider_scale_down':
            recommendations.append("💡 OPTIMIZATION: Consider reducing memory allocation to optimize costs")
        
        # Storage optimizations
        for storage_rec in storage_analysis['recommendations']:
            recommendations.append(f"💾 STORAGE: {storage_rec}")
        
        # General optimizations
        if cpu_analysis['efficiency_score'] < 60 or memory_analysis['efficiency_score'] < 60:
            recommendations.append("📊 Consider workload optimization to improve resource utilization")
        
        if not recommendations:
            recommendations.append("✅ Current configuration appears well-optimized")
        
        return recommendations

# Fixed show_manual_cluster_configuration function
def show_manual_cluster_configuration():
    """Show manual cluster configuration with Writer/Reader options"""
    
    st.markdown("### 📝 Database Cluster Configuration")
    
    # Number of environments
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    for i in range(num_environments):
        with st.expander(f"🏢 Environment {i+1} - Cluster Configuration", expanded=i == 0):
            
            # Basic environment info
            col1, col2 = st.columns(2)
            
            with col1:
                env_name = st.text_input(
                    "Environment Name",
                    value=default_names[i] if i < len(default_names) else f"Environment_{i+1}",
                    key=f"cluster_env_name_{i}"
                )
                
                environment_type = st.selectbox(
                    "Environment Type",
                    ["Production", "Staging", "Testing", "Development"],
                    index=min(i, 3),
                    key=f"cluster_env_type_{i}"
                )
            
            with col2:
                workload_pattern = st.selectbox(
                    "Workload Pattern",
                    ["balanced", "read_heavy", "write_heavy", "analytics"],
                    key=f"cluster_workload_{i}"
                )
                
                read_write_ratio = st.slider(
                    "Read/Write Ratio (% Reads)",
                    min_value=10, max_value=95, value=70,
                    key=f"cluster_read_ratio_{i}"
                )
            
            # Infrastructure configuration
            st.markdown("#### 💻 Infrastructure Requirements")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_cores = st.number_input(
                    "CPU Cores",
                    min_value=1, max_value=128,
                    value=[4, 8, 16, 32][min(i, 3)],
                    key=f"cluster_cpu_{i}"
                )
                
                ram_gb = st.number_input(
                    "RAM (GB)",
                    min_value=4, max_value=1024,
                    value=[16, 32, 64, 128][min(i, 3)],
                    key=f"cluster_ram_{i}"
                )
            
            with col2:
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"cluster_storage_{i}"
                )
                
                iops_requirement = st.number_input(
                    "IOPS Requirement",
                    min_value=100, max_value=50000,
                    value=[1000, 3000, 5000, 10000][min(i, 3)],
                    key=f"cluster_iops_{i}"
                )
            
            with col3:
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"cluster_connections_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"cluster_usage_{i}"
                )
            
            # Cluster configuration
            st.markdown("#### 🔗 Cluster Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                multi_az_writer = st.checkbox(
                    "Multi-AZ for Writer",
                    value=environment_type in ["Production", "Staging"],
                    key=f"cluster_multi_az_writer_{i}",
                    help="Deploy writer instance across multiple Availability Zones for high availability"
                )
                
                custom_reader_count = st.checkbox(
                    "Custom Reader Count",
                    value=False,
                    key=f"cluster_custom_readers_{i}"
                )
                
                if custom_reader_count:
                    num_readers = st.number_input(
                        "Number of Read Replicas",
                        min_value=0, max_value=5,
                        value=1 if environment_type == "Production" else 0,
                        key=f"cluster_num_readers_{i}"
                    )
                else:
                    # Auto-calculate based on environment and workload
                    cluster_config = DatabaseClusterConfiguration()
                    num_readers = cluster_config.calculate_optimal_readers(
                        environment_type.lower(), workload_pattern, peak_connections
                    )
                    st.info(f"Recommended readers: {num_readers} (auto-calculated)")
            
            with col2:
                multi_az_readers = st.checkbox(
                    "Multi-AZ for Readers",
                    value=environment_type == "Production",
                    key=f"cluster_multi_az_readers_{i}",
                    help="Deploy read replicas across multiple Availability Zones"
                )
                
                if num_readers > 0:
                    custom_reader_instance = st.checkbox(
                        "Custom Reader Instance Size",
                        value=False,
                        key=f"cluster_custom_reader_instance_{i}"
                    )
                    
                    if custom_reader_instance:
                        reader_instance_override = st.selectbox(
                            "Reader Instance Class",
                            ["db.t3.medium", "db.t3.large", "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge"],
                            key=f"cluster_reader_instance_{i}"
                        )
                    else:
                        reader_instance_override = None
                        st.info("Reader size will be auto-calculated based on writer size and workload")
                else:
                    reader_instance_override = None
            
            # Storage configuration
            st.markdown("#### 💾 Storage Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                storage_encrypted = st.checkbox(
                    "Encryption at Rest",
                    value=environment_type in ["Production", "Staging"],
                    key=f"cluster_encryption_{i}"
                )
            
            with col2:
                backup_retention = st.number_input(
                    "Backup Retention (Days)",
                    min_value=1, max_value=35,
                    value=30 if environment_type == "Production" else 7,
                    key=f"cluster_backup_{i}"
                )
            
            with col3:
                auto_storage_scaling = st.checkbox(
                    "Auto Storage Scaling",
                    value=True,
                    key=f"cluster_auto_scaling_{i}",
                    help="Automatically scale storage when needed"
                )
            
            # Store environment configuration
        environment_specs[env_name] = {
            'cpu_cores': cpu_cores,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'iops_requirement': iops_requirement,
            'peak_connections': peak_connections,
            'daily_usage_hours': daily_usage_hours,
            'workload_pattern': workload_pattern,
            'read_write_ratio': read_write_ratio,
            'environment_type': environment_type.lower(),
            'multi_az_writer': multi_az_writer,
            'multi_az_readers': multi_az_readers,
            'num_readers': num_readers if custom_reader_count else None,
            'reader_instance_override': reader_instance_override,
            'storage_encrypted': storage_encrypted,
            'backup_retention': backup_retention,
            'auto_storage_scaling': auto_storage_scaling
        }

if st.button("💾 Save Cluster Configuration", type="primary", use_container_width=True):
    st.session_state.environment_specs = environment_specs
    st.success("✅ Cluster configuration saved!")
    
    # Display summary
    st.markdown("#### 📊 Configuration Summary")
    summary_data = []
    for env_name, specs in environment_specs.items():
        summary_data.append({
            'Environment': env_name,
            'Type': specs['environment_type'].title(),
            'Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Storage': f"{specs['storage_gb']} GB ({specs['iops_requirement']} IOPS)",
            'Workload': f"{specs['workload_pattern']} ({specs['read_write_ratio']}% reads)",
            'Writer Multi-AZ': '✅' if specs['multi_az_writer'] else '❌',
            'Readers': f"{specs.get('num_readers', 'Auto')} ({'Multi-AZ' if specs['multi_az_readers'] else 'Single-AZ'})"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Show storage validation
    if st.session_state.migration_params:
        validate_storage_consistency()
    
    # Option to run immediate analysis
    st.markdown("---")
    if st.button("🚀 Analyze Cluster Configuration", type="secondary", use_container_width=True):
        with st.spinner("🔄 Analyzing cluster requirements..."):
            try:
                # Check if we have enhanced data
                is_enhanced = is_enhanced_environment_data(environment_specs)
                
                if is_enhanced:
                    # Use enhanced analyzer
                    analyzer = EnhancedMigrationAnalyzer()
                    recommendations = analyzer.calculate_enhanced_instance_recommendations(environment_specs)
                    st.session_state.enhanced_recommendations = recommendations
                    
                    # Calculate enhanced costs
                    if st.session_state.migration_params:
                        cost_analysis = analyzer.calculate_enhanced_migration_costs(
                            recommendations, 
                            st.session_state.migration_params
                        )
                        st.session_state.enhanced_analysis_results = cost_analysis
                    
                    st.success("✅ Enhanced cluster analysis complete!")
                    show_cluster_configuration_preview(recommendations)
                else:
                    # Use standard analyzer
                    analyzer = MigrationAnalyzer()
                    recommendations = analyzer.calculate_instance_recommendations(environment_specs)
                    st.session_state.recommendations = recommendations
                    
                    st.success("✅ Standard analysis complete!")
                    st.info("💡 For enhanced cluster analysis, use the cluster configuration options above.")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.code(str(e))

def show_bulk_cluster_upload():
    """Show bulk upload interface for cluster configurations"""
    
    st.markdown("### 📁 Bulk Cluster Configuration Upload")
    
    with st.expander("📋 Download Enhanced Template", expanded=False):
        st.markdown("""
        **Enhanced Template includes:**
        - Writer/Reader configuration
        - Multi-AZ options
        - Storage specifications
        - Workload patterns
        - IOPS requirements
        """)
        
        # Generate enhanced template
        enhanced_template = create_enhanced_cluster_template()
        csv_data = enhanced_template.to_csv(index=False)
        
        st.dataframe(enhanced_template, use_container_width=True)
        
        st.download_button(
            label="📥 Download Enhanced Cluster Template",
            data=csv_data,
            file_name="enhanced_cluster_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Cluster Configuration",
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file with cluster specifications"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Validate required columns
            required_cols = [
                'Environment_Name', 'Environment_Type', 'CPU_Cores', 'RAM_GB', 
                'Storage_GB', 'Workload_Pattern'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your file includes all required columns as shown in the template.")
                return
            
            # Process cluster data
            environment_specs = process_cluster_data(df)
            
            if environment_specs:
                st.session_state.environment_specs = environment_specs
                st.success(f"✅ Successfully processed {len(environment_specs)} cluster configurations!")
                
                # Show summary
                show_cluster_upload_summary(environment_specs)
                
                # Option to run immediate analysis
                if st.button("🚀 Analyze Uploaded Configuration", type="primary"):
                    with st.spinner("🔄 Analyzing uploaded cluster configurations..."):
                        try:
                            analyzer = EnhancedMigrationAnalyzer()
                            recommendations = analyzer.calculate_enhanced_instance_recommendations(environment_specs)
                            st.session_state.enhanced_recommendations = recommendations
                            
                            st.success("✅ Bulk configuration analysis complete!")
                            show_cluster_configuration_preview(recommendations)
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your file format and ensure it matches the template structure.")

def show_simple_configuration():
    """Show simple configuration for legacy compatibility"""
    
    st.markdown("### 📝 Simple Environment Configuration")
    st.info("💡 This is the legacy configuration mode. For advanced features, use Manual Cluster Configuration.")
    
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
                    key=f"simple_env_name_{i}"
                )
                
                cpu_cores = st.number_input(
                    "CPU Cores",
                    min_value=1, max_value=128,
                    value=[4, 8, 16, 32][min(i, 3)],
                    key=f"simple_cpu_{i}"
                )
                
                ram_gb = st.number_input(
                    "RAM (GB)",
                    min_value=4, max_value=1024,
                    value=[16, 32, 64, 128][min(i, 3)],
                    key=f"simple_ram_{i}"
                )
                
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"simple_storage_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"simple_usage_{i}"
                )
                
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"simple_connections_{i}"
                )
                
                environment_specs[env_name] = {
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'daily_usage_hours': daily_usage_hours,
                    'peak_connections': peak_connections,
                    'environment_type': 'production' if 'prod' in env_name.lower() else 'development'
                }
    
    if st.button("💾 Save Simple Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Simple environment configuration saved!")
        
        # Display summary
        st.markdown("#### 📊 Configuration Summary")
        summary_df = pd.DataFrame.from_dict(environment_specs, orient='index')
        st.dataframe(summary_df, use_container_width=True)
        
        # Show storage validation
        if st.session_state.migration_params:
            validate_storage_consistency()

def create_enhanced_cluster_template() -> pd.DataFrame:
    """Create enhanced cluster template with comprehensive fields"""
    
    template_data = {
        'Environment_Name': ['Production-Cluster', 'Staging-Cluster', 'QA-Cluster', 'Dev-Cluster'],
        'Environment_Type': ['Production', 'Staging', 'Testing', 'Development'],
        'CPU_Cores': [32, 16, 8, 4],
        'RAM_GB': [128, 64, 32, 16],
        'Storage_GB': [2000, 1000, 500, 200],
        'IOPS_Requirement': [10000, 5000, 3000, 1000],
        'Peak_Connections': [500, 200, 100, 50],
        'Daily_Usage_Hours': [24, 16, 12, 8],
        'Workload_Pattern': ['read_heavy', 'balanced', 'balanced', 'write_heavy'],
        'Read_Write_Ratio': [80, 70, 60, 40],
        'Multi_AZ_Writer': [True, True, False, False],
        'Multi_AZ_Readers': [True, False, False, False],
        'Num_Readers': [2, 1, 1, 0],
        'Storage_Encrypted': [True, True, False, False],
        'Backup_Retention_Days': [30, 14, 7, 7],
        'Auto_Storage_Scaling': [True, True, True, False]
    }
    
    return pd.DataFrame(template_data)

def process_cluster_data(df: pd.DataFrame) -> Dict:
    """Process uploaded cluster data with error handling"""
    
    environments = {}
    
    try:
        for _, row in df.iterrows():
            env_name = str(row['Environment_Name'])
            
            environments[env_name] = {
                'cpu_cores': int(row.get('CPU_Cores', 4)),
                'ram_gb': int(row.get('RAM_GB', 16)),
                'storage_gb': int(row.get('Storage_GB', 500)),
                'iops_requirement': int(row.get('IOPS_Requirement', 3000)),
                'peak_connections': int(row.get('Peak_Connections', 100)),
                'daily_usage_hours': int(row.get('Daily_Usage_Hours', 24)),
                'workload_pattern': str(row.get('Workload_Pattern', 'balanced')),
                'read_write_ratio': int(row.get('Read_Write_Ratio', 70)),
                'environment_type': str(row.get('Environment_Type', 'Production')).lower(),
                'multi_az_writer': bool(row.get('Multi_AZ_Writer', True)),
                'multi_az_readers': bool(row.get('Multi_AZ_Readers', False)),
                'num_readers': int(row.get('Num_Readers', 1)) if pd.notna(row.get('Num_Readers')) else None,
                'storage_encrypted': bool(row.get('Storage_Encrypted', True)),
                'backup_retention': int(row.get('Backup_Retention_Days', 7)),
                'auto_storage_scaling': bool(row.get('Auto_Storage_Scaling', True))
            }
        
        return environments
    
    except Exception as e:
        st.error(f"Error processing cluster data: {str(e)}")
        return {}

def show_cluster_upload_summary(environment_specs: Dict):
    """Show summary of uploaded cluster configurations"""
    
    st.markdown("#### 📊 Cluster Configuration Summary")
    
    summary_data = []
    for env_name, specs in environment_specs.items():
        summary_data.append({
            'Environment': env_name,
            'Type': specs['environment_type'].title(),
            'Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Storage': f"{specs['storage_gb']} GB ({specs['iops_requirement']} IOPS)",
            'Workload': f"{specs['workload_pattern']} ({specs['read_write_ratio']}% reads)",
            'Writer Multi-AZ': '✅' if specs['multi_az_writer'] else '❌',
            'Readers': f"{specs.get('num_readers', 'Auto')} ({'Multi-AZ' if specs['multi_az_readers'] else 'Single-AZ'})"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Show configuration statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cpu = sum([specs['cpu_cores'] for specs in environment_specs.values()])
        st.metric("Total CPU Cores", total_cpu)
    
    with col2:
        total_ram = sum([specs['ram_gb'] for specs in environment_specs.values()])
        st.metric("Total RAM (GB)", f"{total_ram:,}")
    
    with col3:
        total_storage = sum([specs['storage_gb'] for specs in environment_specs.values()])
        st.metric("Total Storage (GB)", f"{total_storage:,}")
    
    with col4:
        multi_az_count = sum([1 for specs in environment_specs.values() if specs['multi_az_writer']])
        st.metric("Multi-AZ Environments", multi_az_count)

def show_cluster_configuration_preview(recommendations: Dict):
    """Show preview of cluster configuration recommendations"""
    
    st.markdown("#### 🎯 Cluster Configuration Preview")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🏢 {env_name} Cluster Configuration"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Writer Configuration**")
                writer = rec['writer']
                st.write(f"Instance: {writer['instance_class']}")
                st.write(f"Multi-AZ: {'✅ Yes' if writer['multi_az'] else '❌ No'}")
                st.write(f"vCPUs: {writer['cpu_cores']}")
                st.write(f"Memory: {writer['ram_gb']} GB")
            
            with col2:
                st.markdown("**Reader Configuration**")
                readers = rec['readers']
                reader_count = readers['count']
                if reader_count > 0:
                    st.write(f"Count: {reader_count} replicas")
                    st.write(f"Instance: {readers['instance_class']}")
                    st.write(f"Multi-AZ: {'✅ Yes' if readers['multi_az'] else '❌ No'}")
                    st.write(f"vCPUs: {readers.get('cpu_cores', 'N/A')} each")
                else:
                    st.write("No read replicas")
                    st.write("Single writer configuration")
                    st.info("💡 Read replicas can improve read performance")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'✅ Yes' if storage['encrypted'] else '❌ No'}")
                
                # Monthly cost estimate if available
                if 'cost_estimate' in rec:
                    st.write(f"Est. Monthly Cost: ${rec['cost_estimate']:,.0f}")

def validate_storage_consistency():
    """Validate storage consistency between migration and environment configs"""
    
    if not st.session_state.migration_params or not st.session_state.environment_specs:
        return
    
    # Get migration data size
    migration_data_size = st.session_state.migration_params.get('data_size_gb', 0)
    
    # Calculate total environment storage
    total_env_storage = 0
    env_details = []
    
    for env_name, specs in st.session_state.environment_specs.items():
        env_storage = specs.get('storage_gb', 0)
        total_env_storage += env_storage
        
        env_details.append({
            'Environment': env_name,
            'Storage_GB': env_storage,
            'Env_Type': specs.get('environment_type', 'Unknown')
        })
    
    # Show analysis
    st.markdown("#### 💾 Storage Configuration Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Migration Data Size", f"{migration_data_size:,} GB")
    
    with col2:
        st.metric("Total Environment Storage", f"{total_env_storage:,} GB")
    
    with col3:
        if migration_data_size > 0:
            multiplier = total_env_storage / migration_data_size
            st.metric("Storage Multiplier", f"{multiplier:.1f}x")
            
          # Simple validation with recommendations
            if multiplier > 4:
                st.warning("⚠️ Environment storage is much larger than migration data. Consider optimizing storage allocation.")
            elif multiplier < 0.8:
                st.error("❌ Environment storage might be too small. Consider increasing storage capacity.")
            else:
                st.success("✅ Storage sizing looks reasonable for the migration scope.")
        else:
            st.info("ℹ️ Configure migration data size for storage analysis")
    
    # Show environment breakdown
    if env_details:
        st.markdown("**Environment Storage Breakdown:**")
        df = pd.DataFrame(env_details)
        st.dataframe(df, use_container_width=True)

def show_storage_auto_calculator():
    """Helper to auto-calculate environment storage"""
    
    if not st.session_state.migration_params:
        st.warning("Please configure migration parameters first")
        return
    
    migration_data_size = st.session_state.migration_params.get('data_size_gb', 1000)
    
    st.markdown("### 🤖 Storage Auto-Calculator")
    st.info(f"Base migration data size: {migration_data_size:,} GB")
    
    # Environment multipliers
    st.markdown("#### Environment Size Multipliers")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prod_mult = st.slider("Production", 0.8, 2.0, 1.2, 0.1, key="prod_mult")
    with col2:
        stage_mult = st.slider("Staging", 0.5, 1.5, 0.8, 0.1, key="stage_mult")
    with col3:
        test_mult = st.slider("Testing/QA", 0.3, 1.0, 0.6, 0.1, key="test_mult")
    with col4:
        dev_mult = st.slider("Development", 0.1, 0.8, 0.3, 0.1, key="dev_mult")
    
    # Calculate and show recommendations
    st.markdown("#### 📊 Recommended Storage Sizes")
    
    recommendations = {
        'Production': int(migration_data_size * prod_mult),
        'Staging': int(migration_data_size * stage_mult),
        'Testing': int(migration_data_size * test_mult),
        'QA': int(migration_data_size * test_mult),
        'Development': int(migration_data_size * dev_mult)
    }
    
    rec_data = []
    for env_type, recommended_gb in recommendations.items():
        growth_buffer = int(recommended_gb * 0.3)  # 30% growth buffer
        final_size = recommended_gb + growth_buffer
        
        rec_data.append({
            'Environment Type': env_type,
            'Base Size (GB)': f"{recommended_gb:,}",
            'Growth Buffer (GB)': f"{growth_buffer:,}",
            'Recommended Total (GB)': f"{final_size:,}"
        })
    
    rec_df = pd.DataFrame(rec_data)
    st.dataframe(rec_df, use_container_width=True)
    
    # Option to apply to current environments
    if st.session_state.environment_specs:
        st.markdown("#### ⚙️ Apply to Current Environments")
        
        if st.button("🎯 Auto-Calculate Storage for Current Environments"):
            updated_specs = {}
            
            for env_name, specs in st.session_state.environment_specs.items():
                env_type = specs.get('environment_type', env_name).lower()
                
                # Match environment type to multiplier
                if 'prod' in env_type:
                    multiplier = prod_mult
                elif 'stag' in env_type:
                    multiplier = stage_mult
                elif any(x in env_type for x in ['test', 'qa', 'uat']):
                    multiplier = test_mult
                elif 'dev' in env_type:
                    multiplier = dev_mult
                else:
                    multiplier = 0.5  # Default
                
                # Calculate new storage
                base_storage = int(migration_data_size * multiplier)
                recommended_storage = int(base_storage * 1.3)  # Add 30% buffer
                
                # Update specs
                updated_specs[env_name] = {
                    **specs,  # Keep existing configuration
                    'storage_gb': recommended_storage,
                    'original_storage_gb': specs.get('storage_gb', 0),
                    'auto_calculated': True
                }
            
            # Apply changes
            st.session_state.environment_specs = updated_specs
            st.success("✅ Storage auto-calculated and applied!")
            st.balloons()
            
            # Show what changed
            st.markdown("**Changes Applied:**")
            for env_name, specs in updated_specs.items():
                old_size = specs.get('original_storage_gb', 0)
                new_size = specs.get('storage_gb', 0)
                if old_size != new_size:
                    st.write(f"• {env_name}: {old_size:,} GB → {new_size:,} GB")

def show_storage_validation_widget():
    """Show storage validation widget in environment setup"""
    
    # Early return if required data is missing
    if not st.session_state.migration_params or not st.session_state.environment_specs:
        return
    
    # Get migration data size
    migration_data_gb = st.session_state.migration_params.get('data_size_gb', 0)
    storage_manager = StorageConfigurationManager()
    
    st.markdown("#### 🔍 Storage Configuration Validation")
    
    # Validate storage configuration
    validation = storage_manager.validate_storage_configuration(
        migration_data_gb, 
        st.session_state.environment_specs
    )
    
    # Status indicator section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_icon = "✅" if validation['is_valid'] else "❌"
        status_text = 'Valid' if validation['is_valid'] else 'Invalid'
        st.metric("Configuration Status", f"{status_icon} {status_text}")
    
    with col2:
        total_storage = validation['total_env_storage']
        st.metric("Total Environment Storage", f"{total_storage:,} GB")
    
    with col3:
        if validation['migration_to_env_ratio'] > 0:
            ratio = validation['migration_to_env_ratio']
            st.metric("Storage Ratio", f"{ratio:.1f}x")
    
    # Display errors if any
    if validation['errors']:
        st.error("🚨 **Critical Issues:**")
        for error in validation['errors']:
            st.error(f"• {error}")
    
    # Display warnings if any
    if validation['warnings']:
        st.warning("⚠️ **Warnings:**")
        for warning in validation['warnings']:
            st.warning(f"• {warning}")
    
    # Display recommendations if any
    if validation['recommendations']:
        st.info("💡 **Recommendations:**")
        for rec in validation['recommendations']:
            st.info(f"• {rec}")
    
    # Environment-specific analysis section
    if validation['environment_analysis']:
        with st.expander("📊 Environment Analysis Details"):
            analysis_data = []
            
            # Status icon mapping
            status_icons = {
                'good': '✅',
                'warning': '⚠️',
                'critical': '🚨',
                'over_provisioned': '💰'
            }
            
            # Process each environment analysis
            for env_name, analysis in validation['environment_analysis'].items():
                status_icon = status_icons.get(analysis['status'], '❓')
                
                analysis_data.append({
                    'Environment': env_name,
                    'Current (GB)': f"{analysis['current_storage_gb']:,}",
                    'Recommended (GB)': f"{analysis['recommended_storage_gb']:,}",
                    'Difference': f"{analysis['difference_gb']:+,} GB ({analysis['difference_percent']:+.1f}%)",
                    'Status': f"{status_icon} {analysis['status'].title()}"
                })
            
            # Create and display dataframe
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
            else:
                st.info("No environment analysis data available.")

class StorageConfigurationManager:
    """Centralized storage configuration management"""
    
    def __init__(self):
        self.default_environment_multipliers = {
            'production': 1.5,      # Production needs more storage for safety
            'staging': 1.0,         # Staging matches production data
            'testing': 0.7,         # Testing with subset of data
            'qa': 0.6,              # QA with sample data  
            'development': 0.3      # Development with minimal data
        }
        self.growth_buffer = 0.3    # 30% growth buffer
        self.replication_overhead = 0.1  # 10% for transaction logs, backups
    
    def calculate_recommended_storage(self, migration_data_gb: int, environment_type: str, 
                                    environment_count: int = 1) -> Dict:
        """Calculate recommended storage for an environment"""
        
        env_type = environment_type.lower()
        base_multiplier = self.default_environment_multipliers.get(env_type, 1.0)
        
        # Base storage requirement
        base_storage = int(migration_data_gb * base_multiplier)
        
        # Add growth buffer
        growth_storage = int(base_storage * self.growth_buffer)
        
        # Add replication overhead for writer/reader setups
        replication_storage = int(base_storage * self.replication_overhead)
        
        # Total recommended storage
        total_recommended = base_storage + growth_storage + replication_storage
        
        return {
            'base_storage_gb': base_storage,
            'growth_buffer_gb': growth_storage,
            'replication_overhead_gb': replication_storage,
            'total_recommended_gb': total_recommended,
            'environment_type': environment_type,
            'multiplier_used': base_multiplier
        }
    
    def validate_storage_configuration(self, migration_data_gb: int, 
                                     environment_specs: Dict) -> Dict:
        """Validate storage configuration across migration and environments"""
        
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'total_env_storage': 0,
            'migration_to_env_ratio': 0,
            'environment_analysis': {}
        }
        
        if not environment_specs:
            validation_result['errors'].append("No environment specifications found")
            validation_result['is_valid'] = False
            return validation_result
        
        total_env_storage = 0
        
        for env_name, specs in environment_specs.items():
            env_storage = specs.get('storage_gb', 0)
            env_type = specs.get('environment_type', env_name).lower()
            
            total_env_storage += env_storage
            
            # Calculate recommended storage for this environment
            recommended = self.calculate_recommended_storage(
                migration_data_gb, env_type
            )
            
            # Analysis for this environment
            env_analysis = {
                'current_storage_gb': env_storage,
                'recommended_storage_gb': recommended['total_recommended_gb'],
                'difference_gb': env_storage - recommended['total_recommended_gb'],
                'difference_percent': ((env_storage / recommended['total_recommended_gb']) - 1) * 100 if recommended['total_recommended_gb'] > 0 else 0,
                'status': 'good'
            }
            
            # Determine status and generate warnings
            if env_storage < recommended['base_storage_gb']:
                env_analysis['status'] = 'critical'
                validation_result['errors'].append(
                    f"{env_name}: Storage ({env_storage:,} GB) is below minimum recommended ({recommended['base_storage_gb']:,} GB)"
                )
                validation_result['is_valid'] = False
            elif env_storage < recommended['total_recommended_gb'] * 0.9:
                env_analysis['status'] = 'warning'
                validation_result['warnings'].append(
                    f"{env_name}: Storage ({env_storage:,} GB) is below recommended ({recommended['total_recommended_gb']:,} GB)"
                )
            elif env_storage > recommended['total_recommended_gb'] * 2:
                env_analysis['status'] = 'over_provisioned'
                validation_result['warnings'].append(
                    f"{env_name}: Storage ({env_storage:,} GB) is significantly over-provisioned (recommended: {recommended['total_recommended_gb']:,} GB)"
                )
            
            validation_result['environment_analysis'][env_name] = env_analysis
        
        validation_result['total_env_storage'] = total_env_storage
        
        if migration_data_gb > 0:
            validation_result['migration_to_env_ratio'] = total_env_storage / migration_data_gb
            
            # Overall validation
            if validation_result['migration_to_env_ratio'] < 0.8:
                validation_result['errors'].append(
                    f"Total environment storage ({total_env_storage:,} GB) is significantly less than migration data ({migration_data_gb:,} GB)"
                )
                validation_result['is_valid'] = False
            elif validation_result['migration_to_env_ratio'] > 5:
                validation_result['warnings'].append(
                    f"Total environment storage ({total_env_storage:,} GB) is much larger than migration data ({migration_data_gb:,} GB). Consider optimizing."
                )
        
        # Generate recommendations
        if validation_result['warnings'] or validation_result['errors']:
            validation_result['recommendations'].append("Use the Storage Auto-Calculator to optimize storage allocation")
            validation_result['recommendations'].append("Consider environment-specific data requirements")
            validation_result['recommendations'].append("Plan for 30% growth buffer in production environments")
        
        return validation_result

def is_enhanced_environment_data(environment_specs):
    """Check if environment specs contain enhanced cluster data"""
    if not environment_specs:
        return False
    
    sample_spec = next(iter(environment_specs.values()))
    enhanced_fields = ['workload_pattern', 'read_write_ratio', 'multi_az_writer']
    
    return any(field in sample_spec for field in enhanced_fields)

def show_enhanced_environment_setup_with_cluster_config():
    """Enhanced environment setup with Writer/Reader configuration"""
    st.markdown("## 📊 Database Cluster Configuration")
    
    # Add storage auto-calculator at the top
    if st.session_state.migration_params:
        with st.expander("🤖 Storage Auto-Calculator (Optional)", expanded=False):
            show_storage_auto_calculator()
        st.markdown("---")  # Add a separator
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Configuration method selection
    config_method = st.radio(
        "Choose configuration method:",
        [
            "📝 Manual Cluster Configuration", 
            "📁 Bulk Upload with Cluster Details",
            "🔄 Simple Configuration (Legacy)"
        ],
        horizontal=True
    )
    
    if config_method == "📝 Manual Cluster Configuration":
        show_manual_cluster_configuration()
    elif config_method == "📁 Bulk Upload with Cluster Details":
        show_bulk_cluster_upload()
    else:
        show_simple_configuration()
    
    # Add storage validation widget at the end
    if st.session_state.migration_params and st.session_state.environment_specs:
        st.markdown("---")  # Add a separator
        show_storage_validation_widget()

def show_manual_cluster_configuration():
    """Show manual cluster configuration with Writer/Reader options"""
    
    st.markdown("### 📝 Database Cluster Configuration")
    
    # Number of environments
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    for i in range(num_environments):
        with st.expander(f"🏢 Environment {i+1} - Cluster Configuration", expanded=i == 0):
            
            # Basic environment info
            col1, col2 = st.columns(2)
            
            with col1:
                env_name = st.text_input(
                    "Environment Name",
                    value=default_names[i] if i < len(default_names) else f"Environment_{i+1}",
                    key=f"env_name_{i}"
                )
                
                environment_type = st.selectbox(
                    "Environment Type",
                    ["Production", "Staging", "Testing", "Development"],
                    index=min(i, 3),
                    key=f"env_type_{i}"
                )
            
            with col2:
                workload_pattern = st.selectbox(
                    "Workload Pattern",
                    ["balanced", "read_heavy", "write_heavy", "analytics"],
                    key=f"workload_{i}"
                )
                
                read_write_ratio = st.slider(
                    "Read/Write Ratio (% Reads)",
                    min_value=10, max_value=95, value=70,
                    key=f"read_ratio_{i}"
                )
            
            # Infrastructure configuration
            st.markdown("#### 💻 Infrastructure Requirements")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
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
            
            with col2:
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"storage_{i}"
                )
                
                iops_requirement = st.number_input(
                    "IOPS Requirement",
                    min_value=100, max_value=50000,
                    value=[1000, 3000, 5000, 10000][min(i, 3)],
                    key=f"iops_{i}"
                )
            
            with col3:
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"connections_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"usage_{i}"
                )
            
            # Cluster configuration
            st.markdown("#### 🔗 Cluster Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                multi_az_writer = st.checkbox(
                    "Multi-AZ for Writer",
                    value=environment_type in ["Production", "Staging"],
                    key=f"multi_az_writer_{i}",
                    help="Deploy writer instance across multiple Availability Zones for high availability"
                )
                
                custom_reader_count = st.checkbox(
                    "Custom Reader Count",
                    value=False,
                    key=f"custom_readers_{i}"
                )
                
                if custom_reader_count:
                    num_readers = st.number_input(
                        "Number of Read Replicas",
                        min_value=0, max_value=5,
                        value=1 if environment_type == "Production" else 0,
                        key=f"num_readers_{i}"
                    )
                else:
                    # Auto-calculate based on environment and workload
                    cluster_config = DatabaseClusterConfiguration()
                    num_readers = cluster_config.calculate_optimal_readers(
                        environment_type.lower(), workload_pattern, peak_connections
                    )
                    st.info(f"Recommended readers: {num_readers} (auto-calculated)")
            
            with col2:
                multi_az_readers = st.checkbox(
                    "Multi-AZ for Readers",
                    value=environment_type == "Production",
                    key=f"multi_az_readers_{i}",
                    help="Deploy read replicas across multiple Availability Zones"
                )
                
                if num_readers > 0:
                    custom_reader_instance = st.checkbox(
                        "Custom Reader Instance Size",
                        value=False,
                        key=f"custom_reader_instance_{i}"
                    )
                    
                    if custom_reader_instance:
                        reader_instance_override = st.selectbox(
                            "Reader Instance Class",
                            ["db.t3.medium", "db.t3.large", "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge"],
                            key=f"reader_instance_{i}"
                        )
                    else:
                        reader_instance_override = None
                        st.info("Reader size will be auto-calculated based on writer size and workload")
                else:
                    reader_instance_override = None
            
            # Storage configuration
            st.markdown("#### 💾 Storage Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                storage_encrypted = st.checkbox(
                    "Encryption at Rest",
                    value=environment_type in ["Production", "Staging"],
                    key=f"encryption_{i}"
                )
            
            with col2:
                backup_retention = st.number_input(
                    "Backup Retention (Days)",
                    min_value=1, max_value=35,
                    value=30 if environment_type == "Production" else 7,
                    key=f"backup_{i}"
                )
            
            with col3:
                auto_storage_scaling = st.checkbox(
                    "Auto Storage Scaling",
                    value=True,
                    key=f"auto_scaling_{i}",
                    help="Automatically scale storage when needed"
                )

            # Store environment configuration
            environment_specs[env_name] = {
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'iops_requirement': iops_requirement,
                'peak_connections': peak_connections,
                'daily_usage_hours': daily_usage_hours,
                'workload_pattern': workload_pattern,
                'read_write_ratio': read_write_ratio,
                'environment_type': environment_type.lower(),
                'multi_az_writer': multi_az_writer,
                'multi_az_readers': multi_az_readers,
                'num_readers': num_readers if custom_reader_count else None,
                'reader_instance_override': reader_instance_override,
                'storage_encrypted': storage_encrypted,
                'backup_retention': backup_retention,
                'auto_storage_scaling': auto_storage_scaling
            }

    if st.button("💾 Save Cluster Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Cluster configuration saved!")
        
        # Display summary
        st.markdown("#### 📊 Configuration Summary")
        summary_data = []
        for env_name, specs in environment_specs.items():
            summary_data.append({
                'Environment': env_name,
                'Type': specs['environment_type'].title(),
                'Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
                'Storage': f"{specs['storage_gb']} GB ({specs['iops_requirement']} IOPS)",
                'Workload': f"{specs['workload_pattern']} ({specs['read_write_ratio']}% reads)",
                'Writer Multi-AZ': '✅' if specs['multi_az_writer'] else '❌',
                'Readers': f"{specs.get('num_readers', 'Auto')} ({'Multi-AZ' if specs['multi_az_readers'] else 'Single-AZ'})"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Show storage validation
        if st.session_state.migration_params:
            validate_storage_consistency()
        
        # Option to run immediate analysis
        st.markdown("---")
if st.button("🚀 Analyze Cluster Configuration", type="secondary", use_container_width=True):
    with st.spinner("🔄 Analyzing cluster requirements..."):
        try:
            # Check if we have enhanced data
            is_enhanced = is_enhanced_environment_data(environment_specs)
            
            if is_enhanced:
                # Use enhanced analyzer
                analyzer = EnhancedMigrationAnalyzer()
                recommendations = analyzer.calculate_enhanced_instance_recommendations(environment_specs)
                st.session_state.enhanced_recommendations = recommendations
                
                # Calculate enhanced costs
                if st.session_state.migration_params:
                    cost_analysis = analyzer.calculate_enhanced_migration_costs(
                        recommendations, 
                        st.session_state.migration_params
                    )
                    st.session_state.enhanced_analysis_results = cost_analysis
                
                st.success("✅ Enhanced cluster analysis complete!")
                show_cluster_configuration_preview(recommendations)
            else:
                # CORRECTED INDENTATION STARTS HERE
                # Use standard analyzer
                analyzer = MigrationAnalyzer()
                recommendations = analyzer.calculate_instance_recommendations(environment_specs)
                st.session_state.recommendations = recommendations
                
                st.success("✅ Standard analysis complete!")
                st.info("💡 For enhanced cluster analysis, use the cluster configuration options above.")
                # CORRECTED INDENTATION ENDS HERE
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.code(str(e))

class DatabaseClusterConfiguration:
    """Enhanced database cluster configuration with Writer/Reader support"""
    
    @staticmethod
    def calculate_optimal_readers(environment_type: str, workload_pattern: str, connections: int) -> int:
        """Calculate optimal number of read replicas"""
        
        base_readers = {
            'production': 2,
            'staging': 1,
            'testing': 1,
            'development': 0
        }
        
        # Adjust based on workload pattern
        workload_multipliers = {
            'read_heavy': 1.5,
            'balanced': 1.0,
            'write_heavy': 0.5,
            'analytics': 2.0
        }
        
        # Adjust based on connection count
        if connections > 1000:
            connection_factor = 1.5
        elif connections > 500:
            connection_factor = 1.2
        else:
            connection_factor = 1.0
        
        optimal_readers = int(
            base_readers.get(environment_type, 1) * 
            workload_multipliers.get(workload_pattern, 1.0) * 
            connection_factor
        )
        
        return max(0, min(optimal_readers, 5))  # Cap at 5 readers
    
    @staticmethod
    def recommend_reader_instance_size(writer_instance: str, read_ratio: float = 0.7) -> str:
        """Recommend reader instance size based on writer and read ratio"""
        
        # Instance sizing hierarchy
        instance_hierarchy = [
            'db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.t3.xlarge',
            'db.r5.large', 'db.r5.xlarge', 'db.r5.2xlarge', 'db.r5.4xlarge', 'db.r5.8xlarge'
        ]
        
        try:
            writer_index = instance_hierarchy.index(writer_instance)
            
            # For heavy read workloads, readers might need to be same size or larger
            if read_ratio > 0.8:
                reader_index = writer_index
            elif read_ratio > 0.5:
                reader_index = max(0, writer_index - 1)
            else:
                reader_index = max(0, writer_index - 2)
            
            return instance_hierarchy[reader_index]
            
        except ValueError:
            # If writer instance not in hierarchy, default to r5.large
            return 'db.r5.large'

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
        
        # Check if engines are in the same family
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
        
        # Base complexity from source engine
        source_complexity = cls.ENGINES.get(source_engine, {}).get('complexity_multiplier', 1.0)
        target_complexity = cls.ENGINES.get(target_engine, {}).get('complexity_multiplier', 1.0)
        
        # Additional complexity for heterogeneous migrations
        migration_type = cls.get_migration_type(source_engine, target_engine)
        
        if migration_type == "heterogeneous":
            if 'oracle' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.5  # Oracle to PostgreSQL is complex
            elif 'sql-server' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.0  # SQL Server to PostgreSQL
            else:
                return source_complexity * 1.5  # Other heterogeneous migrations
        else:
            return max(source_complexity, target_complexity)
    
    @classmethod
    def get_supported_features(cls, engine: str) -> list:
        """Get supported features for an engine"""
        return cls.ENGINES.get(engine, {}).get('features', [])
    
    @classmethod
    def get_aws_targets(cls, source_engine: str) -> list:
        """Get available AWS target engines for a source engine"""
        return cls.ENGINES.get(source_engine, {}).get('aws_targets', [])

class MigrationAnalyzer:
    """Basic migration analyzer for standard environment configurations"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = EnhancedAWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key
           
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations for environments"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs.get('cpu_cores', 4)
            ram_gb = specs.get('ram_gb', 16)
            storage_gb = specs.get('storage_gb', 500)
            
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
                'peak_connections': specs.get('peak_connections', 100)
            }
        
        return recommendations
        
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs based on recommendations"""
        
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
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks  # t3.large instance
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000  # $8k per week
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        # Calculate contingency and total
        base_cost = migration_costs['dms_instance'] + migration_costs['data_transfer'] + migration_costs['professional_services']
        migration_costs['contingency'] = base_cost * 0.2
        migration_costs['total'] = base_cost + migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }
    
    def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate AI insights synchronously for Streamlit"""
        
        if not self.anthropic_api_key:
            return {
                'error': 'No API key provided',
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params),
                'success': False
            }
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            context = f"""
            You are an AWS database migration expert. Analyze this project:
            
            MIGRATION: {migration_params.get('source_engine')} → {migration_params.get('target_engine')}
            DATA SIZE: {migration_params.get('data_size_gb', 0):,} GB
            TIMELINE: {migration_params.get('migration_timeline_weeks', 0)} weeks
            MONTHLY COST: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
            MIGRATION COST: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}
            
            Provide specific insights for:
            1. Top 3 migration risks and how to mitigate them
            2. Cost optimization opportunities
            3. Timeline feasibility and recommendations
            4. Technical considerations for this specific database migration
            
            Be concise but actionable.
            """
            
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": context}]
            )
            
            return {
                'ai_analysis': message.content[0].text,
                'source': 'Claude AI (Real)',
                'model': 'claude-3-sonnet-20240229',
                'success': True
            }
            
        except ImportError:
            return {
                'error': 'Run: pip install anthropic', 
                'source': 'Library Error', 
                'success': False,
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params)
            }
        except Exception as e:
            return {
                'error': f'Claude AI failed: {str(e)}', 
                'source': 'API Error', 
                'success': False,
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params)
            }
    
    def _get_fallback_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Enhanced fallback insights when AI is unavailable"""
        
        monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        data_size = migration_params.get('data_size_gb', 0)
        team_size = migration_params.get('team_size', 5)
        source_engine = migration_params.get('source_engine', 'unknown')
        target_engine = migration_params.get('target_engine', 'unknown')
        
        # Generate more intelligent fallback based on data
        risk_level = "Medium"
        if timeline_weeks < 8 or data_size > 10000 or team_size < 3:
            risk_level = "High"
        elif timeline_weeks > 16 and data_size < 1000 and team_size >= 5:
            risk_level = "Low"
        
        cost_efficiency = "Good"
        if monthly_cost > 10000:
            cost_efficiency = "Review required - high monthly cost"
        elif monthly_cost < 1000:
            cost_efficiency = "Very efficient"
        
        return {
            'summary': f"Migration analysis complete. Monthly cost: ${monthly_cost:,.0f}. Risk level: {risk_level}.",
            'cost_optimization': f"Monthly AWS cost of ${monthly_cost:,.0f} appears {cost_efficiency.lower()} for this migration scale. Consider Reserved Instances for 30-40% savings on production workloads. Evaluate right-sizing opportunities post-migration.",
            'risk_assessment': f"Migration risk level: {risk_level}. Key risks include {source_engine} to {target_engine} compatibility, application dependencies, and data validation. Implement comprehensive testing strategy and maintain rollback procedures.",
            'migration_strategy': f"Recommended approach: Phased migration starting with non-production environments. Use AWS DMS for continuous replication with minimal downtime cutover. Timeline of {timeline_weeks} weeks allows for proper validation.",
            'timeline_analysis': f"Timeline allocation: 20% planning, 40% migration execution, 30% testing/validation, 10% go-live. With {data_size:,} GB of data, allocate sufficient time for initial load and ongoing synchronization.",
            'technical_recommendations': f"Technical focus areas: Schema conversion ({source_engine} → {target_engine}), stored procedure migration, index optimization, and connection string updates across {migration_params.get('num_applications', 'unknown')} applications.",
            'optimization_opportunities': "Post-migration opportunities: Instance right-sizing based on actual usage, storage optimization, backup strategy refinement, and performance monitoring implementation. Plan optimization review 30-60 days post-migration.",
            'recommendations': [
                "Start with development/QA environments for validation",
                "Implement comprehensive backup and rollback procedures",
                "Use AWS DMS for minimal downtime migration",
                "Plan for application connection string updates",
                "Establish performance baselines before migration",
                "Consider Reserved Instances for cost optimization",
                "Implement monitoring and alerting for new environment",
                "Schedule post-migration optimization review"
            ],
            'source': 'Enhanced Fallback Analysis'
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
    
    def _calculate_instance_class(self, cpu_cores: int, ram_gb: int, env_type: str) -> str:
        """Calculate appropriate instance class"""
        
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
        elif cpu_cores <= 64 and ram_gb <= 256:
            instance_class = 'db.r5.4xlarge'
        else:
            instance_class = 'db.r5.8xlarge'
        
        # Environment-specific adjustments
        if env_type == 'development' and 'r5' in instance_class:
            # Downsize for development
            downsized = {
                'db.r5.8xlarge': 'db.r5.4xlarge',
                'db.r5.4xlarge': 'db.r5.2xlarge',
                'db.r5.2xlarge': 'db.r5.xlarge',
                'db.r5.xlarge': 'db.r5.large',
                'db.r5.large': 'db.t3.large'
            }
            instance_class = downsized.get(instance_class, instance_class)
        
        return instance_class
    
    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate cost for a single environment"""
        
        # Get pricing
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )
        
        # Calculate monthly hours
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        # Instance cost
        instance_cost = pricing['hourly'] * monthly_hours
        
        # Storage cost
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        
        # Backup cost (estimate 20% of storage)
        backup_cost = storage_cost * 0.2
        
        # Total monthly cost
        total_monthly = instance_cost + storage_cost + backup_cost
        
        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly,
            'pricing_source': pricing.get('source', 'Unknown')
        }
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs"""
        
        use_direct_connect = migration_params.get('use_direct_connect', False)
        
        # Internet transfer
        internet_cost = data_size_gb * 0.09  # $0.09 per GB
        
        # Direct Connect transfer
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02  # $0.02 per GB
        else:
            dx_cost = internet_cost
        
        return {
            'internet': internet_cost,
            'direct_connect': dx_cost,
            'total': min(internet_cost, dx_cost)
        }

class EnhancedAWSPricingAPI:
    """Enhanced AWS Pricing API with Writer/Reader and Aurora support"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        
    def calculate_cluster_cost(self, region: str, engine: str, 
                          writer_instance: str, writer_multi_az: bool,
                          reader_instances: List[Tuple[str, bool]]) -> float:
        """Calculate total cost for writer and readers"""
        total_cost = 0
    
        # Writer cost
        writer_pricing = self.get_rds_pricing(region, engine, writer_instance, writer_multi_az)
        total_cost += writer_pricing['hourly']
    
        for reader_instance, multi_az in reader_instances:
            reader_pricing = self.get_rds_pricing(region, engine, reader_instance, multi_az)
            total_cost += reader_pricing['hourly']
    
        return total_cost
        
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get RDS pricing for specific instance with Multi-AZ support"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Enhanced pricing data with Multi-AZ and Aurora support
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.xlarge': {'hourly': 0.408, 'hourly_multi_az': 0.816, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.8xlarge': {'hourly': 3.84, 'hourly_multi_az': 7.68, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-postgresql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.8xlarge': {'hourly': 4.64, 'hourly_multi_az': 4.64, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.12xlarge': {'hourly': 6.96, 'hourly_multi_az': 6.96, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.16xlarge': {'hourly': 9.28, 'hourly_multi_az': 9.28, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.24xlarge': {'hourly': 13.92, 'hourly_multi_az': 13.92, 'storage_gb': 0.10, 'io_request': 0.20},
                },
                'oracle-ee': {
                    'db.t3.medium': {'hourly': 0.408, 'hourly_multi_az': 0.816, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 3.84, 'hourly_multi_az': 7.68, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 7.68, 'hourly_multi_az': 15.36, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'mysql': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-mysql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.8xlarge': {'hourly': 4.64, 'hourly_multi_az': 4.64, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.12xlarge': {'hourly': 6.96, 'hourly_multi_az': 6.96, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.16xlarge': {'hourly': 9.28, 'hourly_multi_az': 9.28, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.24xlarge': {'hourly': 13.92, 'hourly_multi_az': 13.92, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.medium': {'hourly': 0.082, 'hourly_multi_az': 0.082, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.large': {'hourly': 0.164, 'hourly_multi_az': 0.164, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.xlarge': {'hourly': 0.328, 'hourly_multi_az': 0.328, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.2xlarge': {'hourly': 0.656, 'hourly_multi_az': 0.656, 'storage_gb': 0.10, 'io_request': 0.20}
                }
            }
        }

    def calculate_cluster_cost(self, region: str, engine: str, 
                          writer_instance: str, writer_multi_az: bool,
                          reader_instances: List[Tuple[str, bool]]) -> float:
        """Calculate total cost for writer and readers"""
        total_cost = 0
    
        # Writer cost
        writer_pricing = self.get_rds_pricing(region, engine, writer_instance, writer_multi_az)
        total_cost += writer_pricing['hourly']
    
        for reader_instance, multi_az in reader_instances:
            reader_pricing = self.get_rds_pricing(region, engine, reader_instance, multi_az)
            total_cost += reader_pricing['hourly']
    
        return total_cost
        
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get RDS pricing for specific instance with Multi-AZ support"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Enhanced pricing data with Multi-AZ and Aurora support
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.xlarge': {'hourly': 0.408, 'hourly_multi_az': 0.816, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.8xlarge': {'hourly': 3.84, 'hourly_multi_az': 7.68, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-postgresql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.8xlarge': {'hourly': 4.64, 'hourly_multi_az': 4.64, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.12xlarge': {'hourly': 6.96, 'hourly_multi_az': 6.96, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.16xlarge': {'hourly': 9.28, 'hourly_multi_az': 9.28, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.24xlarge': {'hourly': 13.92, 'hourly_multi_az': 13.92, 'storage_gb': 0.10, 'io_request': 0.20},
                },
                'oracle-ee': {
                    'db.t3.medium': {'hourly': 0.408, 'hourly_multi_az': 0.816, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 3.84, 'hourly_multi_az': 7.68, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 7.68, 'hourly_multi_az': 15.36, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'mysql': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-mysql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.8xlarge': {'hourly': 4.64, 'hourly_multi_az': 4.64, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.medium': {'hourly': 0.082, 'hourly_multi_az': 0.082, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.large': {'hourly': 0.164, 'hourly_multi_az': 0.164, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.xlarge': {'hourly': 0.328, 'hourly_multi_az': 0.328, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.t3.2xlarge': {'hourly': 0.656, 'hourly_multi_az': 0.656, 'storage_gb': 0.10, 'io_request': 0.20}
                }
            }
        }
        
        engine_pricing = pricing_data.get(region, {}).get(engine, {})
        instance_pricing = engine_pricing.get(instance_class, {
            'hourly': 0.5, 
            'hourly_multi_az': 1.0, 
            'storage_gb': 0.115, 
            'iops_gb': 0.10,
            'io_request': 0.20
        })
        
        # Select appropriate pricing based on Multi-AZ
        if multi_az and 'aurora' not in engine:
            hourly_cost = instance_pricing['hourly_multi_az']
        else:
            hourly_cost = instance_pricing['hourly']
        
        result = {
            'hourly': hourly_cost,
            'storage_gb': instance_pricing['storage_gb'],
            'iops_gb': instance_pricing.get('iops_gb', 0.10),
            'io_request': instance_pricing.get('io_request', 0.20),
            'is_aurora': 'aurora' in engine,
            'multi_az': multi_az
        }
        
        self.cache[cache_key] = result
        return result


# ALSO ADD this if MigrationAnalyzer class is missing:

class MigrationAnalyzer:
    """Basic migration analyzer for standard environment configurations"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = EnhancedAWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key
           
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations for environments"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            
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
                'peak_connections': specs.get('peak_connections', 100)
            }
        
        return recommendations
        
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs based on recommendations"""
        
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
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks  # t3.large instance
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000  # $8k per week
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        # Calculate contingency and total
        base_cost = migration_costs['dms_instance'] + migration_costs['data_transfer'] + migration_costs['professional_services']
        migration_costs['contingency'] = base_cost * 0.2
        migration_costs['total'] = base_cost + migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }

    def _calculate_reader_count(self, read_ratio: int, workload_pattern: str, env_type: str) -> int:
        """Calculate number of read replicas needed"""
        if env_type != 'production':
            return 0  # Only production gets readers
            
        if workload_pattern == 'read_heavy' and read_ratio >= 70:
            return 2
        elif workload_pattern == 'read_heavy' and read_ratio >= 50:
            return 1
        elif workload_pattern == 'mixed' and read_ratio >= 60:
            return 1
        return 0
    
    def _calculate_reader_instance_class(self, writer_instance: str, env_type: str) -> str:
        """Determine appropriate reader instance class"""
        if env_type != 'production':
            return "N/A"
        
        # Readers typically match writer class or one size smaller
        instance_map = {
            'db.r5.8xlarge': 'db.r5.4xlarge',
            'db.r5.4xlarge': 'db.r5.2xlarge',
            'db.r5.2xlarge': 'db.r5.xlarge',
            'db.r5.xlarge': 'db.r5.large',
            'db.r5.large': 'db.t3.large',
            'db.t3.large': 'db.t3.medium'
        }
        return instance_map.get(writer_instance, writer_instance)

    async def generate_ai_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate AI insights (optional, requires API key)"""
        
        if not self.anthropic_api_key:
            return {'error': 'No API key provided'}
        
        try:
            # Simple insights without actual API call for now
            insights = {
                'cost_efficiency': f"Monthly AWS cost of ${cost_analysis['monthly_aws_cost']:,.0f} represents efficient cloud migration",
                'migration_strategy': f"Recommended {migration_params.get('migration_timeline_weeks', 12)}-week timeline is appropriate for this scale",
                'risk_assessment': "Migration complexity is manageable with proper planning and resources"
            }
            return insights
        except Exception as e:
            return {'error': str(e)}
    
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
    
    def _calculate_instance_class(self, cpu_cores: int, ram_gb: int, env_type: str) -> str:
        """Calculate appropriate instance class"""
        
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
        elif cpu_cores <= 64 and ram_gb <= 256:
            instance_class = 'db.r5.4xlarge'
        else:
            instance_class = 'db.r5.8xlarge'
        
        # Environment-specific adjustments
        if env_type == 'development' and 'r5' in instance_class:
            # Downsize for development
            downsized = {
                'db.r5.8xlarge': 'db.r5.4xlarge',
                'db.r5.4xlarge': 'db.r5.2xlarge',
                'db.r5.2xlarge': 'db.r5.xlarge',
                'db.r5.xlarge': 'db.r5.large',
                'db.r5.large': 'db.t3.large'
            }
            instance_class = downsized.get(instance_class, instance_class)
        
        return instance_class
    
    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate cost for a single environment"""
        
        # Get pricing
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )
        
        # Calculate monthly hours
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        # Instance cost
        instance_cost = pricing['hourly'] * monthly_hours
        
        # Storage cost
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        
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
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs"""
        
        use_direct_connect = migration_params.get('use_direct_connect', False)
        
        # Internet transfer
        internet_cost = data_size_gb * 0.09  # $0.09 per GB
        
        # Direct Connect transfer
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02  # $0.02 per GB
        else:
            dx_cost = internet_cost
        
        return {
            'internet': internet_cost,
            'direct_connect': dx_cost,
            'total': min(internet_cost, dx_cost)
        }

class VRopsMetricsAnalyzer:
    def __init__(self):
        # Initialize your analyzer
        pass
    
    def analyze_metrics(self, metrics_data):
        # Your analysis logic here
        analysis_results = {}
        
        for env_name, metrics in metrics_data.items():
            # CPU Analysis
            cpu_analysis = self._analyze_cpu_metrics(metrics)
            
            # Memory Analysis
            memory_analysis = self._analyze_memory_metrics(metrics)
            
            # Storage Analysis
            storage_analysis = self._analyze_storage_metrics(metrics)
            
            # Generate instance recommendations
            instance_recommendations = self._generate_instance_recommendations(
                cpu_analysis, memory_analysis, storage_analysis
            )
            
            # Calculate performance scores
            performance_scores = self._calculate_performance_scores(
                cpu_analysis, memory_analysis, storage_analysis
            )
            
            analysis_results[env_name] = {
                'cpu_analysis': cpu_analysis,
                'memory_analysis': memory_analysis,
                'storage_analysis': storage_analysis,
                'instance_recommendations': instance_recommendations,
                'performance_scores': performance_scores
            }
        
        return analysis_results
    
    def _analyze_cpu_metrics(self, metrics):
        """Analyze CPU metrics and provide recommendations"""
        max_cpu = metrics.get('max_cpu_usage_percent', 0)
        avg_cpu = metrics.get('avg_cpu_usage_percent', 0)
        cpu_cores = metrics.get('cpu_cores_allocated', 1)
        cpu_ready = metrics.get('cpu_ready_time_ms', 0)
        
        # Determine scaling recommendation
        if max_cpu > 90:
            scaling_recommendation = "Scale up immediately - CPU over-utilized"
            required_capacity = 150
        elif max_cpu > 80:
            scaling_recommendation = "Scale up - approaching CPU limits"
            required_capacity = 130
        elif max_cpu > 70:
            scaling_recommendation = "Monitor closely - adequate capacity"
            required_capacity = 110
        elif max_cpu < 30:
            scaling_recommendation = "Consider scaling down - under-utilized"
            required_capacity = 80
        else:
            scaling_recommendation = "Current capacity is appropriate"
            required_capacity = 100
        
        return {
            'max_usage_percent': max_cpu,
            'avg_usage_percent': avg_cpu,
            'cpu_cores_allocated': cpu_cores,
            'cpu_ready_time_ms': cpu_ready,
            'scaling_recommendation': scaling_recommendation,
            'required_capacity_percent': required_capacity,
            'cpu_health_score': max(0, 100 - max_cpu)
        }
    
    def _analyze_memory_metrics(self, metrics):
        """Analyze memory metrics and provide recommendations"""
        max_mem = metrics.get('max_memory_usage_percent', 0)
        avg_mem = metrics.get('avg_memory_usage_percent', 0)
        mem_allocated = metrics.get('memory_allocated_gb', 1)
        mem_balloon = metrics.get('memory_balloon_gb', 0)
        
        # Determine scaling recommendation
        if max_mem > 95:
            scaling_recommendation = "Scale up immediately - Memory critically high"
            required_capacity = 150
        elif max_mem > 85:
            scaling_recommendation = "Scale up - Memory highly utilized"
            required_capacity = 130
        elif max_mem > 75:
            scaling_recommendation = "Monitor - Memory adequately utilized"
            required_capacity = 110
        elif max_mem < 40:
            scaling_recommendation = "Consider scaling down - Memory under-utilized"
            required_capacity = 80
        else:
            scaling_recommendation = "Current memory allocation is appropriate"
            required_capacity = 100
        
        # Check for memory pressure indicators
        memory_pressure = mem_balloon > 0
        
        return {
            'max_usage_percent': max_mem,
            'avg_usage_percent': avg_mem,
            'memory_allocated_gb': mem_allocated,
            'memory_balloon_gb': mem_balloon,
            'memory_pressure': memory_pressure,
            'scaling_recommendation': scaling_recommendation,
            'required_capacity_percent': required_capacity,
            'memory_health_score': max(0, 100 - max_mem)
        }
    
    def _analyze_storage_metrics(self, metrics):
        """Analyze storage metrics and provide recommendations"""
        max_iops = metrics.get('max_iops_total', 0)
        max_latency = metrics.get('max_disk_latency_ms', 0)
        storage_used = metrics.get('storage_used_gb', 0)
        storage_allocated = metrics.get('storage_allocated_gb', 1)
        
        # Calculate storage utilization
        storage_utilization = (storage_used / storage_allocated) * 100 if storage_allocated > 0 else 0
        
        # Determine IOPS and latency health
        iops_health = 100
        latency_health = 100
        
        if max_latency > 20:
            latency_health = 50
        elif max_latency > 10:
            latency_health = 75
        
        if max_iops > 10000:
            iops_health = 90
        elif max_iops > 5000:
            iops_health = 95
        
        # Storage recommendations
        if storage_utilization > 90:
            storage_recommendation = "Expand storage immediately"
        elif storage_utilization > 80:
            storage_recommendation = "Plan for storage expansion"
        elif storage_utilization < 30:
            storage_recommendation = "Consider storage optimization"
        else:
            storage_recommendation = "Storage utilization is healthy"
        
        return {
            'max_iops': max_iops,
            'max_latency_ms': max_latency,
            'storage_used_gb': storage_used,
            'storage_allocated_gb': storage_allocated,
            'storage_utilization_percent': storage_utilization,
            'storage_recommendation': storage_recommendation,
            'iops_health_score': iops_health,
            'latency_health_score': latency_health,
            'storage_health_score': max(0, 100 - storage_utilization)
        }
    
    def _generate_instance_recommendations(self, cpu_analysis, memory_analysis, storage_analysis):
        """Generate AWS instance recommendations based on analysis"""
        recommendations = []
        
        # Determine required resources
        cpu_requirement = cpu_analysis['cpu_cores_allocated'] * (cpu_analysis['required_capacity_percent'] / 100)
        memory_requirement = memory_analysis['memory_allocated_gb'] * (memory_analysis['required_capacity_percent'] / 100)
        
        # Instance recommendations based on requirements
        instance_options = [
            {'instance_type': 'db.t3.medium', 'vcpu': 2, 'memory_gb': 4, 'fit_score': 0},
            {'instance_type': 'db.t3.large', 'vcpu': 2, 'memory_gb': 8, 'fit_score': 0},
            {'instance_type': 'db.t3.xlarge', 'vcpu': 4, 'memory_gb': 16, 'fit_score': 0},
            {'instance_type': 'db.r5.large', 'vcpu': 2, 'memory_gb': 16, 'fit_score': 0},
            {'instance_type': 'db.r5.xlarge', 'vcpu': 4, 'memory_gb': 32, 'fit_score': 0},
            {'instance_type': 'db.r5.2xlarge', 'vcpu': 8, 'memory_gb': 64, 'fit_score': 0},
            {'instance_type': 'db.r5.4xlarge', 'vcpu': 16, 'memory_gb': 128, 'fit_score': 0}
        ]
        
        # Score each instance based on fit
        for instance in instance_options:
            cpu_fit = 100 - abs(instance['vcpu'] - cpu_requirement) * 10
            memory_fit = 100 - abs(instance['memory_gb'] - memory_requirement) * 2
            
            # Ensure instance meets minimum requirements
            if instance['vcpu'] >= cpu_requirement and instance['memory_gb'] >= memory_requirement:
                instance['fit_score'] = (cpu_fit + memory_fit) / 2
                
                # Generate recommendation reason
                if instance['fit_score'] > 80:
                    instance['recommendation_reason'] = "Excellent fit for current workload"
                elif instance['fit_score'] > 60:
                    instance['recommendation_reason'] = "Good fit with some overhead"
                elif instance['fit_score'] > 40:
                    instance['recommendation_reason'] = "Adequate but may be over-provisioned"
                else:
                    instance['recommendation_reason'] = "Meets requirements but not optimal"
                
                recommendations.append(instance)
        
        # Sort by fit score and return top 3
        recommendations.sort(key=lambda x: x['fit_score'], reverse=True)
        return recommendations[:3]
    
    def _calculate_performance_scores(self, cpu_analysis, memory_analysis, storage_analysis):
        """Calculate overall performance health scores"""
        cpu_health = cpu_analysis['cpu_health_score']
        memory_health = memory_analysis['memory_health_score']
        storage_health = storage_analysis['storage_health_score']
        
        # Calculate overall health as weighted average
        overall_health = (cpu_health * 0.4 + memory_health * 0.4 + storage_health * 0.2)
        
        return {
            'cpu_health': int(cpu_health),
            'memory_health': int(memory_health),
            'storage_health': int(storage_health),
            'overall_health': int(overall_health)
        }

@dataclass
class InstanceSpecs:
    """Instance specifications with performance metrics"""
    instance_class: str
    vcpu: int
    memory_gb: float
    network_performance: str
    ebs_bandwidth_mbps: int
    hourly_cost: float
    suitable_for: List[str]
    max_connections: int
    iops_capability: int

class OptimizedReaderWriterAnalyzer:
    """Advanced Reader/Writer optimization with intelligent recommendations"""
    
    def __init__(self):
        self.instance_specs = self._initialize_instance_specs()
        self.pricing_data = self._initialize_pricing_data()
        self.optimization_goal = 'balanced'
        self.consider_reserved_instances = True
        self.environment_priority = 'production_first'
        
    def _initialize_instance_specs(self) -> Dict[str, InstanceSpecs]:
        """Initialize comprehensive instance specifications"""
        return {
            # T3 Series - Burstable Performance
            'db.t3.micro': InstanceSpecs('db.t3.micro', 2, 1, 'Low to Moderate', 87, 0.0255, ['development', 'testing'], 87, 1000),
            'db.t3.small': InstanceSpecs('db.t3.small', 2, 2, 'Low to Moderate', 174, 0.051, ['development', 'testing'], 174, 2000),
            'db.t3.medium': InstanceSpecs('db.t3.medium', 2, 4, 'Low to Moderate', 347, 0.102, ['development', 'small_production'], 347, 3000),
            'db.t3.large': InstanceSpecs('db.t3.large', 2, 8, 'Low to Moderate', 695, 0.204, ['development', 'small_production'], 695, 4000),
            'db.t3.xlarge': InstanceSpecs('db.t3.xlarge', 4, 16, 'Low to Moderate', 695, 0.408, ['staging', 'medium_production'], 1000, 5000),
            'db.t3.2xlarge': InstanceSpecs('db.t3.2xlarge', 8, 32, 'Low to Moderate', 695, 0.816, ['staging', 'medium_production'], 1500, 6000),
            
            # R5 Series - Memory Optimized
            'db.r5.large': InstanceSpecs('db.r5.large', 2, 16, 'Up to 10 Gbps', 693, 0.24, ['memory_intensive', 'analytics'], 1000, 7500),
            'db.r5.xlarge': InstanceSpecs('db.r5.xlarge', 4, 32, 'Up to 10 Gbps', 1387, 0.48, ['memory_intensive', 'production'], 2000, 10000),
            'db.r5.2xlarge': InstanceSpecs('db.r5.2xlarge', 8, 64, 'Up to 10 Gbps', 2775, 0.96, ['large_production', 'analytics'], 3000, 15000),
            'db.r5.4xlarge': InstanceSpecs('db.r5.4xlarge', 16, 128, '10 Gbps', 4750, 1.92, ['large_production', 'high_memory'], 5000, 25000),
            'db.r5.8xlarge': InstanceSpecs('db.r5.8xlarge', 32, 256, '10 Gbps', 6800, 3.84, ['enterprise', 'high_performance'], 8000, 40000),
            'db.r5.12xlarge': InstanceSpecs('db.r5.12xlarge', 48, 384, '12 Gbps', 9500, 5.76, ['enterprise', 'very_high_memory'], 12000, 60000),
            'db.r5.16xlarge': InstanceSpecs('db.r5.16xlarge', 64, 512, '20 Gbps', 13600, 7.68, ['enterprise', 'extreme_performance'], 16000, 80000),
            'db.r5.24xlarge': InstanceSpecs('db.r5.24xlarge', 96, 768, '25 Gbps', 19000, 11.52, ['enterprise', 'maximum_performance'], 24000, 120000),
            
            # R6i Series - Latest Generation Memory Optimized
            'db.r6i.large': InstanceSpecs('db.r6i.large', 2, 16, 'Up to 12.5 Gbps', 1000, 0.252, ['memory_intensive', 'latest_gen'], 1200, 8000),
            'db.r6i.xlarge': InstanceSpecs('db.r6i.xlarge', 4, 32, 'Up to 12.5 Gbps', 2000, 0.504, ['production', 'latest_gen'], 2400, 12000),
            'db.r6i.2xlarge': InstanceSpecs('db.r6i.2xlarge', 8, 64, 'Up to 12.5 Gbps', 4000, 1.008, ['large_production', 'latest_gen'], 4000, 18000),
            'db.r6i.4xlarge': InstanceSpecs('db.r6i.4xlarge', 16, 128, '12.5 Gbps', 8000, 2.016, ['enterprise', 'latest_gen'], 6000, 30000),
            
            # C5 Series - Compute Optimized
            'db.c5.large': InstanceSpecs('db.c5.large', 2, 4, 'Up to 10 Gbps', 693, 0.192, ['compute_intensive', 'low_memory'], 1000, 5000),
            'db.c5.xlarge': InstanceSpecs('db.c5.xlarge', 4, 8, 'Up to 10 Gbps', 1387, 0.384, ['compute_intensive', 'oltp'], 2000, 8000),
            'db.c5.2xlarge': InstanceSpecs('db.c5.2xlarge', 8, 16, 'Up to 10 Gbps', 2775, 0.768, ['high_cpu', 'oltp'], 3000, 12000),
            'db.c5.4xlarge': InstanceSpecs('db.c5.4xlarge', 16, 32, '10 Gbps', 4750, 1.536, ['very_high_cpu', 'oltp'], 5000, 20000),
        }
    
    def _initialize_pricing_data(self) -> Dict:
        """Initialize comprehensive pricing data for different regions and deployment options"""
        return {
            'us-east-1': {
                'reserved_1_year': {'discount': 0.35, 'upfront_ratio': 0.0},
                'reserved_3_year': {'discount': 0.55, 'upfront_ratio': 0.0},
                'spot': {'discount': 0.70, 'availability': 0.85},
                'multi_az_multiplier': 2.0,
                'cross_az_transfer_gb': 0.01,
                'backup_storage_gb': 0.095,
                'snapshot_storage_gb': 0.095
            }
        }
    
    def set_optimization_preferences(self, goal='balanced', consider_reserved=True, environment_priority='production_first'):
        """Set optimization preferences"""
        self.optimization_goal = goal
        self.consider_reserved_instances = consider_reserved
        self.environment_priority = environment_priority
    
    def optimize_cluster_configuration(self, environment_specs: Dict) -> Dict:
        """Generate optimized Reader/Writer recommendations with detailed cost analysis"""
        
        optimized_recommendations = {}
        
        for env_name, specs in environment_specs.items():
            optimization = self._optimize_single_environment(env_name, specs)
            optimized_recommendations[env_name] = optimization
        
        return optimized_recommendations
    
    def _optimize_single_environment(self, env_name: str, specs: Dict) -> Dict:
        """Optimize configuration for a single environment"""
        
        # Extract environment characteristics
        cpu_cores = specs.get('cpu_cores', 4)
        ram_gb = specs.get('ram_gb', 16)
        storage_gb = specs.get('storage_gb', 500)
        iops_requirement = specs.get('iops_requirement', 3000)
        peak_connections = specs.get('peak_connections', 100)
        workload_pattern = specs.get('workload_pattern', 'balanced')
        read_write_ratio = specs.get('read_write_ratio', 70)
        environment_type = specs.get('environment_type', 'production')
        daily_usage_hours = specs.get('daily_usage_hours', 24)
        
        # Analyze workload characteristics
        workload_analysis = self._analyze_workload_characteristics(
            cpu_cores, ram_gb, iops_requirement, peak_connections, 
            workload_pattern, read_write_ratio
        )
        
        # Optimize Writer configuration
        writer_optimization = self._optimize_writer_instance(workload_analysis, environment_type)
        
        # Optimize Reader configuration
        reader_optimization = self._optimize_reader_configuration(
            writer_optimization, workload_analysis, environment_type
        )
        
        # Calculate comprehensive costs
        cost_analysis = self._calculate_comprehensive_costs(
            writer_optimization, reader_optimization, storage_gb, 
            iops_requirement, environment_type, daily_usage_hours
        )
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            workload_analysis, writer_optimization, reader_optimization, cost_analysis
        )
        
        return {
            'environment_name': env_name,
            'environment_type': environment_type,
            'workload_analysis': workload_analysis,
            'writer_optimization': writer_optimization,
            'reader_optimization': reader_optimization,
            'cost_analysis': cost_analysis,
            'recommendations': recommendations,
            'optimization_score': self._calculate_optimization_score(
                workload_analysis, writer_optimization, reader_optimization, cost_analysis
            )
        }
    
    def _analyze_workload_characteristics(self, cpu_cores: int, ram_gb: int, 
                                        iops_requirement: int, peak_connections: int,
                                        workload_pattern: str, read_write_ratio: int) -> Dict:
        """Analyze workload characteristics to determine optimal configuration"""
        
        # Calculate workload intensity
        cpu_intensity = min(100, (cpu_cores / 64) * 100)
        memory_intensity = min(100, (ram_gb / 768) * 100)
        io_intensity = min(100, (iops_requirement / 80000) * 100)
        connection_intensity = min(100, (peak_connections / 16000) * 100)
        
        # Determine workload type
        if workload_pattern == 'read_heavy' or read_write_ratio >= 80:
            workload_type = 'read_heavy'
            read_scaling_factor = 2.0
        elif workload_pattern == 'write_heavy' or read_write_ratio <= 30:
            workload_type = 'write_heavy'
            read_scaling_factor = 0.5
        elif workload_pattern == 'analytics':
            workload_type = 'analytics'
            read_scaling_factor = 3.0
        else:
            workload_type = 'balanced'
            read_scaling_factor = 1.2
        
        # Calculate complexity score
        complexity_score = (cpu_intensity + memory_intensity + io_intensity + connection_intensity) / 4
        
        # Calculate optimal reader count
        base_readers = {
            'read_heavy': 3,
            'analytics': 2,
            'balanced': 1,
            'write_heavy': 0
        }
        
        complexity_adjustment = int(complexity_score / 30)
        connection_adjustment = int(peak_connections / 2000)
        optimal_reader_count = base_readers.get(workload_type, 1) + complexity_adjustment + connection_adjustment
        optimal_reader_count = min(5, max(0, optimal_reader_count))
        
        # Determine performance requirements
        avg_intensity = (cpu_intensity + memory_intensity + io_intensity) / 3
        if avg_intensity >= 80:
            performance_requirements = 'high_performance'
        elif avg_intensity >= 60:
            performance_requirements = 'medium_performance'
        elif avg_intensity >= 40:
            performance_requirements = 'standard_performance'
        else:
            performance_requirements = 'basic_performance'
        
        return {
            'cpu_intensity': cpu_intensity,
            'memory_intensity': memory_intensity,
            'io_intensity': io_intensity,
            'connection_intensity': connection_intensity,
            'complexity_score': complexity_score,
            'workload_type': workload_type,
            'read_scaling_factor': read_scaling_factor,
            'recommended_reader_count': optimal_reader_count,
            'performance_requirements': performance_requirements
        }
    
    def _optimize_writer_instance(self, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Writer instance selection"""
        
        performance_req = workload_analysis['performance_requirements']
        complexity_score = workload_analysis['complexity_score']
        
        # Filter suitable instances
        suitable_instances = []
        
        for instance_class, specs in self.instance_specs.items():
            if self._is_instance_suitable_for_writer(specs, performance_req, environment_type, complexity_score):
                suitable_instances.append((instance_class, specs))
        
        # Score instances
        scored_instances = []
        for instance_class, specs in suitable_instances:
            score = self._score_writer_instance(specs, workload_analysis, environment_type)
            scored_instances.append((score, instance_class, specs))
        
        # Sort by score
        scored_instances.sort(reverse=True)
        
        # Get top recommendation
        if scored_instances:
            score, instance_class, specs = scored_instances[0]
            
            return {
                'instance_class': instance_class,
                'specs': specs,
                'score': score,
                'multi_az': environment_type in ['production', 'staging'],
                'reasoning': self._generate_writer_reasoning(specs, workload_analysis, environment_type),
                'monthly_cost': self._calculate_instance_monthly_cost(specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(specs, environment_type) * 12
            }
        else:
            # Fallback
            fallback_instance = 'db.r5.large'
            fallback_specs = self.instance_specs[fallback_instance]
            
            return {
                'instance_class': fallback_instance,
                'specs': fallback_specs,
                'score': 50.0,
                'multi_az': True,
                'reasoning': 'Fallback recommendation',
                'monthly_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type) * 12
            }
    
    def _is_instance_suitable_for_writer(self, specs: InstanceSpecs, performance_req: str, 
                                       environment_type: str, complexity_score: float) -> bool:
        """Check if instance is suitable for writer role"""
        
        performance_minimums = {
            'high_performance': {'min_vcpu': 16, 'min_memory': 64, 'min_iops': 20000},
            'medium_performance': {'min_vcpu': 8, 'min_memory': 32, 'min_iops': 10000},
            'standard_performance': {'min_vcpu': 4, 'min_memory': 16, 'min_iops': 5000},
            'basic_performance': {'min_vcpu': 2, 'min_memory': 8, 'min_iops': 2000}
        }
        
        minimums = performance_minimums.get(performance_req, performance_minimums['standard_performance'])
        
        if (specs.vcpu < minimums['min_vcpu'] or 
            specs.memory_gb < minimums['min_memory'] or 
            specs.iops_capability < minimums['min_iops']):
            return False
        
        if environment_type == 'production':
            if specs.vcpu < 4 or specs.memory_gb < 16:
                return False
            if 't3' in specs.instance_class and complexity_score > 50:
                return False
        
        return True
    
    def _score_writer_instance(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> float:
        """Score writer instance based on multiple criteria"""
        
        score = 0.0
        
        # Performance scoring (40% weight)
        performance_score = min(100, (specs.vcpu * 10 + specs.memory_gb / 4 + specs.iops_capability / 500))
        score += performance_score * 0.4
        
        # Cost efficiency scoring (30% weight)
        cost_per_vcpu = specs.hourly_cost / specs.vcpu
        cost_per_gb_memory = specs.hourly_cost / specs.memory_gb
        cost_efficiency = max(0, 100 - (cost_per_vcpu * 100 + cost_per_gb_memory * 10))
        score += cost_efficiency * 0.3
        
        # Suitability scoring (20% weight)
        suitability_score = 60
        if environment_type in specs.suitable_for:
            suitability_score = 100
        elif any(env in specs.suitable_for for env in ['production', 'large_production', 'enterprise']):
            suitability_score = 80
        score += suitability_score * 0.2
        
       # Network performance scoring (10% weight)
        network_score = 60  # Base score
        if '25 Gbps' in specs.network_performance:
            network_score = 100
        elif '20 Gbps' in specs.network_performance:
            network_score = 90
        elif '12' in specs.network_performance:
            network_score = 80
        elif '10 Gbps' in specs.network_performance:
            network_score = 70
        score += network_score * 0.1
        
        return min(100, score)
    
    def _optimize_reader_configuration(self, writer_optimization: Dict, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Reader configuration based on writer and workload"""
        
        recommended_count = workload_analysis['recommended_reader_count']
        
        if recommended_count == 0:
            return {
                'count': 0,
                'instance_class': None,
                'specs': None,
                'single_reader_monthly_cost': 0,
                'total_monthly_cost': 0,
                'total_annual_cost': 0,
                'multi_az': False,
                'reasoning': 'No read replicas needed for this workload pattern'
            }
        
        # Determine optimal reader instance size
        writer_specs = writer_optimization['specs']
        reader_instance_class = self._calculate_optimal_reader_size(writer_specs, workload_analysis, environment_type)
        reader_specs = self.instance_specs[reader_instance_class]
        
        # Calculate costs
        single_reader_monthly_cost = self._calculate_instance_monthly_cost(reader_specs, environment_type, is_reader=True)
        total_monthly_cost = single_reader_monthly_cost * recommended_count
        total_annual_cost = total_monthly_cost * 12
        
        return {
            'count': recommended_count,
            'instance_class': reader_instance_class,
            'specs': reader_specs,
            'single_reader_monthly_cost': single_reader_monthly_cost,
            'total_monthly_cost': total_monthly_cost,
            'total_annual_cost': total_annual_cost,
            'multi_az': environment_type == 'production',
            'reasoning': self._generate_reader_reasoning(recommended_count, reader_specs, workload_analysis)
        }
    
    def _calculate_optimal_reader_size(self, writer_specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
        """Calculate optimal reader instance size"""
        
        workload_type = workload_analysis['workload_type']
        read_scaling_factor = workload_analysis['read_scaling_factor']
        
        if workload_type == 'read_heavy' or workload_type == 'analytics':
            target_vcpu = int(writer_specs.vcpu * read_scaling_factor)
            target_memory = writer_specs.memory_gb * read_scaling_factor
        else:
            target_vcpu = max(2, int(writer_specs.vcpu * 0.7))
            target_memory = writer_specs.memory_gb * 0.7
        
        # Find best matching instance
        best_match = None
        best_score = 0
        
        for instance_class, specs in self.instance_specs.items():
            if specs.vcpu >= target_vcpu * 0.8 and specs.memory_gb >= target_memory * 0.8:
                size_match_score = 100 - abs(specs.vcpu - target_vcpu) * 5 - abs(specs.memory_gb - target_memory) * 2
                cost_efficiency_score = 100 - (specs.hourly_cost * 10)
                overall_score = size_match_score * 0.7 + cost_efficiency_score * 0.3
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = instance_class
        
        return best_match or 'db.r5.large'
    
    def _calculate_instance_monthly_cost(self, specs: InstanceSpecs, environment_type: str, is_reader: bool = False) -> float:
        """Calculate monthly cost for an instance"""
        
        base_hourly_cost = specs.hourly_cost
        
        # Apply Multi-AZ multiplier if needed
        if environment_type in ['production', 'staging'] and not is_reader:
            base_hourly_cost *= self.pricing_data['us-east-1']['multi_az_multiplier']
        
        # Calculate monthly cost (730 hours per month)
        monthly_cost = base_hourly_cost * 730
        
        return monthly_cost
    
    def _calculate_comprehensive_costs(self, writer_optimization: Dict, reader_optimization: Dict,
                                     storage_gb: int, iops_requirement: int, environment_type: str,
                                     daily_usage_hours: int) -> Dict:
        """Calculate comprehensive cost analysis"""
        
        # Instance costs
        writer_monthly_cost = writer_optimization['monthly_cost']
        reader_monthly_cost = reader_optimization['total_monthly_cost']
        total_instance_monthly_cost = writer_monthly_cost + reader_monthly_cost
        
        # Storage costs
        storage_costs = self._calculate_storage_costs(storage_gb, iops_requirement, environment_type)
        
        # Additional costs
        backup_monthly_cost = storage_gb * 0.095
        monitoring_monthly_cost = 50 if environment_type == 'production' else 20
        cross_az_monthly_cost = 0
        
        if reader_optimization['count'] > 0:
            estimated_cross_az_gb = storage_gb * 0.1
            cross_az_monthly_cost = estimated_cross_az_gb * 0.01
        
        total_monthly_cost = (total_instance_monthly_cost + storage_costs['total_monthly'] + 
                            backup_monthly_cost + monitoring_monthly_cost + cross_az_monthly_cost)
        
        # Reserved Instance calculations
        reserved_1_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 1)
        reserved_3_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 3)
        
        return {
            'monthly_breakdown': {
                'writer_instance': writer_monthly_cost,
                'reader_instances': reader_monthly_cost,
                'storage': storage_costs['total_monthly'],
                'backup': backup_monthly_cost,
                'monitoring': monitoring_monthly_cost,
                'cross_az_transfer': cross_az_monthly_cost,
                'total': total_monthly_cost
            },
            'annual_breakdown': {
                'writer_instance': writer_monthly_cost * 12,
                'reader_instances': reader_monthly_cost * 12,
                'storage': storage_costs['total_monthly'] * 12,
                'backup': backup_monthly_cost * 12,
                'monitoring': monitoring_monthly_cost * 12,
                'cross_az_transfer': cross_az_monthly_cost * 12,
                'total': total_monthly_cost * 12
            },
            'storage_details': storage_costs,
            'reserved_instance_options': {
                '1_year': reserved_1_year,
                '3_year': reserved_3_year
            },
            'cost_optimization_opportunities': []
        }
    
    def _calculate_storage_costs(self, storage_gb: int, iops_requirement: int, environment_type: str) -> Dict:
        """Calculate detailed storage costs"""
        
        if iops_requirement > 16000:
            storage_type = 'io2'
            base_cost_per_gb = 0.125
            iops_cost_per_iops = 0.065
        elif iops_requirement > 3000:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            additional_iops = max(0, iops_requirement - 3000)
            iops_cost_per_iops = 0.005 if additional_iops > 0 else 0
        else:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            iops_cost_per_iops = 0
            additional_iops = 0
        
        base_storage_cost = storage_gb * base_cost_per_gb
        iops_cost = (additional_iops if storage_type == 'gp3' else iops_requirement) * iops_cost_per_iops
        total_monthly_cost = base_storage_cost + iops_cost
        
        return {
            'storage_type': storage_type,
            'storage_gb': storage_gb,
            'iops_provisioned': iops_requirement,
            'base_storage_cost': base_storage_cost,
            'iops_cost': iops_cost,
            'total_monthly': total_monthly_cost,
            'cost_per_gb': base_cost_per_gb,
            'cost_per_iops': iops_cost_per_iops
        }
    
    def _calculate_reserved_instance_savings(self, monthly_instance_cost: float, years: int) -> Dict:
        """Calculate Reserved Instance savings"""
        
        if years == 1:
            discount = self.pricing_data['us-east-1']['reserved_1_year']['discount']
        else:
            discount = self.pricing_data['us-east-1']['reserved_3_year']['discount']
        
        annual_on_demand = monthly_instance_cost * 12
        annual_reserved = annual_on_demand * (1 - discount)
        total_on_demand = annual_on_demand * years
        total_reserved = annual_reserved * years
        total_savings = total_on_demand - total_reserved
        
        return {
            'term_years': years,
            'discount_percentage': discount * 100,
            'annual_cost': annual_reserved,
            'total_cost': total_reserved,
            'total_savings': total_savings,
            'monthly_cost': annual_reserved / 12
        }
    
    def _generate_optimization_recommendations(self, workload_analysis: Dict, writer_optimization: Dict,
                                             reader_optimization: Dict, cost_analysis: Dict) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        
        recommendations = []
        
        if workload_analysis['complexity_score'] > 80:
            recommendations.append("Consider upgrading to latest generation instances (R6i) for better price/performance")
        
        total_annual_cost = cost_analysis['annual_breakdown']['total']
        if total_annual_cost > 50000:
            recommendations.append("Evaluate Reserved Instances for significant cost savings on predictable workloads")
        
        if reader_optimization['count'] > 1:
            recommendations.append("Implement Aurora Auto Scaling to optimize reader count based on actual demand")
        
        recommendations.append("Set up Enhanced Monitoring and Performance Insights for optimization opportunities")
        recommendations.append("Configure CloudWatch alarms for CPU, memory, and connection metrics")
        
        return recommendations
    
    def _calculate_optimization_score(self, workload_analysis: Dict, writer_optimization: Dict,
                                    reader_optimization: Dict, cost_analysis: Dict) -> float:
        """Calculate overall optimization score"""
        
        performance_score = min(100, writer_optimization['score'])
        cost_per_vcpu = cost_analysis['monthly_breakdown']['total'] / writer_optimization['specs'].vcpu
        cost_efficiency_score = max(0, 100 - (cost_per_vcpu / 10))
        
        config_score = 80
        if reader_optimization['count'] > 0 and workload_analysis['workload_type'] in ['read_heavy', 'analytics']:
            config_score += 15
        elif reader_optimization['count'] == 0 and workload_analysis['workload_type'] == 'write_heavy':
            config_score += 15
        
        overall_score = (performance_score * 0.4 + cost_efficiency_score * 0.4 + config_score * 0.2)
        
        return min(100, overall_score)
    
    def _generate_writer_reasoning(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
        """Generate reasoning for writer instance recommendation"""
        
        reasoning_parts = []
        
        if workload_analysis['complexity_score'] > 70:
            reasoning_parts.append(f"High complexity workload requires {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
        else:
            reasoning_parts.append(f"Balanced configuration with {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
        
        if environment_type == 'production':
            reasoning_parts.append("Production-grade instance with high availability features")
        else:
            reasoning_parts.append(f"Cost-optimized for {environment_type} environment")
        
        if '25 Gbps' in specs.network_performance:
            reasoning_parts.append("Enhanced networking for high-throughput workloads")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_reader_reasoning(self, count: int, specs: InstanceSpecs, workload_analysis: Dict) -> str:
        """Generate reasoning for reader configuration"""
        
        if count == 0:
            return "No read replicas needed for write-heavy or low-complexity workloads"
        
        workload_type = workload_analysis['workload_type']
        
        reasoning_parts = [
            f"{count} read replica{'s' if count > 1 else ''} recommended for {workload_type} workload",
            f"Each reader: {specs.vcpu} vCPUs, {specs.memory_gb}GB memory"
        ]
        
        if workload_type == 'read_heavy':
            reasoning_parts.append("Multiple readers will distribute read load effectively")
        elif workload_type == 'analytics':
            reasoning_parts.append("Dedicated readers for analytical queries to avoid impact on primary")
        
        return ". ".join(reasoning_parts) + "."


# Integration with your existing Streamlit app
def integrate_with_existing_app():
    """Integration instructions for your existing app"""
    
    st.markdown("## 🔗 Integration Instructions")
    
    st.code("""
    # Add this to your show_enhanced_environment_setup_with_cluster_config() function:
    
    if st.button("🚀 Run Optimization Analysis", type="primary"):
        optimizer = OptimizedReaderWriterAnalyzer()
        optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
        st.session_state.optimization_results = optimization_results
        display_optimization_results(optimization_results)
    
    # Add this to your results dashboard:
    
    if hasattr(st.session_state, 'optimization_results'):
        display_optimization_results(st.session_state.optimization_results)
    """)


def show_optimized_recommendations():
    """Show optimized Reader/Writer recommendations"""
    
    st.markdown("## 🚀 AI-Optimized Reader/Writer Recommendations")
    
    if not st.session_state.environment_specs:
        st.warning("⚠️ Please configure environments first.")
        return
    
    # Configuration options
    st.markdown("### ⚙️ Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Cost Efficiency", "Performance", "Balanced"],
            index=2
        )
    
    with col2:
        consider_reserved_instances = st.checkbox(
            "Include Reserved Instance Analysis",
            value=True
        )
    
    with col3:
        environment_priority = st.selectbox(
            "Environment Priority",
            ["Production First", "All Equal", "Cost First"],
            index=0
        )
    
    # Run optimization button
    if st.button("🧠 Generate AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing workloads and optimizing configurations..."):
            
            # Initialize optimizer
            optimizer = OptimizedReaderWriterAnalyzer()
            
            # Apply optimization settings
            optimizer.set_optimization_preferences(
                goal=optimization_goal.lower().replace(" ", "_"),
                consider_reserved=consider_reserved_instances,
                environment_priority=environment_priority.lower().replace(" ", "_")
            )
            
            # Run optimization
            optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
            
            # Store results
            st.session_state.optimization_results = optimization_results
            
            st.success("✅ Optimization complete!")
    
    # Display results if available
    if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
        display_optimization_results(st.session_state.optimization_results)
        
        # Export recommendations
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv_data = export_recommendations_to_csv(st.session_state.optimization_results)
                st.download_button(
                    label="📥 Download Recommendations CSV",
                    data=csv_data,
                    file_name=f"reader_writer_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                pdf_data = generate_optimization_report_pdf(st.session_state.optimization_results)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"reader_writer_optimization_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )


def show_enhanced_environment_analysis():
    """Show enhanced environment analysis with Writer/Reader details"""
    
    st.markdown("### 🏢 Enhanced Environment Analysis")
    
    recommendations = st.session_state.enhanced_recommendations
    environment_specs = st.session_state.environment_specs
    
    # Environment comparison with cluster details
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs[env_name]
        
        # Writer configuration
        writer_config = f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})"
        
        # Reader configuration
        reader_count = rec['readers']['count']
        if reader_count > 0:
            reader_config = f"{reader_count} x {rec['readers']['instance_class']}"
        else:
            reader_config = "No readers"
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec['environment_type'].title(),
            'Current Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Writer Instance': writer_config,
            'Read Replicas': reader_config,
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'Workload Pattern': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })
    
    env_df = pd.DataFrame(env_comparison_data)
    st.dataframe(env_df, use_container_width=True)
    
    # Detailed environment insights
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
                st.write(f"IOPS Requirement: {specs.get('iops_requirement', 'N/A')}")
                st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
            
            with col2:
                st.markdown("**Writer Configuration**")
                writer = rec['writer']
                st.write(f"Instance: {writer['instance_class']}")
                st.write(f"Multi-AZ: {'Yes' if writer['multi_az'] else 'No'}")
                st.write(f"CPU Cores: {writer['cpu_cores']}")
                st.write(f"RAM: {writer['ram_gb']} GB")
                
                st.markdown("**Reader Configuration**")
                readers = rec['readers']
                if readers['count'] > 0:
                    st.write(f"Count: {readers['count']}")
                    st.write(f"Instance: {readers['instance_class']}")
                    st.write(f"Multi-AZ: {'Yes' if readers['multi_az'] else 'No'}")
                else:
                    st.write("No read replicas")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'Yes' if storage['encrypted'] else 'No'}")
                st.write(f"Backup Retention: {storage['backup_retention_days']} days")
                
                st.markdown("**Workload Characteristics**")
                st.write(f"Pattern: {rec['workload_pattern']}")
                st.write(f"Read/Write Ratio: {rec['read_write_ratio']}% reads")
                st.write(f"Peak Connections: {rec['connections']}")
            
            # Optimization recommendations
            st.markdown("**💡 Optimization Notes**")
            
            if rec['environment_type'] == 'production':
                st.success("✅ Production-grade configuration with high availability")
                if readers['count'] > 0:
                    st.info(f"📊 {readers['count']} read replicas will help distribute read load")
            elif rec['environment_type'] == 'development':
                st.info("💡 Cost-optimized configuration for development")
                if readers['count'] == 0:
                    st.info("💰 No read replicas to minimize development costs")
            
            if rec['workload_pattern'] == 'read_heavy' and readers['count'] > 0:
                st.success(f"📈 Read-heavy workload well-suited for {readers['count']} read replicas")
            elif rec['workload_pattern'] == 'read_heavy' and readers['count'] == 0:
                st.warning("⚠️ Read-heavy workload might benefit from read replicas")
            
            if storage['type'] == 'io2':
                st.info("⚡ High-performance io2 storage for demanding IOPS requirements")
            elif storage['type'] == 'gp3':
                st.info("⚖️ Balanced gp3 storage for general-purpose workloads")


# Fixed main section for migration configuration
def show_migration_configuration():
    """Show migration configuration interface with growth planning"""
    
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
    
    # Migration parameters with corrected data size section
    st.markdown("### 💾 Migration Data Configuration")
    
    st.info("""
    **📚 Understanding Storage Configuration:**
    
    - **Migration Data Size** (below): Total data to be migrated from source to target
    - **Environment Storage** (configured later): Operational storage allocation for each environment
    
    The system will help you calculate appropriate environment storage based on migration data size.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Source Data Analysis")
        data_size_gb = st.number_input(
            "Total Migration Data Size (GB)", 
            min_value=1, 
            max_value=100000, 
            value=1000,
            help="Total size of data to be migrated from source database"
        )
        
        # Migration complexity parameters
        num_applications = st.number_input("Connected Applications", min_value=1, max_value=100, value=5)
        num_stored_procedures = st.number_input("Stored Procedures/Functions", min_value=0, max_value=10000, value=50)
    
    with col2:
        st.markdown("#### 🎯 Migration Parameters")
        migration_timeline_weeks = st.number_input("Migration Timeline (weeks)", min_value=4, max_value=52, value=12)
        team_size = st.number_input("Team Size", min_value=1, max_value=20, value=5)
        team_expertise = st.selectbox("Team Expertise", ["low", "medium", "high"], index=1)
        
        # Infrastructure parameters
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"], index=0)
        use_direct_connect = st.checkbox("Use Direct Connect", value=False)
        bandwidth_mbps = st.number_input("Available Bandwidth (Mbps)", min_value=100, max_value=10000, value=1000)
        migration_budget = st.number_input("Migration Budget ($)", min_value=10000, max_value=10000000, value=500000)
    
    # AI Configuration
    st.markdown("### 🤖 AI Integration")
    anthropic_api_key = st.text_input(
        "Anthropic API Key (Optional)",
        type="password",
        help="Provide your Anthropic API key for AI-powered insights"
    )
    
   # Save configuration button
if st.button("💾 Save Configuration", type="primary", use_container_width=True):
    # ADD THIS RIGHT BEFORE THE SAVE BUTTON
    if st.session_state.environment_specs:
        st.markdown("### 📊 Current Storage Configuration")
        # FIXED: Removed extra spaces at start of this line
        total_env_storage = sum([specs.get('storage_gb', 0) for specs in st.session_state.environment_specs.values()])
                     
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Migration Data", f"{data_size_gb:,} GB")
        with col2:
            st.metric("Total Environment Storage", f"{total_env_storage:,} GB")
        with col3:
            if data_size_gb > 0:
                ratio = total_env_storage / data_size_gb
                st.metric("Storage Ratio", f"{ratio:.1f}x")
        
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
            'estimated_migration_cost': 0,
            # ADD THESE NEW GROWTH PARAMETERS:
            'annual_data_growth': annual_data_growth,
            'annual_user_growth': annual_user_growth,
            'annual_transaction_growth': annual_transaction_growth,
            'growth_scenario': growth_scenario,
            'seasonality_factor': seasonality_factor,
            'scaling_strategy': scaling_strategy
        }
        
        st.success("✅ Configuration saved! Proceed to Environment Setup.")
        st.balloons()

def show_environment_analysis():
    """Show environment analysis dashboard"""
    
    st.markdown("### 🏢 Environment Analysis")
    
    # Check for enhanced recommendations first
    if hasattr(st.session_state, 'enhanced_recommendations') and st.session_state.enhanced_recommendations:
        show_enhanced_environment_analysis()
        return
    
    if not hasattr(st.session_state, 'recommendations') or not st.session_state.recommendations:
        st.warning("Environment analysis not available. Please run the analysis first.")
        return
    
    recommendations = st.session_state.recommendations
    environment_specs = st.session_state.environment_specs
    
    if not environment_specs:
        st.warning("No environment specifications available.")
        return
    
    # Environment comparison
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs.get(env_name, {})
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec.get('environment_type', 'Unknown').title(),
            'Current Resources': f"{specs.get('cpu_cores', 'N/A')} cores, {specs.get('ram_gb', 'N/A')} GB RAM",
            'Recommended Instance': rec.get('instance_class', 'N/A'),
            'Storage': f"{specs.get('storage_gb', 'N/A')} GB",
            'Multi-AZ': 'Yes' if rec.get('multi_az', False) else 'No',
            'Daily Usage': f"{specs.get('daily_usage_hours', 'N/A')} hours"
        })
    
    if env_comparison_data:
        env_df = pd.DataFrame(env_comparison_data)
        st.dataframe(env_df, use_container_width=True)
    else:
        st.warning("No environment comparison data available.")
    
    # Environment-specific insights
    st.markdown("#### 💡 Environment Insights")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🔍 {env_name} Environment Details"):
            specs = environment_specs.get(env_name, {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Configuration**")
                st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                st.write(f"Daily Usage: {specs.get('daily_usage_hours', 'N/A')} hours")
            
            with col2:
                st.markdown("**AWS Recommendation**")
                st.write(f"Instance: {rec.get('instance_class', 'N/A')}")
                st.write(f"Environment Type: {rec.get('environment_type', 'Unknown').title()}")
                st.write(f"Multi-AZ: {'Yes' if rec.get('multi_az', False) else 'No'}")
                
                # Get instance specs (simplified)
                instance_class = rec.get('instance_class', '')
                if 'xlarge' in instance_class:
                    aws_cpu = 16 if '4xlarge' in instance_class else 8 if '2xlarge' in instance_class else 4
                    aws_ram = aws_cpu * 8
                elif 'large' in instance_class:
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
                env_type = rec.get('environment_type', '')
                if env_type == 'production':
                    st.write("✅ Production-grade configuration")
                    st.write("✅ Multi-AZ for high availability")
                elif env_type == 'development':
                    st.write("💡 Cost-optimized for development")
                    st.write("💡 Single-AZ to reduce costs")
                
                daily_hours = specs.get('daily_usage_hours', 24)
                if daily_hours < 12:
                    st.write("⚡ Consider Aurora Serverless for variable workloads")



def show_environment_setup():
    """Show environment setup interface with vROps support"""
            #show_enhanced_environment_setup_with_vrops()
    
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

def show_analysis_section_fixed():
    """Show analysis and recommendations section - UPDATED"""
    
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
            if count < 4:  # Show max 4 environments
                cpu_cores = specs.get('cpu_cores', 'N/A')
                ram_gb = specs.get('ram_gb', 'N/A')
                st.markdown(f"• **{env_name}:** {cpu_cores} cores, {ram_gb} GB RAM")
                count += 1
        
        if len(envs) > 4:
            st.markdown(f"• ... and {len(envs) - 4} more environments")
    
    # Detect configuration type
    is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
    
    if is_enhanced:
        st.info("🔬 Enhanced cluster configuration detected - Writer/Reader analysis will be performed")
    else:
        st.info("📊 Standard configuration detected - Basic RDS analysis will be performed")
    
    # Run analysis - USE THE FIXED FUNCTION
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        # Clear any previous results
        st.session_state.analysis_results = None
        if hasattr(st.session_state, 'enhanced_analysis_results'):
            st.session_state.enhanced_analysis_results = None
        
        with st.spinner("🔄 Analyzing migration requirements..."):
            run_streamlit_migration_analysis()  # Use the fixed function

# ADD THIS FUNCTION to your streamlit_app.py file:

def run_streamlit_migration_analysis():
    """Run migration analysis with growth projections - ENHANCED VERSION"""
    
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
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: Growth Analysis (NEW)
        st.write("📈 Calculating 3-year growth projections...")
        growth_analyzer = GrowthAwareCostAnalyzer()
        growth_analysis = growth_analyzer.calculate_3_year_growth_projection(
            cost_analysis, st.session_state.migration_params
        )
        st.session_state.growth_analysis = growth_analysis
        
        # Step 5: AI insights
        if anthropic_api_key:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = asyncio.run(analyzer.generate_ai_insights(cost_analysis, st.session_state.migration_params))
                st.session_state.ai_insights = ai_insights
                st.success("✅ AI insights generated")
            except Exception as e:
                st.warning(f"AI insights failed: {str(e)}")
                st.session_state.ai_insights = {
                    'summary': f"Migration analysis complete. Monthly cost: ${cost_analysis['monthly_aws_cost']:,.0f}",
                    'error': str(e)
                }
        else:
            st.info("ℹ️ Provide Anthropic API key for AI insights")
        
        st.success("✅ Analysis complete with growth projections!")
        
        # Show enhanced summary
        show_analysis_summary_with_growth()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        create_basic_fallback()

def show_analysis_summary_with_growth():
    """Enhanced analysis summary including growth metrics"""
    
    st.markdown("#### 🎯 Analysis Summary with Growth Projections")
    
    col1, col2, col3, col4 = st.columns(4)
    
    results = st.session_state.analysis_results
    
    with col1:
        st.metric("Current Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
    
    with col2:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    
    with col3:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_percent = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_percent:.1f}%")
        else:
            st.metric("3-Year Growth", "Calculating...")
    
    with col4:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            total_investment = st.session_state.growth_analysis['growth_summary']['total_3_year_investment']
            st.metric("3-Year Investment", f"${total_investment:,.0f}")
        else:
            if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
                risk_level = st.session_state.risk_assessment['risk_level']['level']
                st.metric("Risk Level", risk_level)
    
    st.info("📈 View detailed growth projections in the 'Results Dashboard' section")

def show_simple_summary():
    """Show simple analysis summary"""
    
    st.markdown("#### 🎯 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    results = st.session_state.analysis_results
    
    if results:
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


def create_basic_fallback():
    """Create basic fallback when analysis completely fails"""
    
    st.warning("⚠️ Creating fallback analysis...")
    
    # Basic recommendations
    recommendations = {}
    total_cost = 0
    
    for env_name, specs in st.session_state.environment_specs.items():
        recommendations[env_name] = {
            'environment_type': 'production' if 'prod' in env_name.lower() else 'development',
            'instance_class': 'db.r5.large',
            'cpu_cores': specs.get('cpu_cores', 4),
            'ram_gb': specs.get('ram_gb', 16),
            'storage_gb': specs.get('storage_gb', 500),
            'multi_az': 'prod' in env_name.lower()
        }
        total_cost += 500  # $500 per environment estimate
    
    st.session_state.recommendations = recommendations
    
    # Basic cost analysis
    st.session_state.analysis_results = {
        'monthly_aws_cost': total_cost,
        'annual_aws_cost': total_cost * 12,
        'environment_costs': {env: {'total_monthly': 500, 'instance_cost': 400, 'storage_cost': 100} 
                            for env in recommendations.keys()},
        'migration_costs': {'total': 50000, 'dms_instance': 20000, 'data_transfer': 10000, 'professional_services': 20000}
    }
    
    # Basic risk assessment
    st.session_state.risk_assessment = get_fallback_risk_assessment()
    
    st.success("✅ Fallback analysis created")
                
def run_enhanced_migration_analysis():
    """Run enhanced migration analysis with Writer/Reader support"""
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedMigrationAnalyzer()
        
         # Step 1: Calculate enhanced recommendations
        st.write("📊 Calculating cluster recommendations...")
        recommendations = analyzer.calculate_enhanced_instance_recommendations(st.session_state.environment_specs)
        st.session_state.enhanced_recommendations = recommendations
        
        # Step 2: Calculate enhanced costs
        st.write("💰 Analyzing cluster costs...")
        cost_analysis = analyzer.calculate_enhanced_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.enhanced_analysis_results = cost_analysis
        
        # Step 3: Generate risk assessment using enhanced data
        st.write("⚠️ Assessing risks...")
        risk_assessment = calculate_migration_risks(st.session_state.migration_params, recommendations)
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: Generate cost comparison
        st.write("📈 Generating cost comparisons...")
        generate_enhanced_cost_visualizations()
        
        st.success("✅ Enhanced cluster analysis complete!")
        
        # Show summary
        show_enhanced_analysis_summary()
        
        # Provide navigation hint
        st.info("📈 View detailed results in the 'Results Dashboard' section")
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        st.code(str(e))
        
        # Provide troubleshooting info
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("If the error persists:")
        st.markdown("1. Check that all environment fields are properly filled")
        st.markdown("2. Verify that numerical values are within valid ranges")
        st.markdown("3. Try using the 'Simple Configuration' option instead")

def show_results_dashboard():
    """Show comprehensive results dashboard with vROps analysis"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run the migration analysis first.")
        return
    
    st.markdown("## 📊 Migration Analysis Results")
    
     # FIXED: Define has_enhanced_results properly
    has_enhanced_results = (
        hasattr(st.session_state, 'enhanced_analysis_results') and 
        st.session_state.enhanced_analysis_results is not None
    )
    
    
    # Create tabs for different views - ADD VROPS TAB
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "💰 Cost Summary",
        "📈 Growth Projections",
        "📊 vROps Analysis",  # <-- NEW TAB
        "💎 Enhanced Analysis",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "📊 Visualizations",
        "🤖 AI Insights",
        "📅 Timeline"
    ])
    
    with tab1:
        show_basic_cost_summary()
    
    with tab2:
        show_growth_analysis_dashboard()
    
    with tab3:  # <-- NEW TAB CONTENT
        show_vrops_results_tab()
    
    with tab4:
        if has_enhanced_results:
            show_enhanced_cost_analysis()
        else:
            st.info("💡 Enhanced cost analysis not available.")
            show_basic_cost_summary()

    with tab5:
        show_environment_analysis_tab()

    with tab6:
        show_visualizations_tab()

    with tab7:
        show_ai_insights_tab()

    with tab8:
        show_timeline_analysis_tab()

def show_basic_cost_summary():
    """Show basic cost summary from analysis results"""
    
    if not st.session_state.analysis_results:
        st.error("No analysis results available.")
        return
    
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
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_percent = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_percent:.1f}%")
        else:
            total_cost = annual_cost + migration_cost
            st.metric("Total First Year", f"${total_cost:,.0f}")
    
    # Environment costs breakdown
    st.markdown("### 💰 Cost Breakdown by Environment")
    
    env_costs = results.get('environment_costs', {})
    if env_costs:
        for env_name, costs in env_costs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                
                # Check if it's enhanced results format
                if isinstance(costs, dict) and 'instance_cost' in costs:
                    # Enhanced format
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Instance Cost", f"${costs.get('instance_cost', 0):,.2f}/month")
                        st.metric("Storage Cost", f"${costs.get('storage_cost', 0):,.2f}/month")
                    
                    with col2:
                        st.metric("Reader Instances", f"${costs.get('reader_costs', 0):,.2f}/month")
                        st.metric("Backup Cost", f"${costs.get('backup_cost', 0):,.2f}/month")
                    
                    with col3:
                        total_env_cost = costs.get('total_monthly_cost', 
                                                 sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'reader_costs', 'backup_cost']]))
                        st.metric("Total Monthly", f"${total_env_cost:,.2f}")
                        
                        
                       
                else:
                    # Simple format - just show the cost
                    if isinstance(costs, (int, float)):
                        st.metric("Monthly Cost", f"${costs:,.2f}")
                    else:
                        st.write("Cost information not available in expected format")
    
    # Migration timeline and costs
    if 'migration_costs' in results:
        st.markdown("### 🚀 Migration Investment")
        
        migration = results['migration_costs']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Schema Migration", f"${migration.get('schema_migration', 0):,.0f}")
        
        with col2:
            st.metric("Data Transfer", f"${migration.get('data_transfer', 0):,.0f}")
        
        with col3:
            st.metric("Testing & Validation", f"${migration.get('testing', 0):,.0f}")

def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    # Check if growth analysis exists
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
        
        # Show basic growth planning instead
        st.markdown("#### 🎯 Growth Planning Preview")
        st.info("""
        **Growth analysis will show:**
        - 3-year cost projections with growth factors
        - Resource scaling requirements
        - Seasonal peak planning
        - Cost optimization opportunities
        - Scaling recommendations by year
        
        Run the migration analysis to see detailed growth projections.
        """)
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
    # Key Growth Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "3-Year Growth",
            f"{growth_summary['total_3_year_growth_percent']:.1f}%",
            delta=f"CAGR: {growth_summary['compound_annual_growth_rate']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Current Annual Cost",
            f"${growth_summary['year_0_cost']:,.0f}",
            delta="Baseline"
        )
    
    with col3:
        st.metric(
            "Year 3 Projected Cost",
            f"${growth_summary['year_3_cost']:,.0f}",
            delta=f"+${growth_summary['year_3_cost'] - growth_summary['year_0_cost']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Total 3-Year Investment",
            f"${growth_summary['total_3_year_investment']:,.0f}",
            delta=f"Avg: ${growth_summary['average_annual_cost']:,.0f}/year"
        )
    
    # Growth Projection Charts
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")
        
        # Show basic growth table instead
        st.markdown("#### 📈 Year-by-Year Projections")
        projections = growth_analysis['yearly_projections']
        
        years_data = []
        for year in range(4):
            year_data = projections[f'year_{year}']
            years_data.append({
                'Year': f'Year {year}' if year > 0 else 'Current',
                'Monthly Cost': f"${year_data['total_monthly']:,.0f}",
                'Annual Cost': f"${year_data['total_annual']:,.0f}",
                'Peak Cost': f"${year_data['peak_annual']:,.0f}"
            })
        
        st.table(years_data)
    
    # Scaling Recommendations
    st.markdown("#### 🎯 Scaling Recommendations")
    
    recommendations = growth_analysis.get('scaling_recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': '#e53e3e',
                'Medium': '#d69e2e',
                'Low': '#38a169'
            }.get(rec['priority'], '#666666')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 15px; margin: 10px 0; background: {priority_color}22;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {priority_color};">{rec['type']} (Year {rec['year']})</strong><br>
                        {rec['description']}<br>
                        <em>Action: {rec['action']}</em>
                    </div>
                    <div style="text-align: right;">
                        <strong>Priority: {rec['priority']}</strong><br>
                        <span style="color: #38a169;">Potential Savings: ${rec['estimated_savings']:,.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical scaling issues identified in the 3-year projection.")
        
        # Show default recommendations
        st.markdown("#### 💡 General Growth Recommendations")
        default_recommendations = [
            "📊 **Monitor growth trends** - Track actual vs projected growth quarterly",
            "💰 **Reserved Instances** - Consider 1-3 year commitments for 30-40% savings",
            "🔄 **Auto-scaling** - Implement auto-scaling for variable workloads",
            "📈 **Capacity planning** - Review and adjust capacity every 6 months",
            "🗄️ **Data lifecycle** - Implement archiving for older data to reduce storage costs"
        ]
        
        for rec in default_recommendations:
            st.markdown(rec)

def show_risk_assessment_tab():
    """Show risk assessment results"""
    
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        show_risk_assessment_robust()
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")

def show_environment_analysis_tab():
    """Show environment analysis"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Environment analysis not available. Please run the migration analysis first.")
        return
        
    st.markdown("### 🏢 Environment Analysis")
    
    # Show environment specifications
    if hasattr(st.session_state, 'environment_specs') and st.session_state.environment_specs:
        st.markdown("#### 📋 Environment Specifications")
        
        for env_name, specs in st.session_state.environment_specs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Specs:**")
                    st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                    st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                    st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                    st.write(f"Daily Usage: {specs.get('daily_usage_hours', 'N/A')} hours")
                
                with col2:
                    st.markdown("**Workload:**")
                    st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
                    st.write(f"Workload Pattern: {specs.get('workload_pattern', 'N/A')}")
                    st.write(f"Environment Type: {specs.get('environment_type', 'N/A')}")
    
    # Show recommendations if available
    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
        st.markdown("#### 💡 Environment Recommendations")
        
        recommendations = st.session_state.recommendations
        for env_name, rec in recommendations.items():
            with st.expander(f"🎯 {env_name.title()} Recommendations"):
                if isinstance(rec, dict):
                    st.markdown(f"**Recommended Instance:** {rec.get('instance_class', 'N/A')}")
                    st.markdown(f"**Environment Type:** {rec.get('environment_type', 'N/A')}")
                    st.markdown(f"**Multi-AZ:** {'Yes' if rec.get('multi_az', False) else 'No'}")
                    
                    if 'reasoning' in rec:
                        st.markdown("**Reasoning:**")
                        st.write(rec['reasoning'])

def show_visualizations_tab():
    """Show visualization charts"""
    
    st.markdown("### 📊 Cost & Performance Visualizations")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Visualizations not available. Please run the migration analysis first.")
        return
    
    try:
        # Cost comparison chart
        results = st.session_state.analysis_results
        
        # Create simple cost visualization
        env_costs = results.get('environment_costs', {})
        
        if env_costs:
            # Environment cost comparison
            env_names = []
            monthly_costs = []
            
            for env_name, costs in env_costs.items():
                env_names.append(env_name.title())
                
                if isinstance(costs, dict):
                    cost = costs.get('total_monthly_cost', 
                                   sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'reader_costs', 'backup_cost']]))
                else:
                    cost = float(costs) if costs else 0
                
                monthly_costs.append(cost)
            
            if env_names and monthly_costs:
                fig = go.Figure(data=[
                    go.Bar(x=env_names, y=monthly_costs, marker_color='#3182ce')
                ])
                
                fig.update_layout(
                    title='Monthly Cost by Environment',
                    xaxis_title='Environment',
                    yaxis_title='Monthly Cost ($)',
                    height=400
                )
                
                # FIXED: Added unique key
                st.plotly_chart(fig, use_container_width=True, key="env_cost_visualization_chart")
        
        # Growth visualization if available
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            st.markdown("#### 📈 Growth Projections")
            try:
                charts = create_growth_projection_charts(st.session_state.growth_analysis)
                # FIXED: Added unique keys for each chart
                for i, chart in enumerate(charts):
                    st.plotly_chart(chart, use_container_width=True, key=f"viz_growth_chart_{i}")
            except Exception as e:
                st.error(f"Error creating growth charts: {str(e)}")
        
           # Enhanced cost chart if available
        if hasattr(st.session_state, 'enhanced_cost_chart') and st.session_state.enhanced_cost_chart:
            st.markdown("#### 💎 Enhanced Cost Analysis")
            # FIXED: Added unique key
            st.plotly_chart(st.session_state.enhanced_cost_chart, use_container_width=True, key="enhanced_cost_visualization_chart")
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def display_optimization_results(optimization_results: Dict):
    """Display comprehensive optimization results in Streamlit"""
    
    st.markdown("# 🚀 Optimized Reader/Writer Recommendations")
    
    # Overall summary
    total_monthly_cost = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                             for env in optimization_results.values()])
    total_annual_cost = total_monthly_cost * 12
    avg_optimization_score = sum([env['optimization_score'] 
                                 for env in optimization_results.values()]) / len(optimization_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Monthly Cost", f"${total_monthly_cost:,.0f}")
    
    with col2:
        st.metric("Total Annual Cost", f"${total_annual_cost:,.0f}")
    
    with col3:
        st.metric("Avg Optimization Score", f"{avg_optimization_score:.1f}/100")
    
    with col4:
        total_instances = sum([1 + env['reader_optimization']['count'] 
                              for env in optimization_results.values()])
        st.metric("Total Instances", total_instances)
    
    # Detailed results for each environment
    for env_name, optimization in optimization_results.items():
        with st.expander(f"🏢 {env_name.title()} - Optimization Score: {optimization['optimization_score']:.1f}/100", expanded=True):
            
            # Configuration overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✍️ Writer Configuration")
                writer = optimization['writer_optimization']
                st.info(f"""
                **Instance:** {writer['instance_class']}  
                **vCPUs:** {writer['specs'].vcpu}  
                **Memory:** {writer['specs'].memory_gb} GB  
                **Network:** {writer['specs'].network_performance}  
                **Multi-AZ:** {'✅ Yes' if writer['multi_az'] else '❌ No'}  
                **Monthly Cost:** ${writer['monthly_cost']:,.0f}  
                **Annual Cost:** ${writer['annual_cost']:,.0f}
                """)
                
                st.markdown("**Reasoning:**")
                st.write(writer['reasoning'])
            
            with col2:
                st.markdown("#### 📖 Reader Configuration")
                reader = optimization['reader_optimization']
                
                if reader['count'] > 0:
                    st.success(f"""
                    **Count:** {reader['count']} replicas  
                    **Instance:** {reader['instance_class']}  
                    **vCPUs:** {reader['specs'].vcpu} each  
                    **Memory:** {reader['specs'].memory_gb} GB each  
                    **Multi-AZ:** {'✅ Yes' if reader['multi_az'] else '❌ No'}  
                    **Cost per Reader:** ${reader['single_reader_monthly_cost']:,.0f}/month  
                    **Total Monthly Cost:** ${reader['total_monthly_cost']:,.0f}  
                    **Total Annual Cost:** ${reader['total_annual_cost']:,.0f}
                    """)
                else:
                    st.warning("**No read replicas recommended**")
                
                st.markdown("**Reasoning:**")
                st.write(reader['reasoning'])
            
            # Cost breakdown chart
            st.markdown("#### 💰 Cost Breakdown")
            
            cost_data = optimization['cost_analysis']['monthly_breakdown']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Writer Instance', 'Reader Instances', 'Storage', 'Backup', 'Monitoring', 'Cross-AZ Transfer'],
                values=[cost_data['writer_instance'], cost_data['reader_instances'], 
                       cost_data['storage'], cost_data['backup'], cost_data['monitoring'], cost_data['cross_az_transfer']],
                hole=0.4,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
            )])
            
            fig.update_layout(
                title=f"{env_name} Monthly Cost Distribution - Total: ${cost_data['total']:,.0f}",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"optimization_cost_chart_{env_name}")
            
            # Reserved Instance savings
            st.markdown("#### 💳 Reserved Instance Savings Opportunities")
            
            ri_1_year = optimization['cost_analysis']['reserved_instance_options']['1_year']
            ri_3_year = optimization['cost_analysis']['reserved_instance_options']['3_year']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1-Year Reserved Instances**")
                st.metric("Annual Savings", f"${ri_1_year['total_savings']:,.0f}", 
                         delta=f"{ri_1_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_1_year['monthly_cost']:,.0f}")
            
            with col2:
                st.markdown("**3-Year Reserved Instances**")
                st.metric("Total 3-Year Savings", f"${ri_3_year['total_savings']:,.0f}", 
                         delta=f"{ri_3_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_3_year['monthly_cost']:,.0f}")


def show_ai_insights_tab():
    """Show AI insights if available"""
    
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        st.markdown("### 🤖 AI-Powered Insights")
        
        insights = st.session_state.ai_insights
        
        if 'error' in insights:
            st.warning(f"AI insights partially available: {insights.get('summary', 'Analysis complete')}")
            st.error(f"Error: {insights['error']}")
        else:
            # Show AI insights
            if 'summary' in insights:
                st.markdown("#### 📝 Executive Summary")
                st.write(insights['summary'])
            
            if 'recommendations' in insights:
                st.markdown("#### 💡 AI Recommendations")
                for rec in insights['recommendations']:
                    st.markdown(f"• {rec}")
            
            if 'cost_optimization' in insights:
                st.markdown("#### 💰 Cost Optimization")
                st.write(insights['cost_optimization'])
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
    
    # Key milestones
    st.markdown("#### 🎯 Key Milestones")
    
    milestones = [
        f"Week {int(phases[0]['weeks'])}: Assessment Complete",
        f"Week {int(phases[0]['weeks'] + phases[1]['weeks'])}: Schema Migration Complete",
        f"Week {int(sum(p['weeks'] for p in phases[:3]))}: Data Migration Complete",
        f"Week {int(sum(p['weeks'] for p in phases[:4]))}: Testing Complete",
        f"Week {timeline_weeks}: Go-Live"
    ]
    
    for milestone in milestones:
        st.markdown(f"• {milestone}")
    
    # Team and resources
    st.markdown("#### 👥 Team & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Team Configuration:**")
        st.write(f"Team Size: {params.get('team_size', 5)} people")
        st.write(f"Expertise Level: {params.get('team_expertise', 'medium').title()}")
        st.write(f"Timeline: {timeline_weeks} weeks")
    
    with col2:
        st.markdown("**Migration Budget:**")
        budget = params.get('migration_budget', 500000)
        st.write(f"Total Budget: ${budget:,.0f}")
        weekly_budget = budget / timeline_weeks if timeline_weeks > 0 else 0
        st.write(f"Weekly Budget: ${weekly_budget:,.0f}")

def show_reports_section():
    """Show reports and export section - ROBUST VERSION"""
    
    st.markdown("## 📄 Reports & Export")
    
    # Check for both regular and enhanced analysis results
    has_regular_results = st.session_state.analysis_results is not None
    has_enhanced_results = hasattr(st.session_state, 'enhanced_analysis_results') and st.session_state.enhanced_analysis_results is not None
    
    if not has_regular_results and not has_enhanced_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        st.info("👆 Go to 'Analysis & Recommendations' section and click '🚀 Run Comprehensive Analysis'")
        
        # Show current status
        st.markdown("### 📊 Current Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            config_status = "✅ Complete" if st.session_state.migration_params else "❌ Missing"
            st.metric("Migration Config", config_status)
        
        with col2:
            env_status = "✅ Complete" if st.session_state.environment_specs else "❌ Missing"
            st.metric("Environment Setup", env_status)
        
        with col3:
            analysis_status = "✅ Complete" if (has_regular_results or has_enhanced_results) else "❌ Pending"
            st.metric("Analysis Results", analysis_status)
        
        return
    
    # Determine which results to use
    if has_enhanced_results:
        results = st.session_state.enhanced_analysis_results
        recommendations = getattr(st.session_state, 'enhanced_recommendations', {})
        st.info("📊 Using Enhanced Analysis Results")
    else:
        results = st.session_state.analysis_results
        recommendations = getattr(st.session_state, 'recommendations', {})
        st.info("📊 Using Standard Analysis Results")
    
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
                    pdf_buffer = generate_executive_summary_pdf_robust(
                        results,
                        st.session_state.migration_params
                    )
                    
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
        st.markdown("Detailed technical analysis for architects and engineers")
        st.markdown("**Includes:**")
        st.markdown("• Environment specifications")
        st.markdown("• Instance recommendations")
        st.markdown("• Detailed cost breakdown")
        st.markdown("• Technical considerations")
        
        if st.button("📄 Generate Technical PDF", key="tech_pdf", use_container_width=True):
            with st.spinner("Generating technical report..."):
                try:
                    pdf_buffer = generate_technical_report_pdf_robust(
                        results,
                        recommendations,
                        st.session_state.migration_params
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Technical Report",
                            data=pdf_buffer.getvalue(),
                            file_name=f"AWS_Migration_Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col3:
        st.markdown("#### 📊 Data Export")
        st.markdown("Raw data for further analysis")
        st.markdown("**Includes:**")
        st.markdown("• Cost analysis data")
        st.markdown("• Environment specifications")
        st.markdown("• Risk assessment data")
        st.markdown("• Recommendations")
        
        if st.button("📊 Export Data (CSV)", key="csv_export", use_container_width=True):
            try:
                # Prepare CSV data
                csv_data = prepare_csv_export_data(results, recommendations)
                
                if csv_data:
                    csv_string = csv_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV Data",
                        data=csv_string,
                        file_name=f"AWS_Migration_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("No data available for export")
            except Exception as e:
                st.error(f"Error preparing CSV: {str(e)}")
    
    # Bulk download option
    st.markdown("---")
    st.markdown("### 📦 Bulk Download")
    
    if st.button("📊 Generate All Reports", key="bulk_reports", use_container_width=True):
        with st.spinner("Generating all reports... This may take a moment..."):
            try:
                # Create ZIP file with all reports
                zip_buffer = create_bulk_reports_zip(results, recommendations, st.session_state.migration_params)
                
                if zip_buffer:
                    st.download_button(
                        label="📥 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"AWS_Migration_Complete_Analysis_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to generate reports package")
            except Exception as e:
                st.error(f"Error generating bulk reports: {str(e)}")

def show_optimized_recommendations():
    """Show optimized Reader/Writer recommendations"""
    
    st.markdown("## 🧠 AI-Optimized Reader/Writer Recommendations")
    
    if not st.session_state.environment_specs:
        st.warning("⚠️ Please configure environments first.")
        st.info("👆 Go to 'Environment Setup' section to configure your database environments")
        return
    
    # Show current environment count
    env_count = len(st.session_state.environment_specs)
    is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Environments Configured", env_count)
    with col2:
        config_type = "Enhanced Cluster Config" if is_enhanced else "Basic Config"
        st.metric("Configuration Type", config_type)
    
    if not is_enhanced:
        st.info("💡 For best results, use the Enhanced Environment Setup with cluster configuration")
    
    # Configuration options
    st.markdown("### ⚙️ Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Cost Efficiency", "Performance", "Balanced"],
            index=2,
            help="Choose whether to prioritize cost savings, performance, or a balance of both"
        )
    
    with col2:
        consider_reserved_instances = st.checkbox(
            "Include Reserved Instance Analysis",
            value=True,
            help="Calculate potential savings with 1-year and 3-year Reserved Instances"
        )
    
    with col3:
        environment_priority = st.selectbox(
            "Environment Priority",
            ["Production First", "All Equal", "Cost First"],
            index=0,
            help="Prioritization strategy for resource allocation"
        )
    
    # Run optimization button
    if st.button("🧠 Generate AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing workloads and optimizing configurations..."):
            
            try:
                # Initialize optimizer
                optimizer = OptimizedReaderWriterAnalyzer()
                
                # Apply optimization settings
                optimizer.set_optimization_preferences(
                    goal=optimization_goal.lower().replace(" ", "_"),
                    consider_reserved=consider_reserved_instances,
                    environment_priority=environment_priority.lower().replace(" ", "_")
                )
                
                # Run optimization
                optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
                
                # Store results
                st.session_state.optimization_results = optimization_results
                
                st.success("✅ Optimization complete!")
                
                # Show quick summary
                total_monthly = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                                   for env in optimization_results.values()])
                avg_score = sum([env['optimization_score'] 
                               for env in optimization_results.values()]) / len(optimization_results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Monthly Cost", f"${total_monthly:,.0f}")
                with col2:
                    st.metric("Average Optimization Score", f"{avg_score:.1f}/100")
                with col3:
                    potential_ri_savings = sum([
                        env['cost_analysis']['reserved_instance_options']['3_year']['total_savings']
                        for env in optimization_results.values()
                    ])
                    st.metric("Potential 3-Year RI Savings", f"${potential_ri_savings:,.0f}")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.code(str(e))
    
    # Display results if available
    if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
        st.markdown("---")
        display_optimization_results(st.session_state.optimization_results)

# STEP 6: UPDATE YOUR SESSION STATE (find initialize_session_state function and add this line)
# Add this line to your defaults dictionary:
# 'optimization_results': None,

# STEP 7: UPDATE YOUR MAIN NAVIGATION (find your main() function and update the page radio)
# Add "🧠 AI Optimizer" to your page options and add the routing case:
# elif page == "🧠 AI Optimizer":
#     show_optimized_recommendations()

def prepare_csv_export_data(results, recommendations):
    """Prepare CSV data for export"""
    try:
        env_costs = results.get('environment_costs', {})
        if not env_costs:
            return None
        
        csv_data = []
        for env_name, costs in env_costs.items():
            # Handle both enhanced and regular cost structures
            if isinstance(costs, dict):
                # Get cost components safely
                instance_cost = costs.get('instance_cost', costs.get('writer_instance_cost', 0))
                storage_cost = costs.get('storage_cost', 0)
                backup_cost = costs.get('backup_cost', 0)
                total_monthly = costs.get('total_monthly', 0)
                
                # Additional enhanced fields
                reader_costs = costs.get('reader_costs', 0)
                reader_count = costs.get('reader_count', 0)
                
                csv_data.append({
                    'Environment': env_name,
                    'Instance_Cost': instance_cost,
                    'Reader_Costs': reader_costs,
                    'Reader_Count': reader_count,
                    'Storage_Cost': storage_cost,
                    'Backup_Cost': backup_cost,
                    'Total_Monthly': total_monthly,
                    'Total_Annual': total_monthly * 12
                })
        
        if csv_data:
            return pd.DataFrame(csv_data)
        else:
            return None
            
    except Exception as e:
        print(f"Error preparing CSV data: {e}")
        return None

def generate_executive_summary_pdf_robust(results, migration_params):
    """Generate executive summary PDF - ROBUST VERSION"""
    
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
        
        # Key metrics table
        migration_costs = results.get('migration_costs', {})
        
        metrics_data = [
            ['Metric', 'Value', 'Impact'],
            ['Monthly AWS Cost', f"${results.get('monthly_aws_cost', 0):,.0f}", 'Operational'],
            ['Annual AWS Cost', f"${results.get('annual_aws_cost', results.get('monthly_aws_cost', 0) * 12):,.0f}", 'Budget Planning'],
            ['Migration Investment', f"${migration_costs.get('total', 0):,.0f}", 'One-time'],
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
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating executive PDF: {e}")
        return None

def generate_technical_report_pdf_robust(results, recommendations, migration_params):
    """Generate technical report PDF - ROBUST VERSION"""
    
    try:
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
            ['Environments', str(len(recommendations)) if recommendations else '0'],
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
        story.append(Spacer(1, 20))
        
        # Environment recommendations if available
        if recommendations:
            story.append(Paragraph("Environment Recommendations", styles['Heading2']))
            
            env_data = [['Environment', 'Instance Class', 'Configuration', 'Multi-AZ']]
            
            for env_name, rec in recommendations.items():
                if 'writer' in rec:  # Enhanced recommendations
                    instance_class = rec['writer']['instance_class']
                    config = f"Writer + {rec['readers']['count']} readers"
                    multi_az = 'Yes' if rec['writer']['multi_az'] else 'No'
                else:  # Regular recommendations
                    instance_class = rec.get('instance_class', 'N/A')
                    config = f"{rec.get('cpu_cores', 'N/A')} cores, {rec.get('ram_gb', 'N/A')} GB"
                    multi_az = 'Yes' if rec.get('multi_az', False) else 'No'
                
                env_data.append([env_name, instance_class, config, multi_az])
            
            env_table = Table(env_data)
            env_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            
            story.append(env_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating technical PDF: {e}")
        return None

def create_bulk_reports_zip(results, recommendations, migration_params):
    """Create ZIP file with all reports"""
    
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
            
            # Technical report
            tech_pdf = generate_technical_report_pdf_robust(results, recommendations, migration_params)
            if tech_pdf:
                zip_file.writestr(
                    f"Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    tech_pdf.getvalue()
                )
            
            # CSV data
            csv_data = prepare_csv_export_data(results, recommendations)
            if csv_data is not None:
                zip_file.writestr(
                    f"Migration_Analysis_Data_{datetime.now().strftime('%Y%m%d')}.csv",
                    csv_data.to_csv(index=False)
                )
        
        zip_buffer.seek(0)
        return zip_buffer
        
    except Exception as e:
        print(f"Error creating ZIP file: {e}")
        return None

# Enhanced Environment Setup Interface
def show_enhanced_environment_setup_with_cluster_config():
    """Enhanced environment setup with Writer/Reader configuration"""
    st.markdown("## 📊 Database Cluster Configuration")
    
    # ADD THIS RIGHT AFTER THE TITLE
    if st.session_state.migration_params:
        with st.expander("🤖 Storage Auto-Calculator (Optional)", expanded=False):
            show_storage_auto_calculator()
        st.markdown("---")  # Add a separator
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return  # This return is now properly inside the function
    
    # Configuration method selection
    config_method = st.radio(
        "Choose configuration method:",
        [
            "📝 Manual Cluster Configuration", 
            "📁 Bulk Upload with Cluster Details",
            "🔄 Simple Configuration (Legacy)"
        ],
        horizontal=True
    )
    
    if config_method == "📝 Manual Cluster Configuration":
        show_manual_cluster_configuration()
    elif config_method == "📁 Bulk Upload with Cluster Details":
        show_bulk_cluster_upload()
    else:
        show_simple_configuration()
    
    # Add this near the end of your environment setup function
    if st.session_state.migration_params and st.session_state.environment_specs:
        st.markdown("---")  # Add a separator
        show_storage_validation_widget()
        
def show_manual_cluster_configuration():
    """Show manual cluster configuration with Writer/Reader options"""
    
    st.markdown("### 📝 Database Cluster Configuration")
    
    # Number of environments
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    for i in range(num_environments):
        with st.expander(f"🏢 Environment {i+1} - Cluster Configuration", expanded=i == 0):
            
            # Basic environment info
            col1, col2 = st.columns(2)
            
            with col1:
                env_name = st.text_input(
                    "Environment Name",
                    value=default_names[i] if i < len(default_names) else f"Environment_{i+1}",
                    key=f"env_name_{i}"
                )
                
                environment_type = st.selectbox(
                    "Environment Type",
                    ["Production", "Staging", "Testing", "Development"],
                    index=min(i, 3),
                    key=f"env_type_{i}"
                )
            
            with col2:
                workload_pattern = st.selectbox(
                    "Workload Pattern",
                    ["balanced", "read_heavy", "write_heavy", "analytics"],
                    key=f"workload_{i}"
                )
                
                read_write_ratio = st.slider(
                    "Read/Write Ratio (% Reads)",
                    min_value=10, max_value=95, value=70,
                    key=f"read_ratio_{i}"
                )
            
            # Infrastructure configuration
            st.markdown("#### 💻 Infrastructure Requirements")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
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
            
            with col2:
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"storage_{i}"
                )
                
                iops_requirement = st.number_input(
                    "IOPS Requirement",
                    min_value=100, max_value=50000,
                    value=[1000, 3000, 5000, 10000][min(i, 3)],
                    key=f"iops_{i}"
                )
            
            with col3:
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"connections_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"usage_{i}"
                )
            
            # Cluster configuration
            st.markdown("#### 🔗 Cluster Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                multi_az_writer = st.checkbox(
                    "Multi-AZ for Writer",
                    value=environment_type in ["Production", "Staging"],
                    key=f"multi_az_writer_{i}",
                    help="Deploy writer instance across multiple Availability Zones for high availability"
                )
                
                custom_reader_count = st.checkbox(
                    "Custom Reader Count",
                    value=False,
                    key=f"custom_readers_{i}"
                )
                
                if custom_reader_count:
                    num_readers = st.number_input(
                        "Number of Read Replicas",
                        min_value=0, max_value=5,
                        value=1 if environment_type == "Production" else 0,
                        key=f"num_readers_{i}"
                    )
                else:
                    # Auto-calculate based on environment and workload
                    cluster_config = DatabaseClusterConfiguration()
                    num_readers = cluster_config.calculate_optimal_readers(
                        environment_type.lower(), workload_pattern, peak_connections
                    )
                    st.info(f"Recommended readers: {num_readers} (auto-calculated)")
            
            with col2:
                multi_az_readers = st.checkbox(
                    "Multi-AZ for Readers",
                    value=environment_type == "Production",
                    key=f"multi_az_readers_{i}",
                    help="Deploy read replicas across multiple Availability Zones"
                )
                
                if num_readers > 0:
                    custom_reader_instance = st.checkbox(
                        "Custom Reader Instance Size",
                        value=False,
                        key=f"custom_reader_instance_{i}"
                    )
                    
                    if custom_reader_instance:
                        reader_instance_override = st.selectbox(
                            "Reader Instance Class",
                            ["db.t3.medium", "db.t3.large", "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge"],
                            key=f"reader_instance_{i}"
                        )
                    else:
                        reader_instance_override = None
                        st.info("Reader size will be auto-calculated based on writer size and workload")
                else:
                    reader_instance_override = None
            
            # Storage configuration
            st.markdown("#### 💾 Storage Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                storage_encrypted = st.checkbox(
                    "Encryption at Rest",
                    value=environment_type in ["Production", "Staging"],
                    key=f"encryption_{i}"
                )
            
            with col2:
                backup_retention = st.number_input(
                    "Backup Retention (Days)",
                    min_value=1, max_value=35,
                    value=30 if environment_type == "Production" else 7,
                    key=f"backup_{i}"
                )
            
            with col3:
                auto_storage_scaling = st.checkbox(
                    "Auto Storage Scaling",
                    value=True,
                    key=f"auto_scaling_{i}",
                    help="Automatically scale storage when needed"
                )
            
            # Store environment configuration
            environment_specs[env_name] = {
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'iops_requirement': iops_requirement,
                'peak_connections': peak_connections,
                'daily_usage_hours': daily_usage_hours,
                'workload_pattern': workload_pattern,
                'read_write_ratio': read_write_ratio,
                'environment_type': environment_type.lower(),
                'multi_az_writer': multi_az_writer,
                'multi_az_readers': multi_az_readers,
                'num_readers': num_readers if custom_reader_count else None,
                'reader_instance_override': reader_instance_override,
                'storage_encrypted': storage_encrypted,
                'backup_retention': backup_retention,
                'auto_storage_scaling': auto_storage_scaling
            }
    
    if st.button("💾 Save Cluster Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Cluster configuration saved!")
        
        # Run enhanced analysis
        if st.button("🚀 Analyze Cluster Configuration", type="secondary", use_container_width=True):
            with st.spinner("🔄 Analyzing cluster requirements..."):
                analyzer = EnhancedMigrationAnalyzer()
                recommendations = analyzer.calculate_enhanced_instance_recommendations(environment_specs)
                st.session_state.enhanced_recommendations = recommendations
                
                # Show preview
                show_cluster_configuration_preview(recommendations)

def show_bulk_cluster_upload():
    """Show bulk upload interface for cluster configurations"""
    
    st.markdown("### 📁 Bulk Cluster Configuration Upload")
    
    with st.expander("📋 Download Enhanced Template", expanded=False):
        st.markdown("""
        **Enhanced Template includes:**
        - Writer/Reader configuration
        - Multi-AZ options
        - Storage specifications
        - Workload patterns
        - IOPS requirements
        """)
        
        # Generate enhanced template
        enhanced_template = create_enhanced_cluster_template()
        csv_data = enhanced_template.to_csv(index=False)
        
        st.dataframe(enhanced_template, use_container_width=True)
        
        st.download_button(
            label="📥 Download Enhanced Cluster Template",
            data=csv_data,
            file_name="enhanced_cluster_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Cluster Configuration",
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file with cluster specifications"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Process cluster data
            environment_specs = process_cluster_data(df)
            
            if environment_specs:
                st.session_state.environment_specs = environment_specs
                st.success(f"✅ Successfully processed {len(environment_specs)} cluster configurations!")
                
                # Show summary
                show_cluster_upload_summary(environment_specs)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def create_enhanced_cluster_template() -> pd.DataFrame:
    """Create enhanced cluster template"""
    
    template_data = {
        'Environment_Name': ['Production-Cluster', 'Staging-Cluster', 'QA-Cluster', 'Dev-Cluster'],
        'Environment_Type': ['Production', 'Staging', 'Testing', 'Development'],
        'CPU_Cores': [32, 16, 8, 4],
        'RAM_GB': [128, 64, 32, 16],
        'Storage_GB': [2000, 1000, 500, 200],
        'IOPS_Requirement': [10000, 5000, 3000, 1000],
        'Peak_Connections': [500, 200, 100, 50],
        'Daily_Usage_Hours': [24, 16, 12, 8],
        'Workload_Pattern': ['read_heavy', 'balanced', 'balanced', 'write_heavy'],
        'Read_Write_Ratio': [80, 70, 60, 40],
        'Multi_AZ_Writer': [True, True, False, False],
        'Multi_AZ_Readers': [True, False, False, False],
        'Num_Readers': [2, 1, 1, 0],
        'Storage_Encrypted': [True, True, False, False],
        'Backup_Retention_Days': [30, 14, 7, 7],
        'Auto_Storage_Scaling': [True, True, True, False]
    }
    
    return pd.DataFrame(template_data)

def process_cluster_data(df: pd.DataFrame) -> Dict:
    """Process uploaded cluster data"""
    
    environments = {}
    
    for _, row in df.iterrows():
        env_name = str(row['Environment_Name'])
        
        environments[env_name] = {
            'cpu_cores': int(row.get('CPU_Cores', 4)),
            'ram_gb': int(row.get('RAM_GB', 16)),
            'storage_gb': int(row.get('Storage_GB', 500)),
            'iops_requirement': int(row.get('IOPS_Requirement', 3000)),
            'peak_connections': int(row.get('Peak_Connections', 100)),
            'daily_usage_hours': int(row.get('Daily_Usage_Hours', 24)),
            'workload_pattern': str(row.get('Workload_Pattern', 'balanced')),
            'read_write_ratio': int(row.get('Read_Write_Ratio', 70)),
            'environment_type': str(row.get('Environment_Type', 'Production')).lower(),
            'multi_az_writer': bool(row.get('Multi_AZ_Writer', True)),
            'multi_az_readers': bool(row.get('Multi_AZ_Readers', False)),
            'num_readers': int(row.get('Num_Readers', 1)) if pd.notna(row.get('Num_Readers')) else None,
            'storage_encrypted': bool(row.get('Storage_Encrypted', True)),
            'backup_retention': int(row.get('Backup_Retention_Days', 7)),
            'auto_storage_scaling': bool(row.get('Auto_Storage_Scaling', True))
        }
    
    return environments

def show_cluster_upload_summary(environment_specs: Dict):
    """Show summary of uploaded cluster configurations"""
    
    st.markdown("#### 📊 Cluster Configuration Summary")
    
    summary_data = []
    for env_name, specs in environment_specs.items():
        summary_data.append({
            'Environment': env_name,
            'Type': specs['environment_type'].title(),
            'Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Storage': f"{specs['storage_gb']} GB ({specs['iops_requirement']} IOPS)",
            'Workload': f"{specs['workload_pattern']} ({specs['read_write_ratio']}% reads)",
            'Writer Multi-AZ': '✅' if specs['multi_az_writer'] else '❌',
            'Readers': f"{specs.get('num_readers', 'Auto')} ({'Multi-AZ' if specs['multi_az_readers'] else 'Single-AZ'})"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def show_cluster_configuration_preview(recommendations: Dict):
    """Show preview of cluster configuration recommendations"""
    
    st.markdown("#### 🎯 Cluster Configuration Preview")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🏢 {env_name} Cluster Configuration"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Writer Configuration**")
                st.write(f"Instance: {rec['writer']['instance_class']}")
                st.write(f"Multi-AZ: {'✅ Yes' if rec['writer']['multi_az'] else '❌ No'}")
                st.write(f"Resources: {rec['writer']['cpu_cores']} cores, {rec['writer']['ram_gb']} GB RAM")
            
            with col2:
                st.markdown("**Reader Configuration**")
                reader_count = rec['readers']['count']
                if reader_count > 0:
                    st.write(f"Count: {reader_count} replicas")
                    st.write(f"Instance: {rec['readers']['instance_class']}")
                    st.write(f"Multi-AZ: {'✅ Yes' if rec['readers']['multi_az'] else '❌ No'}")
                else:
                    st.write("No read replicas")
                    st.write("Single writer configuration")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'✅ Yes' if storage['encrypted'] else '❌ No'}")

# Enhanced Cost Analysis Functions
def show_enhanced_cost_analysis():
    """Show enhanced cost analysis with Writer/Reader breakdown"""
    
    st.markdown("### 💰 Enhanced Cost Analysis")
    
    if not hasattr(st.session_state, 'enhanced_analysis_results'):
        st.warning("Please run the enhanced analysis first.")
        return
    
    results = st.session_state.enhanced_analysis_results
    
    # Overall cost metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Monthly Cost",
            f"${results['monthly_aws_cost']:,.0f}",
            delta=f"${results['annual_aws_cost']:,.0f}/year"
        )
    
    with col2:
        # Calculate writer vs reader costs
        total_writer_cost = sum([env['writer_instance_cost'] for env in results['environment_costs'].values()])
        total_reader_cost = sum([env['reader_costs'] for env in results['environment_costs'].values()])
        
        st.metric(
            "Writer Instances",
            f"${total_writer_cost:,.0f}/month",
            delta=f"{(total_writer_cost/results['monthly_aws_cost']*100):.1f}% of total"
        )
    
    with col3:
        st.metric(
            "Reader Instances",
            f"${total_reader_cost:,.0f}/month",
            delta=f"{(total_reader_cost/results['monthly_aws_cost']*100):.1f}% of total" if total_reader_cost > 0 else "No readers"
        )
    
    with col4:
        total_storage_cost = sum([env['storage_cost'] for env in results['environment_costs'].values()])
        st.metric(
            "Storage & I/O",
            f"${total_storage_cost:,.0f}/month",
            delta=f"{(total_storage_cost/results['monthly_aws_cost']*100):.1f}% of total"
        )
    
    # Detailed environment breakdown
    st.markdown("#### 🏢 Environment Cost Breakdown")
    
    for env_name, costs in results['environment_costs'].items():
        with st.expander(f"💵 {env_name} - Total: ${costs['total_monthly']:,.0f}/month"):
            
            # Create cost breakdown chart
            cost_categories = ['Writer Instance', 'Reader Instances', 'Storage', 'Backup', 'Monitoring', 'Cross-AZ Transfer']
            cost_values = [
                costs['writer_instance_cost'],
                costs['reader_costs'],
                costs['storage_cost'],
                costs['backup_cost'],
                costs['monitoring_cost'],
                costs['cross_az_cost']
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=cost_categories,
                values=cost_values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title=f"{env_name} Cost Distribution",
                height=400
            )
            
             # FIXED: Added unique key using counter
            st.plotly_chart(fig, use_container_width=True, key=f"enhanced_cost_breakdown_{chart_counter}")
            chart_counter += 1
            
                       
            # Detailed breakdown table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Instance Costs**")
                st.write(f"Writer: ${costs['writer_instance_cost']:,.0f}")
                if costs['reader_count'] > 0:
                    st.write(f"Readers ({costs['reader_count']}): ${costs['reader_costs']:,.0f}")
                    st.write(f"Reader cost per instance: ${costs['reader_costs']/costs['reader_count']:,.0f}")
                else:
                    st.write("Readers: No read replicas")
            
            with col2:
                st.markdown("**Storage Breakdown**")
                storage_breakdown = costs['storage_breakdown']
                st.write(f"Base Storage: ${storage_breakdown['base_storage_cost']:,.0f}")
                if storage_breakdown['iops_cost'] > 0:
                    st.write(f"Provisioned IOPS: ${storage_breakdown['iops_cost']:,.0f}")
                if storage_breakdown['io_request_cost'] > 0:
                    st.write(f"I/O Requests: ${storage_breakdown['io_request_cost']:,.0f}")
                st.write(f"Storage Type: {storage_breakdown['storage_type'].upper()}")
                st.write(f"Size: {storage_breakdown['storage_size_gb']:,} GB")
            
            # Configuration details
            st.markdown("**Configuration Details**")
            writer_config = costs['writer_config']
            reader_config = costs['reader_config']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("*Writer:*")
                st.write(f"Instance: {writer_config['instance_class']}")
                st.write(f"Multi-AZ: {'✅ Yes' if writer_config['multi_az'] else '❌ No'}")
            
            with col2:
                st.markdown("*Readers:*")
                if reader_config['count'] > 0:
                    st.write(f"Count: {reader_config['count']}")
                    st.write(f"Instance: {reader_config['instance_class']}")
                    st.write(f"Multi-AZ: {'✅ Yes' if reader_config['multi_az'] else '❌ No'}")
                else:
                    st.write("No read replicas configured")

# Enhanced Analysis Runner
def run_enhanced_migration_analysis():
    """Run enhanced migration analysis with Writer/Reader support"""
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedMigrationAnalyzer()
        
        # Step 1: Calculate enhanced recommendations
        st.write("📊 Calculating cluster recommendations...")
        recommendations = analyzer.calculate_enhanced_instance_recommendations(st.session_state.environment_specs)
        st.session_state.enhanced_recommendations = recommendations
        
        # Step 2: Calculate enhanced costs
        st.write("💰 Analyzing cluster costs...")
        cost_analysis = analyzer.calculate_enhanced_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.enhanced_analysis_results = cost_analysis
        
        # Step 3: Generate cost comparison
        st.write("📈 Generating cost comparisons...")
        generate_enhanced_cost_visualizations()
        
        st.success("✅ Enhanced analysis complete!")
        
        # Show summary
        show_enhanced_analysis_summary()
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        st.code(str(e))

def show_enhanced_analysis_summary():
    """Show enhanced analysis summary"""
    
    st.markdown("#### 🎯 Enhanced Analysis Summary")
    
    results = st.session_state.enhanced_analysis_results
    recommendations = st.session_state.enhanced_recommendations
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_writers = len(recommendations)
    total_readers = sum([rec['readers']['count'] for rec in recommendations.values()])
    multi_az_envs = sum([1 for rec in recommendations.values() if rec['writer']['multi_az']])
    
    with col1:
        st.metric("Total Environments", total_writers)
    
    with col2:
        st.metric("Total Read Replicas", total_readers)
    
    with col3:
        st.metric("Multi-AZ Environments", multi_az_envs)
    
    with col4:
        avg_monthly_cost = results['monthly_aws_cost'] / total_writers
        st.metric("Avg Cost per Environment", f"${avg_monthly_cost:,.0f}/month")
    
    # Configuration overview
    st.markdown("#### 📋 Configuration Overview")
    
    config_summary = []
    for env_name, rec in recommendations.items():
        config_summary.append({
            'Environment': env_name,
            'Writer': f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})",
            'Readers': f"{rec['readers']['count']} x {rec['readers']['instance_class']}" if rec['readers']['count'] > 0 else "None",
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'IOPS': f"{rec['storage']['iops']:,}",
            'Workload': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })
    
    config_df = pd.DataFrame(config_summary)
    st.dataframe(config_df, use_container_width=True)

def generate_enhanced_cost_visualizations():
    """Generate enhanced cost visualizations"""
    
    results = st.session_state.enhanced_analysis_results
    
    # Writer vs Reader cost comparison
    env_names = list(results['environment_costs'].keys())
    writer_costs = [results['environment_costs'][env]['writer_instance_cost'] for env in env_names]
    reader_costs = [results['environment_costs'][env]['reader_costs'] for env in env_names]
    storage_costs = [results['environment_costs'][env]['storage_cost'] for env in env_names]
    
# Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Writer Instance', x=env_names, y=writer_costs, marker_color='#3182ce'))
    fig.add_trace(go.Bar(name='Reader Instances', x=env_names, y=reader_costs, marker_color='#38a169'))
    fig.add_trace(go.Bar(name='Storage & I/O', x=env_names, y=storage_costs, marker_color='#d69e2e'))
    
    fig.update_layout(
        title='Monthly Cost Breakdown by Environment',
        xaxis_title='Environment',
        yaxis_title='Monthly Cost ($)',
        barmode='stack',
        height=500
    )
    
    st.session_state.enhanced_cost_chart = fig

# Integration with main application
def integrate_enhanced_cluster_features():
    """Integration instructions for enhanced cluster features"""
    
    # Replace show_environment_setup() with show_enhanced_environment_setup_with_cluster_config()
    # Replace run_migration_analysis() with run_enhanced_migration_analysis()
    # Add show_enhanced_cost_analysis() to the results dashboard
    
    # Add to session state initialization:
    # 'enhanced_recommendations': None,
    # 'enhanced_analysis_results': None,
    # 'enhanced_cost_chart': None
    
    pass

def show_storage_auto_calculator():
    """Helper to auto-calculate environment storage"""
    
    if not st.session_state.migration_params:
        st.warning("Please configure migration parameters first")
        return
    
    migration_data_size = st.session_state.migration_params.get('data_size_gb', 1000)
    
    st.markdown("### 🤖 Storage Auto-Calculator")
    st.info(f"Base migration data size: {migration_data_size:,} GB")
    
    # Environment multipliers
    st.markdown("#### Environment Size Multipliers")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prod_mult = st.slider("Production", 0.8, 2.0, 1.2, 0.1, key="prod_mult")
    with col2:
        stage_mult = st.slider("Staging", 0.5, 1.5, 0.8, 0.1, key="stage_mult")
    with col3:
        test_mult = st.slider("Testing/QA", 0.3, 1.0, 0.6, 0.1, key="test_mult")
    with col4:
        dev_mult = st.slider("Development", 0.1, 0.8, 0.3, 0.1, key="dev_mult")
    
    # Calculate and show recommendations
    st.markdown("#### 📊 Recommended Storage Sizes")
    
    recommendations = {
        'Production': int(migration_data_size * prod_mult),
        'Staging': int(migration_data_size * stage_mult),
        'Testing': int(migration_data_size * test_mult),
        'QA': int(migration_data_size * test_mult),
        'Development': int(migration_data_size * dev_mult)
    }
    
    rec_data = []
    for env_type, recommended_gb in recommendations.items():
        growth_buffer = int(recommended_gb * 0.3)  # 30% growth buffer
        final_size = recommended_gb + growth_buffer
        
        rec_data.append({
            'Environment Type': env_type,
            'Base Size (GB)': f"{recommended_gb:,}",
            'Growth Buffer (GB)': f"{growth_buffer:,}",
            'Recommended Total (GB)': f"{final_size:,}"
        })
    
    rec_df = pd.DataFrame(rec_data)
    st.dataframe(rec_df, use_container_width=True)
    
    # Option to apply to current environments
    if st.session_state.environment_specs:
        st.markdown("#### ⚙️ Apply to Current Environments")
        
        if st.button("🎯 Auto-Calculate Storage for Current Environments"):
            updated_specs = {}
            
            for env_name, specs in st.session_state.environment_specs.items():
                env_type = specs.get('environment_type', env_name).lower()
                
                # Match environment type to multiplier
                if 'prod' in env_type:
                    multiplier = prod_mult
                elif 'stag' in env_type:
                    multiplier = stage_mult
                elif any(x in env_type for x in ['test', 'qa', 'uat']):
                    multiplier = test_mult
                elif 'dev' in env_type:
                    multiplier = dev_mult
                else:
                    multiplier = 0.5  # Default
                
                # Calculate new storage
                base_storage = int(migration_data_size * multiplier)
                recommended_storage = int(base_storage * 1.3)  # Add 30% buffer
                
                # Update specs
                updated_specs[env_name] = {
                    **specs,  # Keep existing configuration
                    'storage_gb': recommended_storage,
                    'original_storage_gb': specs.get('storage_gb', 0),
                    'auto_calculated': True
                }
            
            # Apply changes
            st.session_state.environment_specs = updated_specs
            st.success("✅ Storage auto-calculated and applied!")
            st.balloons()
            
            # Show what changed
            st.markdown("**Changes Applied:**")
            for env_name, specs in updated_specs.items():
                old_size = specs.get('original_storage_gb', 0)
                new_size = specs.get('storage_gb', 0)
                if old_size != new_size:
                    st.write(f"• {env_name}: {old_size:,} GB → {new_size:,} GB")
# ADD this helper function to check data compatibility:

def export_recommendations_to_csv(optimization_results):
    """Export optimization results to CSV format"""
    
    export_data = []
    
    for env_name, optimization in optimization_results.items():
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        costs = optimization['cost_analysis']['monthly_breakdown']
        
        export_data.append({
            'Environment': env_name,
            'Environment_Type': optimization['environment_type'],
            'Optimization_Score': f"{optimization['optimization_score']:.1f}",
            'Writer_Instance': writer['instance_class'],
            'Writer_vCPUs': writer['specs'].vcpu,
            'Writer_Memory_GB': writer['specs'].memory_gb,
            'Writer_Monthly_Cost': f"${writer['monthly_cost']:,.0f}",
            'Writer_Annual_Cost': f"${writer['annual_cost']:,.0f}",
            'Reader_Count': reader['count'],
            'Reader_Instance': reader['instance_class'] if reader['count'] > 0 else 'None',
            'Reader_Monthly_Cost_Total': f"${reader['total_monthly_cost']:,.0f}",
            'Reader_Annual_Cost_Total': f"${reader['total_annual_cost']:,.0f}",
            'Storage_Monthly_Cost': f"${costs['storage']:,.0f}",
            'Total_Monthly_Cost': f"${costs['total']:,.0f}",
            'Total_Annual_Cost': f"${costs['total'] * 12:,.0f}",
            'Reserved_1Y_Savings': f"${optimization['cost_analysis']['reserved_instance_options']['1_year']['total_savings']:,.0f}",
            'Reserved_3Y_Savings': f"${optimization['cost_analysis']['reserved_instance_options']['3_year']['total_savings']:,.0f}",
            'Workload_Type': optimization['workload_analysis']['workload_type'],
            'Complexity_Score': f"{optimization['workload_analysis']['complexity_score']:.1f}",
            'Writer_Reasoning': writer['reasoning'],
            'Reader_Reasoning': reader['reasoning']
        })
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False)

def generate_optimization_report_pdf(optimization_results):
    """Generate detailed PDF report for optimization results"""
    
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Reader/Writer Optimization Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    total_monthly = sum([opt['cost_analysis']['monthly_breakdown']['total'] for opt in optimization_results.values()])
    avg_score = sum([opt['optimization_score'] for opt in optimization_results.values()]) / len(optimization_results)
    
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(f"Total Environments Analyzed: {len(optimization_results)}", styles['Normal']))
    story.append(Paragraph(f"Total Monthly Cost: ${total_monthly:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Total Annual Cost: ${total_monthly * 12:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Average Optimization Score: {avg_score:.1f}/100", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Detailed recommendations for each environment
    for env_name, optimization in optimization_results.items():
        story.append(Paragraph(f"{env_name} Environment", styles['Heading2']))
        
        # Create summary table
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        
        env_data = [
            ['Component', 'Configuration', 'Monthly Cost', 'Annual Cost'],
            ['Writer Instance', 
             f"{writer['instance_class']} ({writer['specs'].vcpu} vCPU, {writer['specs'].memory_gb}GB)",
             f"${writer['monthly_cost']:,.0f}",
             f"${writer['annual_cost']:,.0f}"],
            ['Read Replicas',
             f"{reader['count']} x {reader['instance_class']}" if reader['count'] > 0 else "None",
             f"${reader['total_monthly_cost']:,.0f}",
             f"${reader['total_annual_cost']:,.0f}"],
            ['Total Environment',
             f"Score: {optimization['optimization_score']:.1f}/100",
             f"${optimization['cost_analysis']['monthly_breakdown']['total']:,.0f}",
             f"${optimization['cost_analysis']['monthly_breakdown']['total'] * 12:,.0f}"]
        ]
        
        env_table = Table(env_data, colWidths=[1.5*inch, 2.5*inch, 1.2*inch, 1.2*inch])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        
        story.append(env_table)
        story.append(Spacer(1, 15))
        
        # Add reasoning
        story.append(Paragraph("Writer Reasoning:", styles['Heading3']))
        story.append(Paragraph(writer['reasoning'], styles['Normal']))
        story.append(Paragraph("Reader Reasoning:", styles['Heading3']))
        story.append(Paragraph(reader['reasoning'], styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Step 5: Add optimization preferences to the OptimizedReaderWriterAnalyzer class
# Add this method to the OptimizedReaderWriterAnalyzer class:

def set_optimization_preferences(self, goal='balanced', consider_reserved=True, environment_priority='production_first'):
    """Set optimization preferences"""
    self.optimization_goal = goal
    self.consider_reserved_instances = consider_reserved
    self.environment_priority = environment_priority
    
    # Adjust scoring weights based on goal
    if goal == 'cost_efficiency':
        self.performance_weight = 0.2
        self.cost_weight = 0.6
        self.suitability_weight = 0.2
    elif goal == 'performance':
        self.performance_weight = 0.7
        self.cost_weight = 0.1
        self.suitability_weight = 0.2
    else:  # balanced
        self.performance_weight = 0.4
        self.cost_weight = 0.3
        self.suitability_weight = 0.3

def is_enhanced_environment_data(environment_specs):
    """Check if environment specs contain enhanced cluster data"""
    if not environment_specs:
        return False
    
    sample_spec = next(iter(environment_specs.values()))
    enhanced_fields = ['workload_pattern', 'read_write_ratio', 'multi_az_writer']
    
    return any(field in sample_spec for field in enhanced_fields)


def show_enhanced_environment_analysis():
    """Show enhanced environment analysis with Writer/Reader details"""
    
    st.markdown("### 🏢 Enhanced Environment Analysis")
    
    if not hasattr(st.session_state, 'enhanced_recommendations') or not st.session_state.enhanced_recommendations:
        st.warning("Enhanced recommendations not available.")
        return
    
    recommendations = st.session_state.enhanced_recommendations
    environment_specs = st.session_state.environment_specs
    
    # Environment comparison with cluster details
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs.get(env_name, {})
        
        # Writer configuration
        writer_config = f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})"
        
        # Reader configuration
        reader_count = rec['readers']['count']
        if reader_count > 0:
            reader_config = f"{reader_count} x {rec['readers']['instance_class']}"
        else:
            reader_config = "No readers"
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec['environment_type'].title(),
            'Current Resources': f"{specs.get('cpu_cores', 'N/A')} cores, {specs.get('ram_gb', 'N/A')} GB RAM",
            'Writer Instance': writer_config,
            'Read Replicas': reader_config,
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'Workload Pattern': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })
    
    env_df = pd.DataFrame(env_comparison_data)
    st.dataframe(env_df, use_container_width=True)
    
    # Detailed environment insights
    st.markdown("#### 💡 Environment Insights")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🔍 {env_name} Environment Details"):
            specs = environment_specs.get(env_name, {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Configuration**")
                st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                st.write(f"IOPS Requirement: {specs.get('iops_requirement', 'N/A')}")
                st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
            
            with col2:
                st.markdown("**Writer Configuration**")
                writer = rec['writer']
                st.write(f"Instance: {writer['instance_class']}")
                st.write(f"Multi-AZ: {'Yes' if writer['multi_az'] else 'No'}")
                st.write(f"CPU Cores: {writer['cpu_cores']}")
                st.write(f"RAM: {writer['ram_gb']} GB")
                
                st.markdown("**Reader Configuration**")
                readers = rec['readers']
                if readers['count'] > 0:
                    st.write(f"Count: {readers['count']}")
                    st.write(f"Instance: {readers['instance_class']}")
                    st.write(f"Multi-AZ: {'Yes' if readers['multi_az'] else 'No'}")
                else:
                    st.write("No read replicas")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'Yes' if storage['encrypted'] else 'No'}")
                st.write(f"Backup Retention: {storage['backup_retention_days']} days")
                
                st.markdown("**Workload Characteristics**")
                st.write(f"Pattern: {rec['workload_pattern']}")
                st.write(f"Read/Write Ratio: {rec['read_write_ratio']}% reads")
                st.write(f"Peak Connections: {rec['connections']}")
            
            # Optimization recommendations
            st.markdown("**💡 Optimization Notes**")
            
            if rec['environment_type'] == 'production':
                st.success("✅ Production-grade configuration with high availability")
                if readers['count'] > 0:
                    st.info(f"📊 {readers['count']} read replicas will help distribute read load")
            elif rec['environment_type'] == 'development':
                st.info("💡 Cost-optimized configuration for development")
                if readers['count'] == 0:
                    st.info("💰 No read replicas to minimize development costs")
            
            if rec['workload_pattern'] == 'read_heavy' and readers['count'] > 0:
                st.success(f"📈 Read-heavy workload well-suited for {readers['count']} read replicas")
            elif rec['workload_pattern'] == 'read_heavy' and readers['count'] == 0:
                st.warning("⚠️ Read-heavy workload might benefit from read replicas")
            
            if storage['type'] == 'io2':
                st.info("⚡ High-performance io2 storage for demanding IOPS requirements")
            elif storage['type'] == 'gp3':
                st.info("⚖️ Balanced gp3 storage for general-purpose workloads")
    
def show_enhanced_storage_auto_calculator():
    """Enhanced storage auto-calculator with validation"""
    
    if not st.session_state.migration_params:
        st.warning("Please configure migration parameters first")
        return
    
    migration_data_size = st.session_state.migration_params.get('data_size_gb', 1000)
    storage_manager = StorageConfigurationManager()
    
    st.markdown("### 🤖 Intelligent Storage Calculator")
    
    # Show migration data context
    st.info(f"📊 Base migration data size: **{migration_data_size:,} GB**")
    
    # Environment multiplier configuration
    st.markdown("#### 🎛️ Environment Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Environment Multipliers:**")
        prod_mult = st.slider("Production", 0.8, 3.0, 1.5, 0.1, 
                             help="Production typically needs more storage for safety")
        stage_mult = st.slider("Staging", 0.5, 2.0, 1.0, 0.1,
                              help="Staging usually matches production data size")
        test_mult = st.slider("Testing/QA", 0.2, 1.5, 0.7, 0.1,
                             help="Testing with subset of production data")
        dev_mult = st.slider("Development", 0.1, 1.0, 0.3, 0.1,
                            help="Development with minimal data set")
    
    with col2:
        st.markdown("**Additional Factors:**")
        growth_buffer = st.slider("Growth Buffer (%)", 10, 100, 30, 5,
                                 help="Additional storage for future growth") / 100
        replication_overhead = st.slider("Replication Overhead (%)", 5, 30, 10, 5,
                                       help="Extra storage for logs, backups, etc.") / 100
        
        include_compression = st.checkbox("Include Compression Factor", value=True,
                                        help="Consider 20-30% compression savings")
        compression_factor = 0.7 if include_compression else 1.0
    
    # Calculate recommendations
    st.markdown("#### 📊 Storage Recommendations")
    
    environments = ['Production', 'Staging', 'Testing', 'QA', 'Development']
    multipliers = {'Production': prod_mult, 'Staging': stage_mult, 'Testing': test_mult, 
                   'QA': test_mult, 'Development': dev_mult}
    
    recommendations_data = []
    
    for env_type in environments:
        base_storage = int(migration_data_size * multipliers[env_type] * compression_factor)
        growth_storage = int(base_storage * growth_buffer)
        replication_storage = int(base_storage * replication_overhead)
        total_storage = base_storage + growth_storage + replication_storage
        
        recommendations_data.append({
            'Environment': env_type,
            'Base Storage (GB)': f"{base_storage:,}",
            'Growth Buffer (GB)': f"{growth_storage:,}",
            'Replication Overhead (GB)': f"{replication_storage:,}",
            'Total Recommended (GB)': f"{total_storage:,}",
            'Multiplier': f"{multipliers[env_type]}x"
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    st.dataframe(recommendations_df, use_container_width=True)
    
    # Apply to current environments
    if st.session_state.environment_specs:
        st.markdown("#### ⚙️ Apply to Current Environments")
        
        # Show current vs recommended
        current_analysis = []
        
        for env_name, specs in st.session_state.environment_specs.items():
            env_type = specs.get('environment_type', env_name).lower()
            current_storage = specs.get('storage_gb', 0)
            
            # Match to multiplier
            if 'prod' in env_type:
                multiplier = prod_mult
            elif 'stag' in env_type:
                multiplier = stage_mult
            elif any(x in env_type for x in ['test', 'qa', 'uat']):
                multiplier = test_mult
            elif 'dev' in env_type:
                multiplier = dev_mult
            else:
                multiplier = 1.0
            
            # Calculate recommended
            base_storage = int(migration_data_size * multiplier * compression_factor)
            total_recommended = int(base_storage * (1 + growth_buffer + replication_overhead))
            
            difference = total_recommended - current_storage
            status = "✅ Good" if abs(difference) < current_storage * 0.2 else "⚠️ Review" if difference > 0 else "💰 Over-provisioned"
            
            current_analysis.append({
                'Environment': env_name,
                'Current (GB)': f"{current_storage:,}",
                'Recommended (GB)': f"{total_recommended:,}",
                'Difference (GB)': f"{difference:+,}",
                'Status': status
            })
        
        analysis_df = pd.DataFrame(current_analysis)
        st.dataframe(analysis_df, use_container_width=True)
        
        # Apply changes button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Apply Recommended Storage", type="primary"):
                updated_specs = {}
                
                for env_name, specs in st.session_state.environment_specs.items():
                    env_type = specs.get('environment_type', env_name).lower()
                    
                    # Match to multiplier
                    if 'prod' in env_type:
                        multiplier = prod_mult
                    elif 'stag' in env_type:
                        multiplier = stage_mult
                    elif any(x in env_type for x in ['test', 'qa', 'uat']):
                        multiplier = test_mult
                    elif 'dev' in env_type:
                        multiplier = dev_mult
                    else:
                        multiplier = 1.0
                    
                    # Calculate new storage
                    base_storage = int(migration_data_size * multiplier * compression_factor)
                    total_recommended = int(base_storage * (1 + growth_buffer + replication_overhead))
                    
                    updated_specs[env_name] = {
                        **specs,
                        'storage_gb': total_recommended,
                        'original_storage_gb': specs.get('storage_gb', 0),
                        'auto_calculated': True,
                        'calculation_details': {
                            'base_storage': base_storage,
                            'growth_buffer': int(base_storage * growth_buffer),
                            'replication_overhead': int(base_storage * replication_overhead),
                            'multiplier': multiplier,
                            'compression_factor': compression_factor
                        }
                    }
                
                st.session_state.environment_specs = updated_specs
                st.success("✅ Storage recommendations applied!")
                st.balloons()
        
        with col2:
            if st.button("🔍 Validate Current Storage"):
                validation = storage_manager.validate_storage_configuration(
                    migration_data_size, st.session_state.environment_specs
                )
                
                if validation['is_valid']:
                    st.success("✅ Storage configuration is valid!")
                else:
                    st.error("❌ Storage configuration issues found:")
                    for error in validation['errors']:
                        st.error(f"• {error}")
                
                if validation['warnings']:
                    st.warning("⚠️ Storage warnings:")
                    for warning in validation['warnings']:
                        st.warning(f"• {warning}")


def validate_growth_analysis_functions():
    """Validate that all growth analysis functions are properly defined"""
    
    required_functions = [
        'show_basic_cost_summary',
        'show_growth_analysis_dashboard', 
        'show_risk_assessment_tab',
        'show_environment_analysis_tab',
        'show_visualizations_tab',
        'show_ai_insights_tab',
        'show_timeline_analysis_tab',
        'create_growth_projection_charts'
    ]
    
    missing_functions = []
    
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        st.error(f"❌ Missing functions: {', '.join(missing_functions)}")
        return False
    else:
        st.success("✅ All growth analysis functions are properly defined!")
        return True
def validate_storage_consistency():
    """Simple storage validation between migration and environment configs"""
    
    if not st.session_state.migration_params or not st.session_state.environment_specs:
        return
    
    # Get migration data size
    migration_data_size = st.session_state.migration_params.get('data_size_gb', 0)
    
    # Calculate total environment storage
    total_env_storage = 0
    env_details = []
    
    for env_name, specs in st.session_state.environment_specs.items():
        env_storage = specs.get('storage_gb', 0)
        total_env_storage += env_storage
        
        env_details.append({
            'Environment': env_name,
            'Storage_GB': env_storage,
            'Env_Type': specs.get('environment_type', 'Unknown')
        })
    
    # Show analysis
    st.markdown("#### 💾 Storage Configuration Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Migration Data Size", f"{migration_data_size:,} GB")
    
    with col2:
        st.metric("Total Environment Storage", f"{total_env_storage:,} GB")
    
    with col3:
        if migration_data_size > 0:
            multiplier = total_env_storage / migration_data_size
            st.metric("Storage Multiplier", f"{multiplier:.1f}x")
            
            # Simple validation
            if multiplier > 4:
                st.warning("⚠️ Environment storage is much larger than migration data")
            elif multiplier < 0.8:
                st.error("❌ Environment storage might be too small")
            else:
                st.success("✅ Storage sizing looks reasonable")
        else:
            st.info("ℹ️ Configure migration data size for analysis")
    
    # Show environment breakdown
    if env_details:
        st.markdown("**Environment Storage Breakdown:**")
        df = pd.DataFrame(env_details)
        st.dataframe(df, use_container_width=True)

# Add this to your main function to test
def test_growth_setup():
    """Test the growth analysis setup"""
    st.markdown("### 🧪 Growth Analysis Setup Test")
    
    if st.button("🔍 Validate Setup"):
        validate_growth_analysis_functions()
        
        # Test growth analyzer
        try:
            analyzer = GrowthAwareCostAnalyzer()
            st.success("✅ GrowthAwareCostAnalyzer initialized successfully!")
        except Exception as e:
            st.error(f"❌ GrowthAwareCostAnalyzer error: {str(e)}")
        
        # Test session state
        if hasattr(st.session_state, 'growth_analysis'):
            st.info("📊 Growth analysis data available in session state")
        else:
            st.warning("⚠️ No growth analysis data in session state yet")




if __name__ == "__main__":
    main()