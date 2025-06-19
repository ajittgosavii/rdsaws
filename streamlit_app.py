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

# PDF Generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from dataclasses import dataclass

# ===========================
# PAGE CONFIGURATION AND CSS
# ===========================

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

# ===========================
# CORE DATA CLASSES
# ===========================

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

# ===========================
# DATABASE ENGINE DEFINITIONS
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
        migration_type = cls.get_migration_type(source_engine, target_engine)
        
        if migration_type == "heterogeneous":
            if 'oracle' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.5
            elif 'sql-server' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.0
            else:
                return source_complexity * 1.5
        else:
            return source_complexity

# ===========================
# AWS PRICING API (CONSOLIDATED)
# ===========================

class AWSPricingAPI:
    """Unified AWS Pricing API with real-time and fallback support"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        
        # Initialize boto3 pricing client
        try:
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        except Exception as e:
            print(f"Warning: Could not initialize AWS pricing client: {e}")
            self.pricing_client = None
    
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get RDS pricing from AWS API with fallback"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try to get real pricing from AWS
            real_pricing = self._fetch_real_aws_pricing(region, engine, instance_class, multi_az)
            if real_pricing:
                self.cache[cache_key] = real_pricing
                return real_pricing
        except Exception as e:
            print(f"Error fetching real AWS pricing: {e}")
        
        # Fallback to static pricing
        return self._get_fallback_pricing(region, engine, instance_class, multi_az)
    
    def _fetch_real_aws_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool) -> Optional[Dict]:
        """Fetch real pricing from AWS Pricing API"""
        if not self.pricing_client:
            raise Exception("AWS pricing client not available")
        
        engine_mapping = {
            'postgres': 'PostgreSQL',
            'mysql': 'MySQL',
            'aurora-postgresql': 'Aurora PostgreSQL',
            'aurora-mysql': 'Aurora MySQL',
            'oracle-ee': 'Oracle',
            'oracle-se': 'Oracle',
            'sql-server': 'SQL Server'
        }
        
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonRDS',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_aws_region_name(region)},
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_class},
                    {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': engine_mapping.get(engine, 'PostgreSQL')},
                    {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Multi-AZ' if multi_az else 'Single-AZ'}
                ],
                MaxResults=10
            )
            
            if response.get('PriceList'):
                price_data = json.loads(response['PriceList'][0])
                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})
                
                for term_key, term_data in on_demand.items():
                    price_dimensions = term_data.get('priceDimensions', {})
                    for dim_key, dim_data in price_dimensions.items():
                        price_per_unit = float(dim_data.get('pricePerUnit', {}).get('USD', '0'))
                        
                        if price_per_unit > 0:
                            return {
                                'hourly': price_per_unit,
                                'storage_gb': 0.115,
                                'iops_gb': 0.10,
                                'io_request': 0.20,
                                'is_aurora': 'aurora' in engine,
                                'multi_az': multi_az,
                                'source': 'AWS Pricing API'
                            }
            return None
            
        except Exception as e:
            print(f"Error in AWS Pricing API call: {e}")
            return None
    
    def _get_aws_region_name(self, region_code: str) -> str:
        """Convert region code to AWS region name"""
        region_mapping = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)'
        }
        return region_mapping.get(region_code, 'US East (N. Virginia)')
    
    def _get_fallback_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool) -> Dict:
        """Fallback pricing when API is unavailable"""
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-postgresql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                }
            }
        }
        
        engine_pricing = pricing_data.get(region, {}).get(engine, {})
        instance_pricing = engine_pricing.get(instance_class, {
            'hourly': 0.5, 'hourly_multi_az': 1.0, 'storage_gb': 0.115, 'iops_gb': 0.10
        })
        
        hourly_cost = instance_pricing['hourly_multi_az'] if multi_az else instance_pricing['hourly']
        
        return {
            'hourly': hourly_cost,
            'storage_gb': instance_pricing['storage_gb'],
            'iops_gb': instance_pricing.get('iops_gb', 0.10),
            'io_request': instance_pricing.get('io_request', 0.20),
            'is_aurora': 'aurora' in engine,
            'multi_az': multi_az,
            'source': 'Fallback Pricing'
        }

    def calculate_cluster_cost(self, region: str, engine: str, 
                          writer_instance: str, writer_multi_az: bool,
                          reader_instances: List[Tuple[str, bool]]) -> float:
        """Calculate total cost for writer and readers"""
        total_cost = 0
    
        # Writer cost
        writer_pricing = self.get_rds_pricing(region, engine, writer_instance, writer_multi_az)
        total_cost += writer_pricing['hourly']
    
        # Reader costs
        for reader_instance, multi_az in reader_instances:
            reader_pricing = self.get_rds_pricing(region, engine, reader_instance, multi_az)
            total_cost += reader_pricing['hourly']
    
        return total_cost

# ===========================
# GROWTH ANALYSIS MODULE
# ===========================

class GrowthAwareCostAnalyzer:
    """Cost analyzer with 3-year growth projections"""
    
    def __init__(self):
        self.growth_factors = self._initialize_growth_factors()
    
    def _initialize_growth_factors(self):
        """Initialize growth impact factors for different cost components"""
        return {
            'compute': {
                'user_growth_factor': 0.8,
                'transaction_factor': 0.9,
                'data_factor': 0.3
            },
            'storage': {
                'data_growth_factor': 1.0,
                'user_growth_factor': 0.2,
                'transaction_factor': 0.4
            },
            'iops': {
                'transaction_factor': 0.95,
                'user_growth_factor': 0.7,
                'data_factor': 0.5
            },
            'network': {
                'user_growth_factor': 0.8,
                'transaction_factor': 0.6,
                'data_factor': 0.4
            }
        }
    
    def calculate_3_year_growth_projection(self, base_costs: Dict, migration_params: Dict) -> Dict:
        """Calculate 3-year cost projection with growth"""
        data_growth = migration_params.get('annual_data_growth', 15) / 100
        user_growth = migration_params.get('annual_user_growth', 25) / 100
        transaction_growth = migration_params.get('annual_transaction_growth', 20) / 100
        seasonality = migration_params.get('seasonality_factor', 1.2)
        
        projections = {}
        
        for year in range(4):  # Year 0 through Year 3
            year_multiplier = {
                'data': (1 + data_growth) ** year,
                'users': (1 + user_growth) ** year,
                'transactions': (1 + transaction_growth) ** year
            }
            
            year_costs = self._calculate_year_costs(
                base_costs, year_multiplier, seasonality, year
            )
            
            projections[f'year_{year}'] = year_costs
        
        scaling_recommendations = self._generate_scaling_recommendations(projections, migration_params)
        
        return {
            'yearly_projections': projections,
            'scaling_recommendations': scaling_recommendations,
            'growth_summary': self._generate_growth_summary(projections),
            'cost_optimization_opportunities': self._identify_growth_optimizations(projections)
        }
    
    def _calculate_year_costs(self, base_costs: Dict, multipliers: Dict, seasonality: float, year: int) -> Dict:
        """Calculate costs for a specific year with growth factors"""
        year_costs = {}
        
        for env_name, env_costs in base_costs['environment_costs'].items():
            compute_multiplier = (
                multipliers['users'] * self.growth_factors['compute']['user_growth_factor'] +
                multipliers['transactions'] * self.growth_factors['compute']['transaction_factor'] +
                multipliers['data'] * self.growth_factors['compute']['data_factor']
            ) / 3
            
            storage_multiplier = (
                multipliers['data'] * self.growth_factors['storage']['data_growth_factor'] +
                multipliers['users'] * self.growth_factors['storage']['user_growth_factor'] +
                multipliers['transactions'] * self.growth_factors['storage']['transaction_factor']
            ) / 3
            
            base_instance_cost = env_costs.get('instance_cost', env_costs.get('total_monthly', 0))
            base_storage_cost = env_costs.get('storage_cost', base_instance_cost * 0.2)
            
            adjusted_costs = {
                'instance_cost': base_instance_cost * compute_multiplier,
                'storage_cost': base_storage_cost * storage_multiplier,
                'backup_cost': (base_storage_cost * storage_multiplier) * 0.2,
                'monitoring_cost': 50 * compute_multiplier,
                'total_monthly': 0,
                'peak_monthly': 0,
                'resource_scaling': {
                    'compute_scaling': compute_multiplier,
                    'storage_scaling': storage_multiplier,
                }
            }
            
            adjusted_costs['total_monthly'] = sum([
                adjusted_costs['instance_cost'],
                adjusted_costs['storage_cost'],
                adjusted_costs['backup_cost'],
                adjusted_costs['monitoring_cost']
            ])
            
            adjusted_costs['peak_monthly'] = adjusted_costs['total_monthly'] * seasonality
            year_costs[env_name] = adjusted_costs
        
        total_monthly = sum([env['total_monthly'] for env in year_costs.values()])
        total_peak = sum([env['peak_monthly'] for env in year_costs.values()])
        
        return {
            'environment_costs': year_costs,
            'total_monthly': total_monthly,
            'total_annual': total_monthly * 12,
            'peak_monthly': total_peak,
            'peak_annual': total_peak * 12,
            'year': year,
            'growth_multipliers': multipliers
        }
    
    def _generate_scaling_recommendations(self, projections: Dict, migration_params: Dict) -> List[Dict]:
        """Generate scaling recommendations based on growth projections"""
        recommendations = []
        
        for year in range(1, 4):
            current_year = projections[f'year_{year}']
            previous_year = projections[f'year_{year-1}']
            
            cost_increase = ((current_year['total_annual'] / previous_year['total_annual']) - 1) * 100
            
            if cost_increase > 50:
                recommendations.append({
                    'year': year,
                    'type': 'Critical Scaling',
                    'description': f"Year {year} shows {cost_increase:.1f}% cost increase",
                    'action': 'Consider Reserved Instances and architecture optimization',
                    'priority': 'High',
                    'estimated_savings': current_year['total_annual'] * 0.3
                })
            elif cost_increase > 25:
                recommendations.append({
                    'year': year,
                    'type': 'Moderate Scaling',
                    'description': f"Year {year} shows {cost_increase:.1f}% cost increase",
                    'action': 'Plan for capacity increases and evaluate read replicas',
                    'priority': 'Medium',
                    'estimated_savings': current_year['total_annual'] * 0.15
                })
        
        return recommendations
    
    def _generate_growth_summary(self, projections: Dict) -> Dict:
        """Generate growth summary statistics"""
        year_0 = projections['year_0']['total_annual']
        year_3 = projections['year_3']['total_annual']
        
        total_growth = ((year_3 / year_0) - 1) * 100
        cagr = ((year_3 / year_0) ** (1/3) - 1) * 100
        
        return {
            'total_3_year_growth_percent': total_growth,
            'compound_annual_growth_rate': cagr,
            'year_0_cost': year_0,
            'year_3_cost': year_3,
            'total_3_year_investment': sum([
                projections[f'year_{year}']['total_annual'] for year in range(4)
            ]),
            'average_annual_cost': sum([
                projections[f'year_{year}']['total_annual'] for year in range(4)
            ]) / 4
        }
    
    def _identify_growth_optimizations(self, projections: Dict) -> List[Dict]:
        """Identify cost optimization opportunities"""
        optimizations = []
        year_3 = projections['year_3']
        
        # Calculate average scaling factors
        env_count = len(projections['year_0']['environment_costs'])
        if env_count > 0:
            avg_storage_growth = sum([
                env['resource_scaling']['storage_scaling'] 
                for env in year_3['environment_costs'].values()
            ]) / env_count
            
            if avg_storage_growth > 2.5:
                optimizations.append({
                    'opportunity': 'Data Lifecycle Management',
                    'description': f"Storage projected to grow {avg_storage_growth:.1f}x over 3 years",
                    'estimated_savings': year_3['total_annual'] * 0.15,
                    'implementation_effort': 'Medium',
                    'timeline': '3-6 months'
                })
        
        # Default optimization if none found
        if not optimizations:
            optimizations.append({
                'opportunity': 'Continuous Optimization',
                'description': 'Implement regular cost reviews and right-sizing exercises',
                'estimated_savings': year_3['total_annual'] * 0.10,
                'implementation_effort': 'Low',
                'timeline': 'Ongoing'
            })
        
        return optimizations

# ===========================
# READER/WRITER OPTIMIZER
# ===========================

class OptimizedReaderWriterAnalyzer:
    """Advanced Reader/Writer optimization with intelligent recommendations"""
    
    def __init__(self):
        self.instance_specs = self._initialize_instance_specs()
        self.pricing_data = self._initialize_pricing_data()
        
    def _initialize_instance_specs(self) -> Dict[str, InstanceSpecs]:
        """Initialize comprehensive instance specifications"""
        return {
            'db.t3.micro': InstanceSpecs('db.t3.micro', 2, 1, 'Low to Moderate', 87, 0.0255, ['development', 'testing'], 87, 1000),
            'db.t3.small': InstanceSpecs('db.t3.small', 2, 2, 'Low to Moderate', 174, 0.051, ['development', 'testing'], 174, 2000),
            'db.t3.medium': InstanceSpecs('db.t3.medium', 2, 4, 'Low to Moderate', 347, 0.102, ['development', 'small_production'], 347, 3000),
            'db.t3.large': InstanceSpecs('db.t3.large', 2, 8, 'Low to Moderate', 695, 0.204, ['development', 'small_production'], 695, 4000),
            'db.r5.large': InstanceSpecs('db.r5.large', 2, 16, 'Up to 10 Gbps', 693, 0.24, ['memory_intensive', 'analytics'], 1000, 7500),
            'db.r5.xlarge': InstanceSpecs('db.r5.xlarge', 4, 32, 'Up to 10 Gbps', 1387, 0.48, ['memory_intensive', 'production'], 2000, 10000),
            'db.r5.2xlarge': InstanceSpecs('db.r5.2xlarge', 8, 64, 'Up to 10 Gbps', 2775, 0.96, ['large_production', 'analytics'], 3000, 15000),
            'db.r5.4xlarge': InstanceSpecs('db.r5.4xlarge', 16, 128, '10 Gbps', 4750, 1.92, ['large_production', 'high_memory'], 5000, 25000),
        }
    
    def _initialize_pricing_data(self) -> Dict:
        """Initialize pricing data"""
        return {
            'us-east-1': {
                'reserved_1_year': {'discount': 0.35, 'upfront_ratio': 0.0},
                'reserved_3_year': {'discount': 0.55, 'upfront_ratio': 0.0},
                'multi_az_multiplier': 2.0,
            }
        }
    
    def optimize_cluster_configuration(self, environment_specs: Dict) -> Dict:
        """Generate optimized Reader/Writer recommendations"""
        optimized_recommendations = {}
        
        for env_name, specs in environment_specs.items():
            optimization = self._optimize_single_environment(env_name, specs)
            optimized_recommendations[env_name] = optimization
        
        return optimized_recommendations
    
    def _optimize_single_environment(self, env_name: str, specs: Dict) -> Dict:
        """Optimize configuration for a single environment"""
        cpu_cores = specs.get('cpu_cores', 4)
        ram_gb = specs.get('ram_gb', 16)
        storage_gb = specs.get('storage_gb', 500)
        iops_requirement = specs.get('iops_requirement', 3000)
        peak_connections = specs.get('peak_connections', 100)
        workload_pattern = specs.get('workload_pattern', 'balanced')
        read_write_ratio = specs.get('read_write_ratio', 70)
        environment_type = specs.get('environment_type', 'production')
        
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
        
        # Calculate costs
        cost_analysis = self._calculate_comprehensive_costs(
            writer_optimization, reader_optimization, storage_gb, iops_requirement, environment_type
        )
        
        return {
            'environment_name': env_name,
            'environment_type': environment_type,
            'workload_analysis': workload_analysis,
            'writer_optimization': writer_optimization,
            'reader_optimization': reader_optimization,
            'cost_analysis': cost_analysis,
            'optimization_score': self._calculate_optimization_score(
                workload_analysis, writer_optimization, reader_optimization, cost_analysis
            )
        }
    
    def _analyze_workload_characteristics(self, cpu_cores: int, ram_gb: int, 
                                        iops_requirement: int, peak_connections: int,
                                        workload_pattern: str, read_write_ratio: int) -> Dict:
        """Analyze workload characteristics"""
        cpu_intensity = min(100, (cpu_cores / 64) * 100)
        memory_intensity = min(100, (ram_gb / 768) * 100)
        io_intensity = min(100, (iops_requirement / 80000) * 100)
        connection_intensity = min(100, (peak_connections / 16000) * 100)
        
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
        
        complexity_score = (cpu_intensity + memory_intensity + io_intensity + connection_intensity) / 4
        
        # Calculate optimal reader count
        base_readers = {'read_heavy': 3, 'analytics': 2, 'balanced': 1, 'write_heavy': 0}
        complexity_adjustment = int(complexity_score / 30)
        connection_adjustment = int(peak_connections / 2000)
        optimal_reader_count = base_readers.get(workload_type, 1) + complexity_adjustment + connection_adjustment
        optimal_reader_count = min(5, max(0, optimal_reader_count))
        
        return {
            'cpu_intensity': cpu_intensity,
            'memory_intensity': memory_intensity,
            'io_intensity': io_intensity,
            'connection_intensity': connection_intensity,
            'complexity_score': complexity_score,
            'workload_type': workload_type,
            'read_scaling_factor': read_scaling_factor,
            'recommended_reader_count': optimal_reader_count,
            'performance_requirements': 'high_performance' if complexity_score > 80 else 'standard_performance'
        }
    
    def _optimize_writer_instance(self, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Writer instance selection"""
        performance_req = workload_analysis['performance_requirements']
        complexity_score = workload_analysis['complexity_score']
        
        # Filter and score instances
        suitable_instances = []
        for instance_class, specs in self.instance_specs.items():
            if self._is_instance_suitable_for_writer(specs, performance_req, environment_type, complexity_score):
                score = self._score_writer_instance(specs, workload_analysis, environment_type)
                suitable_instances.append((score, instance_class, specs))
        
        suitable_instances.sort(reverse=True)
        
        if suitable_instances:
            score, instance_class, specs = suitable_instances[0]
            return {
                'instance_class': instance_class,
                'specs': specs,
                'score': score,
                'multi_az': environment_type in ['production', 'staging'],
                'reasoning': f"Optimized for {workload_analysis['workload_type']} workload with {specs.vcpu} vCPUs and {specs.memory_gb}GB memory",
                'monthly_cost': self._calculate_instance_monthly_cost(specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(specs, environment_type) * 12
            }
        else:
            # Fallback
            fallback_specs = self.instance_specs['db.r5.large']
            return {
                'instance_class': 'db.r5.large',
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
        if environment_type == 'production' and specs.vcpu < 4:
            return False
        if environment_type == 'production' and 't3' in specs.instance_class and complexity_score > 50:
            return False
        return True
    
    def _score_writer_instance(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> float:
        """Score writer instance based on multiple criteria"""
        performance_score = min(100, (specs.vcpu * 10 + specs.memory_gb / 4 + specs.iops_capability / 500))
        cost_per_vcpu = specs.hourly_cost / specs.vcpu
        cost_efficiency = max(0, 100 - (cost_per_vcpu * 100))
        suitability_score = 100 if environment_type in specs.suitable_for else 60
        
        return (performance_score * 0.4 + cost_efficiency * 0.3 + suitability_score * 0.3)
    
    def _optimize_reader_configuration(self, writer_optimization: Dict, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Reader configuration"""
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
        reader_instance_class = self._calculate_optimal_reader_size(writer_specs, workload_analysis)
        reader_specs = self.instance_specs[reader_instance_class]
        
        single_reader_monthly_cost = self._calculate_instance_monthly_cost(reader_specs, environment_type, is_reader=True)
        total_monthly_cost = single_reader_monthly_cost * recommended_count
        
        return {
            'count': recommended_count,
            'instance_class': reader_instance_class,
            'specs': reader_specs,
            'single_reader_monthly_cost': single_reader_monthly_cost,
            'total_monthly_cost': total_monthly_cost,
            'total_annual_cost': total_monthly_cost * 12,
            'multi_az': environment_type == 'production',
            'reasoning': f"{recommended_count} read replica{'s' if recommended_count > 1 else ''} recommended for {workload_analysis['workload_type']} workload"
        }
    
    def _calculate_optimal_reader_size(self, writer_specs: InstanceSpecs, workload_analysis: Dict) -> str:
        """Calculate optimal reader instance size"""
        workload_type = workload_analysis['workload_type']
        
        if workload_type in ['read_heavy', 'analytics']:
            target_vcpu = writer_specs.vcpu
        else:
            target_vcpu = max(2, int(writer_specs.vcpu * 0.7))
        
        # Find best matching instance
        best_match = 'db.r5.large'
        best_score = 0
        
        for instance_class, specs in self.instance_specs.items():
            if specs.vcpu >= target_vcpu * 0.8:
                size_match_score = 100 - abs(specs.vcpu - target_vcpu) * 5
                cost_efficiency_score = 100 - (specs.hourly_cost * 10)
                overall_score = size_match_score * 0.7 + cost_efficiency_score * 0.3
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = instance_class
        
        return best_match
    
    def _calculate_instance_monthly_cost(self, specs: InstanceSpecs, environment_type: str, is_reader: bool = False) -> float:
        """Calculate monthly cost for an instance"""
        base_hourly_cost = specs.hourly_cost
        
        if environment_type in ['production', 'staging'] and not is_reader:
            base_hourly_cost *= self.pricing_data['us-east-1']['multi_az_multiplier']
        
        return base_hourly_cost * 730  # Monthly hours
    
    def _calculate_comprehensive_costs(self, writer_optimization: Dict, reader_optimization: Dict,
                                     storage_gb: int, iops_requirement: int, environment_type: str) -> Dict:
        """Calculate comprehensive cost analysis"""
        writer_monthly_cost = writer_optimization['monthly_cost']
        reader_monthly_cost = reader_optimization['total_monthly_cost']
        
        # Storage costs
        if iops_requirement > 16000:
            storage_type = 'io2'
            base_cost_per_gb = 0.125
            iops_cost = iops_requirement * 0.065
        elif iops_requirement > 3000:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            additional_iops = max(0, iops_requirement - 3000)
            iops_cost = additional_iops * 0.005
        else:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            iops_cost = 0
        
        storage_monthly_cost = (storage_gb * base_cost_per_gb) + iops_cost
        backup_monthly_cost = storage_gb * 0.095
        monitoring_monthly_cost = 50 if environment_type == 'production' else 20
        
        total_monthly_cost = (writer_monthly_cost + reader_monthly_cost + 
                            storage_monthly_cost + backup_monthly_cost + monitoring_monthly_cost)
        
        # Reserved Instance calculations
        reserved_1_year = self._calculate_reserved_instance_savings(writer_monthly_cost + reader_monthly_cost, 1)
        reserved_3_year = self._calculate_reserved_instance_savings(writer_monthly_cost + reader_monthly_cost, 3)
        
        return {
            'monthly_breakdown': {
                'writer_instance': writer_monthly_cost,
                'reader_instances': reader_monthly_cost,
                'storage': storage_monthly_cost,
                'backup': backup_monthly_cost,
                'monitoring': monitoring_monthly_cost,
                'total': total_monthly_cost
            },
            'annual_breakdown': {
                'total': total_monthly_cost * 12
            },
            'reserved_instance_options': {
                '1_year': reserved_1_year,
                '3_year': reserved_3_year
            }
        }
    
    def _calculate_reserved_instance_savings(self, monthly_instance_cost: float, years: int) -> Dict:
        """Calculate Reserved Instance savings"""
        discount = self.pricing_data['us-east-1']['reserved_1_year']['discount'] if years == 1 else self.pricing_data['us-east-1']['reserved_3_year']['discount']
        
        annual_on_demand = monthly_instance_cost * 12
        annual_reserved = annual_on_demand * (1 - discount)
        total_savings = (annual_on_demand - annual_reserved) * years
        
        return {
            'term_years': years,
            'discount_percentage': discount * 100,
            'annual_cost': annual_reserved,
            'total_savings': total_savings,
            'monthly_cost': annual_reserved / 12
        }
    
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
        
        return (performance_score * 0.4 + cost_efficiency_score * 0.4 + config_score * 0.2)

# ===========================
# NETWORK TRANSFER ANALYZER
# ===========================

class NetworkTransferAnalyzer:
    """Comprehensive network transfer analysis for AWS database migration"""
    
    def __init__(self):
        self.transfer_patterns = self._initialize_transfer_patterns()
    
    def _initialize_transfer_patterns(self) -> Dict:
        """Initialize supported network transfer patterns"""
        return {
            'internet_dms': {
                'name': 'Internet + DMS',
                'description': 'Standard internet connection with AWS Database Migration Service',
                'pros': ['Low setup cost', 'Quick to implement'],
                'cons': ['Variable bandwidth', 'Higher data transfer costs'],
                'complexity': 'Low'
            },
            'dx_dms': {
                'name': 'Direct Connect + DMS',
                'description': 'Dedicated connection with AWS Database Migration Service',
                'pros': ['Predictable bandwidth', 'Lower data transfer costs'],
                'cons': ['Higher setup cost', 'Longer setup time'],
                'complexity': 'Medium'
            },
            'vpn_dms': {
                'name': 'VPN + DMS',
                'description': 'Site-to-site VPN with AWS Database Migration Service',
                'pros': ['Secure connection', 'Quick setup'],
                'cons': ['Internet-dependent', 'Bandwidth limitations'],
                'complexity': 'Medium'
            }
        }
    
    def calculate_transfer_analysis(self, migration_params: Dict) -> Dict:
    """Calculate comprehensive transfer analysis for all patterns"""
    
    try:
        data_size_gb = migration_params.get('data_size_gb', 1000)
        region = migration_params.get('region', 'us-east-1')
        available_bandwidth_mbps = migration_params.get('bandwidth_mbps', 1000)
        security_requirements = migration_params.get('security_requirements', 'standard')
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        budget_constraints = migration_params.get('budget_constraints', 'medium')
        
        results = {}
        
        for pattern_id, pattern_info in self.transfer_patterns.items():
            try:
                results[pattern_id] = self._calculate_pattern_metrics(
                    pattern_id, pattern_info, data_size_gb, region, 
                    available_bandwidth_mbps, security_requirements, timeline_weeks
                )
            except Exception as e:
                print(f"Error calculating pattern {pattern_id}: {e}")
                # Add fallback result for this pattern
                results[pattern_id] = {
                    'transfer_time_hours': 24,
                    'transfer_time_days': 1,
                    'data_transfer_cost': data_size_gb * 0.09,
                    'infrastructure_cost': 1000,
                    'setup_cost': 5000,
                    'total_cost': data_size_gb * 0.09 + 6000,
                    'bandwidth_utilization': 70,
                    'reliability_score': 75,
                    'security_score': 60,
                    'complexity_score': 50
                }
        
        # Add recommendation engine
        try:
            results['recommendations'] = self._generate_recommendations(
                results, migration_params
            )
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            results['recommendations'] = {
                'primary_recommendation': {
                    'pattern_id': 'internet_dms',
                    'pattern_name': 'Internet + DMS',
                    'description': 'Standard internet connection with AWS Database Migration Service',
                    'score': 75.0,
                    'reasoning': 'Fallback recommendation due to analysis error'
                },
                'alternative_options': [],
                'cost_optimization': ['Consider Direct Connect for large data volumes'],
                'risk_considerations': ['Plan for variable internet speeds'],
                'timeline_impact': {
                    'fastest_option': {'pattern': 'Internet + DMS', 'duration_days': 1}
                }
            }
        
        return results
        
    except Exception as e:
        print(f"Error in calculate_transfer_analysis: {e}")
        # Return minimal fallback result
        return {
            'internet_dms': {
                'transfer_time_hours': 24,
                'transfer_time_days': 1,
                'data_transfer_cost': 1000,
                'infrastructure_cost': 1000,
                'setup_cost': 5000,
                'total_cost': 7000,
                'bandwidth_utilization': 70,
                'reliability_score': 75,
                'security_score': 60,
                'complexity_score': 50
            },
            'recommendations': {
                'primary_recommendation': {
                    'pattern_id': 'internet_dms',
                    'pattern_name': 'Internet + DMS',
                    'description': 'Fallback recommendation',
                    'score': 75.0,
                    'reasoning': 'Default configuration'
                }
            }
        }
    
    def _calculate_pattern_metrics(self, pattern_id: str, pattern_info: Dict, 
                             data_size_gb: int, region: str, bandwidth_mbps: int,
                             security_req: str, timeline_weeks: int) -> Dict:
    """Calculate metrics for a specific transfer pattern - WITH ERROR HANDLING"""
    
    try:
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
        else:
            # Return default/fallback metrics
            return {
                'transfer_time_hours': 24,
                'transfer_time_days': 1,
                'data_transfer_cost': data_size_gb * 0.09,
                'infrastructure_cost': 1000,
                'setup_cost': 5000,
                'total_cost': data_size_gb * 0.09 + 6000,
                'bandwidth_utilization': 70,
                'reliability_score': 75,
                'security_score': 60,
                'complexity_score': 50
            }
    except Exception as e:
        print(f"Error calculating metrics for pattern {pattern_id}: {e}")
        # Return fallback metrics
        return {
            'transfer_time_hours': 24,
            'transfer_time_days': 1,
            'data_transfer_cost': data_size_gb * 0.09,
            'infrastructure_cost': 1000,
            'setup_cost': 5000,
            'total_cost': data_size_gb * 0.09 + 6000,
            'bandwidth_utilization': 70,
            'reliability_score': 75,
            'security_score': 60,
            'complexity_score': 50
        }
    
    def _generate_recommendations(self, results: Dict, migration_params: Dict) -> Dict:
        """Generate recommendations for network transfer patterns"""
        data_size_gb = migration_params.get('data_size_gb', 1000)
        
        # Score each pattern
        pattern_scores = {}
        for pattern_id, metrics in results.items():
            if pattern_id == 'recommendations':
                continue
            
            # Simple scoring based on cost and time
            cost_score = max(0, 100 - (metrics['total_cost'] / 1000))
            time_score = max(0, 100 - (metrics['transfer_time_days'] * 10))
            composite_score = (cost_score + time_score) / 2
            
            pattern_scores[pattern_id] = {
                'composite_score': composite_score,
                'metrics': metrics
            }
        
        # Get best pattern
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1]['composite_score'])
        pattern_info = self.transfer_patterns[best_pattern[0]]
        
        return {
            'primary_recommendation': {
                'pattern_id': best_pattern[0],
                'pattern_name': pattern_info['name'],
                'description': pattern_info['description'],
                'score': best_pattern[1]['composite_score'],
                'reasoning': f"Best balance of cost and performance for {data_size_gb:,} GB migration"
            }
        }

# ===========================
# MIGRATION ANALYZER (CONSOLIDATED)
# ===========================

class MigrationAnalyzer:
    """Consolidated migration analyzer with all features"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = AWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key
        
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations"""
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            
            environment_type = self._categorize_environment(env_name)
            instance_class = self._calculate_instance_class(cpu_cores, ram_gb, environment_type)
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
        
        migration_costs = {
            'dms_instance': 0.2 * 24 * 7 * migration_timeline_weeks,
            'data_transfer': data_size_gb * 0.09,
            'professional_services': migration_timeline_weeks * 8000,
            'total': 0
        }
        
        migration_costs['total'] = sum([
            migration_costs['dms_instance'],
            migration_costs['data_transfer'],
            migration_costs['professional_services']
        ])
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs
        }
    
    def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate AI insights synchronously"""
        if not self.anthropic_api_key:
            return {
                'error': 'No Anthropic API key provided',
                'summary': f"Migration analysis complete. Monthly cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}",
                'recommendations': [
                    "Proceed with phased migration approach",
                    "Implement comprehensive testing strategy",
                    "Consider Reserved Instances for cost optimization"
                ]
            }
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            context = f"""
            AWS Migration Analysis:
            - Monthly Cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
            - Migration Cost: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}
            - Data Size: {migration_params.get('data_size_gb', 0):,} GB
            - Timeline: {migration_params.get('migration_timeline_weeks', 0)} weeks
            
            Provide migration insights and recommendations.
            """
            
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": context}]
            )
            
            return {
                'ai_analysis': message.content[0].text,
                'source': 'Claude AI',
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'Claude AI failed: {str(e)}',
                'summary': f"Migration analysis complete. Monthly cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}",
                'success': False
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
        elif cpu_cores <= 32 and ram_gb <= 128:
            instance_class = 'db.r5.2xlarge'
        else:
            instance_class = 'db.r5.4xlarge'
        
        # Environment-specific adjustments
        if env_type == 'development' and 'r5' in instance_class:
            downsized = {
                'db.r5.4xlarge': 'db.r5.2xlarge',
                'db.r5.2xlarge': 'db.r5.xlarge',
                'db.r5.xlarge': 'db.r5.large',
                'db.r5.large': 'db.t3.large'
            }
            instance_class = downsized.get(instance_class, instance_class)
        
        return instance_class
    
    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate cost for a single environment"""
        pricing = self.pricing_api.get_rds_pricing(region, target_engine, rec['instance_class'], rec['multi_az'])
        
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        instance_cost = pricing['hourly'] * monthly_hours
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        backup_cost = storage_cost * 0.2
        
        total_monthly = instance_cost + storage_cost + backup_cost
        
        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly
        }

# ===========================
# RISK ASSESSMENT MODULE
# ===========================

def create_default_risk_assessment() -> Dict:
    """Create a default risk assessment"""
    return {
        'overall_score': 45,
        'risk_level': {'level': 'Medium', 'color': '#d69e2e', 'action': 'Standard migration practices recommended'},
        'technical_risks': {
            'engine_compatibility': 40,
            'data_migration_complexity': 35,
            'application_integration': 45,
            'performance_risk': 30
        },
        'business_risks': {
            'timeline_risk': 50,
            'cost_overrun_risk': 40,
            'business_continuity': 45,
            'resource_availability': 35
        },
        'mitigation_strategies': [
            {
                'risk': 'Migration Planning',
                'strategy': 'Develop comprehensive migration plan with clear milestones',
                'timeline': '1-2 weeks',
                'cost_impact': 'Low'
            },
            {
                'risk': 'Testing & Validation',
                'strategy': 'Implement thorough testing at each migration phase',
                'timeline': '2-3 weeks',
                'cost_impact': 'Medium'
            }
        ]
    }

# ===========================
# UTILITY FUNCTIONS
# ===========================

def is_enhanced_environment_data(environment_specs):
    """Check if environment specs contain enhanced cluster data"""
    if not environment_specs:
        return False
    
    sample_spec = next(iter(environment_specs.values()))
    enhanced_fields = ['workload_pattern', 'read_write_ratio', 'multi_az_writer']
    
    return any(field in sample_spec for field in enhanced_fields)

def create_growth_projection_charts(growth_analysis: Dict) -> List[go.Figure]:
    """Create comprehensive growth projection visualizations"""
    charts = []
    projections = growth_analysis['yearly_projections']
    
    # 3-Year Cost Projection Chart
    years = ['Current', 'Year 1', 'Year 2', 'Year 3']
    annual_costs = [projections[f'year_{i}']['total_annual'] for i in range(4)]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=years, y=annual_costs,
        mode='lines+markers',
        name='Annual Cost Projection',
        line=dict(color='#3182ce', width=3),
        marker=dict(size=10)
    ))
    
    fig1.update_layout(
        title='3-Year Cost Projection with Growth',
        xaxis_title='Timeline',
        yaxis_title='Annual Cost ($)',
        height=500
    )
    
    charts.append(fig1)
    return charts

def prepare_csv_export_data(results, recommendations):
    """Prepare CSV data for export"""
    try:
        env_costs = results.get('environment_costs', {})
        if not env_costs:
            return None
        
        csv_data = []
        for env_name, costs in env_costs.items():
            if isinstance(costs, dict):
                csv_data.append({
                    'Environment': env_name,
                    'Instance_Cost': costs.get('instance_cost', 0),
                    'Storage_Cost': costs.get('storage_cost', 0),
                    'Backup_Cost': costs.get('backup_cost', 0),
                    'Total_Monthly': costs.get('total_monthly', 0),
                    'Total_Annual': costs.get('total_monthly', 0) * 12
                })
        
        return pd.DataFrame(csv_data) if csv_data else None
            
    except Exception as e:
        print(f"Error preparing CSV data: {e}")
        return None

# ===========================
# STREAMLIT INTERFACE FUNCTIONS
# ===========================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'environment_specs': {},
        'migration_params': {},
        'analysis_results': None,
        'recommendations': None,
        'risk_assessment': None,
        'ai_insights': None,
        'enhanced_recommendations': None,
        'enhanced_analysis_results': None,
        'enhanced_cost_chart': None,
        'growth_analysis': None,
        'optimization_results': None,
        'transfer_analysis': None,
        'network_analyzer': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

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

    # Growth Planning
    st.markdown("### 📈 Growth Planning & Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Workload Growth")
        annual_data_growth = st.slider("Annual Data Growth Rate (%)", min_value=0, max_value=100, value=15)
        annual_user_growth = st.slider("Annual User Growth Rate (%)", min_value=0, max_value=200, value=25)
        annual_transaction_growth = st.slider("Annual Transaction Growth Rate (%)", min_value=0, max_value=150, value=20)
    
    with col2:
        st.markdown("#### 🎯 Growth Scenarios")
        growth_scenario = st.selectbox("Growth Scenario", ["Conservative", "Moderate", "Aggressive", "Custom"], index=1)
        
        if growth_scenario != "Custom":
            scenario_multipliers = {"Conservative": 0.7, "Moderate": 1.0, "Aggressive": 1.5}
            multiplier = scenario_multipliers[growth_scenario]
            annual_data_growth = int(annual_data_growth * multiplier)
            annual_user_growth = int(annual_user_growth * multiplier)
            annual_transaction_growth = int(annual_transaction_growth * multiplier)
            
            st.info(f"**{growth_scenario} Scenario Applied:**")
            st.write(f"Data Growth: {annual_data_growth}%")
            st.write(f"User Growth: {annual_user_growth}%")
            st.write(f"Transaction Growth: {annual_transaction_growth}%")
        
        seasonality_factor = st.slider("Seasonality Factor", min_value=1.0, max_value=3.0, value=1.2, step=0.1)
        scaling_strategy = st.selectbox("Scaling Strategy", ["Auto-scaling", "Manual scaling", "Over-provision"])

    # AI Configuration
    st.markdown("### 🤖 AI Integration")
    
    anthropic_api_key = st.text_input("Anthropic API Key (Optional)", type="password")
    
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
            'annual_data_growth': annual_data_growth,
            'annual_user_growth': annual_user_growth,
            'annual_transaction_growth': annual_transaction_growth,
            'growth_scenario': growth_scenario,
            'seasonality_factor': seasonality_factor,
            'scaling_strategy': scaling_strategy
        }
        
        st.success("✅ Configuration saved! Proceed to Environment Setup.")
        st.balloons()

def show_environment_setup():
    """Show environment setup interface"""
    st.markdown("## 📊 Environment Configuration")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    config_method = st.radio(
        "Configuration Method:",
        ["📝 Manual Entry", "📁 Bulk Upload", "🔬 Enhanced Cluster Config"],
        horizontal=True
    )
    
    if config_method == "📁 Bulk Upload":
        show_bulk_upload_interface()
    elif config_method == "🔬 Enhanced Cluster Config":
        show_enhanced_cluster_configuration()
    else:
        show_manual_environment_setup()

def show_manual_environment_setup():
    """Show manual environment setup interface"""
    st.markdown("### 📝 Manual Environment Configuration")
    
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
                
                cpu_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=[4, 8, 16, 32][min(i, 3)], key=f"cpu_{i}")
                ram_gb = st.number_input("RAM (GB)", min_value=4, max_value=1024, value=[16, 32, 64, 128][min(i, 3)], key=f"ram_{i}")
                storage_gb = st.number_input("Storage (GB)", min_value=20, max_value=50000, value=[100, 500, 1000, 2000][min(i, 3)], key=f"storage_{i}")
                daily_usage_hours = st.slider("Daily Usage (Hours)", min_value=1, max_value=24, value=[8, 12, 16, 24][min(i, 3)], key=f"usage_{i}")
                peak_connections = st.number_input("Peak Connections", min_value=1, max_value=10000, value=[20, 50, 100, 500][min(i, 3)], key=f"connections_{i}")
                
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

def show_bulk_upload_interface():
    """Show bulk upload interface for environments"""
    st.markdown("### 📁 Bulk Environment Upload")
    
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
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_cols = ['environment', 'cpu_cores', 'ram_gb', 'storage_gb']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
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
            
            display_df = pd.DataFrame.from_dict(environment_specs, orient='index')
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_enhanced_cluster_configuration():
    """Show enhanced cluster configuration with Writer/Reader options"""
    st.markdown("### 🔬 Enhanced Database Cluster Configuration")
    
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    for i in range(num_environments):
        with st.expander(f"🏢 Environment {i+1} - Cluster Configuration", expanded=i == 0):
            
            col1, col2 = st.columns(2)
            
            with col1:
                env_name = st.text_input("Environment Name", value=default_names[i] if i < len(default_names) else f"Environment_{i+1}", key=f"env_name_{i}")
                environment_type = st.selectbox("Environment Type", ["Production", "Staging", "Testing", "Development"], index=min(i, 3), key=f"env_type_{i}")
            
            with col2:
                workload_pattern = st.selectbox("Workload Pattern", ["balanced", "read_heavy", "write_heavy", "analytics"], key=f"workload_{i}")
                read_write_ratio = st.slider("Read/Write Ratio (% Reads)", min_value=10, max_value=95, value=70, key=f"read_ratio_{i}")
            
            # Infrastructure configuration
            st.markdown("#### 💻 Infrastructure Requirements")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=[4, 8, 16, 32][min(i, 3)], key=f"cpu_{i}")
                ram_gb = st.number_input("RAM (GB)", min_value=4, max_value=1024, value=[16, 32, 64, 128][min(i, 3)], key=f"ram_{i}")
            
            with col2:
                storage_gb = st.number_input("Storage (GB)", min_value=20, max_value=50000, value=[100, 500, 1000, 2000][min(i, 3)], key=f"storage_{i}")
                iops_requirement = st.number_input("IOPS Requirement", min_value=100, max_value=50000, value=[1000, 3000, 5000, 10000][min(i, 3)], key=f"iops_{i}")
            
            with col3:
                peak_connections = st.number_input("Peak Connections", min_value=1, max_value=10000, value=[20, 50, 100, 500][min(i, 3)], key=f"connections_{i}")
                daily_usage_hours = st.slider("Daily Usage (Hours)", min_value=1, max_value=24, value=[8, 12, 16, 24][min(i, 3)], key=f"usage_{i}")
            
            # Cluster configuration
            st.markdown("#### 🔗 Cluster Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                multi_az_writer = st.checkbox("Multi-AZ for Writer", value=environment_type in ["Production", "Staging"], key=f"multi_az_writer_{i}")
                storage_encrypted = st.checkbox("Encryption at Rest", value=environment_type in ["Production", "Staging"], key=f"encryption_{i}")
            
            with col2:
                multi_az_readers = st.checkbox("Multi-AZ for Readers", value=environment_type == "Production", key=f"multi_az_readers_{i}")
                backup_retention = st.number_input("Backup Retention (Days)", min_value=1, max_value=35, value=30 if environment_type == "Production" else 7, key=f"backup_{i}")
            
            # Store enhanced environment configuration
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
                'storage_encrypted': storage_encrypted,
                'backup_retention': backup_retention
            }
    
    if st.button("💾 Save Enhanced Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Enhanced cluster configuration saved!")

def show_network_transfer_analysis():
    """Show network transfer analysis interface - FIXED VERSION"""
    
    st.markdown("## 🌐 Network Transfer Analysis")
    
    # Check prerequisites FIRST
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        st.info("👆 Go to 'Migration Configuration' section to set up your migration parameters")
        return
    
    # Initialize network analyzer if not exists
    if 'network_analyzer' not in st.session_state or st.session_state.network_analyzer is None:
        try:
            st.session_state.network_analyzer = NetworkTransferAnalyzer()
        except Exception as e:
            st.error(f"Error initializing Network Transfer Analyzer: {str(e)}")
            return
    
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
    try:
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
    except Exception as e:
        st.error(f"Error preparing network parameters: {str(e)}")
        return
    
    # Run network analysis
    if st.button("🚀 Analyze Network Transfer Options", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing network transfer patterns..."):
            try:
                transfer_analysis = analyzer.calculate_transfer_analysis(network_params)
                st.session_state.transfer_analysis = transfer_analysis
                st.success("✅ Network analysis complete!")
            except Exception as e:
                st.error(f"❌ Network analysis failed: {str(e)}")
                st.code(str(e))  # Show the full error for debugging
                return
    
    # Display results if available
    if hasattr(st.session_state, 'transfer_analysis') and st.session_state.transfer_analysis is not None:
        try:
            show_network_analysis_results()
        except Exception as e:
            st.error(f"Error displaying network results: {str(e)}")
    else:
        st.info("ℹ️ Run the network analysis to see results and recommendations.")

def show_network_analysis_results():
    """Display network analysis results"""
    if not hasattr(st.session_state, 'transfer_analysis') or st.session_state.transfer_analysis is None:
        st.error("No network analysis results available.")
        return
    
    transfer_analysis = st.session_state.transfer_analysis
    
    st.markdown("### 🎯 Network Recommendations")
    
    recommendations = transfer_analysis.get('recommendations', {})
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
    
    # Pattern comparison
    st.markdown("### 📊 Pattern Comparison")
    
    comparison_data = []
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
        
        pattern_info = st.session_state.network_analyzer.transfer_patterns[pattern_id]
        comparison_data.append({
            'Pattern': pattern_info['name'],
            'Total Cost': f"${metrics['total_cost']:,.0f}",
            'Transfer Duration': f"{metrics['transfer_time_days']:.1f} days",
            'Setup Cost': f"${metrics['setup_cost']:,.0f}",
            'Data Transfer Cost': f"${metrics['data_transfer_cost']:,.0f}",
            'Complexity': pattern_info['complexity']
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

def show_optimized_recommendations():
    """Show optimized Reader/Writer recommendations"""
    st.markdown("## 🧠 AI-Optimized Reader/Writer Recommendations")
    
    if not st.session_state.environment_specs:
        st.warning("⚠️ Please configure environments first.")
        return
    
    env_count = len(st.session_state.environment_specs)
    is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Environments Configured", env_count)
    with col2:
        config_type = "Enhanced Cluster Config" if is_enhanced else "Basic Config"
        st.metric("Configuration Type", config_type)
    
    # Configuration options
    st.markdown("### ⚙️ Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_goal = st.selectbox("Optimization Goal", ["Cost Efficiency", "Performance", "Balanced"], index=2)
    
    with col2:
        consider_reserved_instances = st.checkbox("Include Reserved Instance Analysis", value=True)
    
    with col3:
        environment_priority = st.selectbox("Environment Priority", ["Production First", "All Equal", "Cost First"], index=0)
    
    if st.button("🧠 Generate AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing workloads and optimizing configurations..."):
            try:
                optimizer = OptimizedReaderWriterAnalyzer()
                optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
                st.session_state.optimization_results = optimization_results
                
                st.success("✅ Optimization complete!")
                display_optimization_results(optimization_results)
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")

def display_optimization_results(optimization_results: Dict):
    """Display comprehensive optimization results"""
    st.markdown("# 🚀 Optimized Reader/Writer Recommendations")
    
    # Overall summary
    total_monthly_cost = sum([env['cost_analysis']['monthly_breakdown']['total'] for env in optimization_results.values()])
    avg_optimization_score = sum([env['optimization_score'] for env in optimization_results.values()]) / len(optimization_results)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Monthly Cost", f"${total_monthly_cost:,.0f}")
    
    with col2:
        st.metric("Avg Optimization Score", f"{avg_optimization_score:.1f}/100")
    
    with col3:
        total_instances = sum([1 + env['reader_optimization']['count'] for env in optimization_results.values()])
        st.metric("Total Instances", total_instances)
    
    # Detailed results for each environment
    for env_name, optimization in optimization_results.items():
        with st.expander(f"🏢 {env_name.title()} - Score: {optimization['optimization_score']:.1f}/100", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✍️ Writer Configuration")
                writer = optimization['writer_optimization']
                st.info(f"""
                **Instance:** {writer['instance_class']}  
                **vCPUs:** {writer['specs'].vcpu}  
                **Memory:** {writer['specs'].memory_gb} GB  
                **Monthly Cost:** ${writer['monthly_cost']:,.0f}
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
                    **Total Monthly Cost:** ${reader['total_monthly_cost']:,.0f}
                    """)
                else:
                    st.warning("**No read replicas recommended**")
                
                st.markdown("**Reasoning:**")
                st.write(reader['reasoning'])

def run_migration_analysis():
    """Run comprehensive migration analysis"""
    try:
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
        
        # Step 3: Growth Analysis
        st.write("📈 Calculating 3-year growth projections...")
        growth_analyzer = GrowthAwareCostAnalyzer()
        growth_analysis = growth_analyzer.calculate_3_year_growth_projection(cost_analysis, st.session_state.migration_params)
        st.session_state.growth_analysis = growth_analysis
        
        # Step 4: Risk assessment
        st.write("⚠️ Assessing risks...")
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        
        # Step 5: AI insights
        if anthropic_api_key:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = analyzer.generate_ai_insights_sync(cost_analysis, st.session_state.migration_params)
                st.session_state.ai_insights = ai_insights
                
                if ai_insights.get('success'):
                    st.success("✅ AI insights generated")
                else:
                    st.warning("⚠️ AI insights partially available")
                    
            except Exception as e:
                st.warning(f"AI insights failed: {str(e)}")
                st.session_state.ai_insights = {'error': str(e)}
        
        st.success("✅ Analysis complete!")
        
        # Show summary
        show_analysis_summary()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if not hasattr(st.session_state, 'risk_assessment'):
            st.session_state.risk_assessment = create_default_risk_assessment()

def show_analysis_summary():
    """Show analysis summary after completion"""
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
            if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
                growth_percent = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
                st.metric("3-Year Growth", f"{growth_percent:.1f}%")
    
    st.info("📈 View detailed results in the 'Results Dashboard' section")

def show_analysis_section():
    """Show analysis and recommendations section"""
    st.markdown("## 🚀 Migration Analysis & Recommendations")
    
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
        
        count = 0
        for env_name, specs in envs.items():
            if count < 4:
                cpu_cores = specs.get('cpu_cores', 'N/A')
                ram_gb = specs.get('ram_gb', 'N/A')
                st.markdown(f"• **{env_name}:** {cpu_cores} cores, {ram_gb} GB RAM")
                count += 1
        
        if len(envs) > 4:
            st.markdown(f"• ... and {len(envs) - 4} more environments")
    
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing migration requirements..."):
            run_migration_analysis()

def show_results_dashboard():
    """Show comprehensive results dashboard"""
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run the migration analysis first.")
        return
    
    st.markdown("## 📊 Migration Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Cost Summary",
        "📈 Growth Projections",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "🤖 AI Insights",
        "📅 Timeline"
    ])
    
    with tab1:
        show_cost_summary()
    
    with tab2:
        show_growth_analysis_dashboard()
    
    with tab3:
        show_risk_assessment_dashboard()
    
    with tab4:
        show_environment_analysis()
    
    with tab5:
        show_ai_insights()
    
    with tab6:
        show_timeline_analysis()

def show_cost_summary():
    """Show cost summary from analysis results"""
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
                if isinstance(costs, dict) and 'instance_cost' in costs:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Instance Cost", f"${costs.get('instance_cost', 0):,.2f}/month")
                        st.metric("Storage Cost", f"${costs.get('storage_cost', 0):,.2f}/month")
                    
                    with col2:
                        st.metric("Backup Cost", f"${costs.get('backup_cost', 0):,.2f}/month")
                    
                    with col3:
                        total_env_cost = costs.get('total_monthly', sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'backup_cost']]))
                        st.metric("Total Monthly", f"${total_env_cost:,.2f}")
                else:
                    if isinstance(costs, (int, float)):
                        st.metric("Monthly Cost", f"${costs:,.2f}")

def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Run the migration analysis to see growth projections.")
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
    # Key Growth Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("3-Year Growth", f"{growth_summary['total_3_year_growth_percent']:.1f}%", delta=f"CAGR: {growth_summary['compound_annual_growth_rate']:.1f}%")
    
    with col2:
        st.metric("Current Annual Cost", f"${growth_summary['year_0_cost']:,.0f}", delta="Baseline")
    
    with col3:
        st.metric("Year 3 Projected Cost", f"${growth_summary['year_3_cost']:,.0f}", delta=f"+${growth_summary['year_3_cost'] - growth_summary['year_0_cost']:,.0f}")
    
    with col4:
        st.metric("Total 3-Year Investment", f"${growth_summary['total_3_year_investment']:,.0f}", delta=f"Avg: ${growth_summary['average_annual_cost']:,.0f}/year")
    
    # Growth Projection Charts
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        for i, chart in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True, key=f"growth_chart_{i}")
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")

def show_risk_assessment_dashboard():
    """Show risk assessment results"""
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        risk_assessment = st.session_state.risk_assessment
        
        # Overall risk level display
        risk_level = risk_assessment.get('risk_level', {'level': 'Unknown', 'color': '#666666', 'action': 'Assessment needed'})
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
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")

def show_environment_analysis():
    """Show environment analysis"""
    if not st.session_state.analysis_results:
        st.warning("⚠️ Environment analysis not available.")
        return
        
    st.markdown("### 🏢 Environment Analysis")
    
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

def show_ai_insights():
    """Show AI insights if available"""
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        st.markdown("### 🤖 AI-Powered Insights")
        
        insights = st.session_state.ai_insights
        
        if 'error' in insights:
            st.warning(f"AI insights partially available")
            st.error(f"Error: {insights['error']}")
        
        if 'ai_analysis' in insights:
            st.markdown("#### 📝 AI Analysis")
            st.write(insights['ai_analysis'])
        
        if 'summary' in insights:
            st.markdown("#### 📝 Summary")
            st.write(insights['summary'])
        
        if 'recommendations' in insights:
            st.markdown("#### 💡 Recommendations")
            for rec in insights['recommendations']:
                st.markdown(f"• {rec}")
    else:
        st.info("🤖 AI insights not available. Provide an Anthropic API key in the configuration to enable AI-powered analysis.")

def show_timeline_analysis():
    """Show migration timeline analysis"""
    st.markdown("### 📅 Migration Timeline & Milestones")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Timeline analysis not available.")
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
    for phase in phases:
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
    
    results = st.session_state.analysis_results
    
    # Report generation options
    st.markdown("### 📊 Available Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👔 Executive Summary")
        st.markdown("**Includes:**")
        st.markdown("• High-level cost analysis")
        st.markdown("• ROI and timeline overview")
        st.markdown("• Key recommendations")
        
        if st.button("📄 Generate Executive PDF", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                try:
                    # Simple text-based report for demo
                    report_content = f"""
                    AWS Migration Executive Summary
                    
                    Migration Analysis Results:
                    - Monthly AWS Cost: ${results.get('monthly_aws_cost', 0):,.0f}
                    - Annual AWS Cost: ${results.get('monthly_aws_cost', 0) * 12:,.0f}
                    - Migration Investment: ${results.get('migration_costs', {}).get('total', 0):,.0f}
                    
                    Key Recommendations:
                    • Proceed with phased migration approach
                    • Implement comprehensive testing protocols
                    • Consider Reserved Instances for cost optimization
                    
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    """
                    
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=report_content,
                        file_name=f"AWS_Migration_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    with col2:
        st.markdown("#### 🔧 Technical Report")
        st.markdown("**Includes:**")
        st.markdown("• Environment specifications")
        st.markdown("• Instance recommendations")
        st.markdown("• Detailed cost breakdown")
        
        if st.button("📄 Generate Technical Report", use_container_width=True):
            st.info("Technical report generation would be implemented here")
    
    with col3:
        st.markdown("#### 📊 Data Export")
        st.markdown("**Includes:**")
        st.markdown("• Cost analysis data")
        st.markdown("• Environment specifications")
        st.markdown("• Risk assessment data")
        
        if st.button("📊 Export Data (CSV)", use_container_width=True):
            try:
                csv_data = prepare_csv_export_data(results, st.session_state.recommendations)
                
                if csv_data is not None:
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
                "🧠 AI Optimizer",
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
            results = st.session_state.analysis_results
            st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
        else:
            st.info("ℹ️ Analysis pending")
    
    # Main content area
    if page == "🔧 Migration Configuration":
        show_migration_configuration()
    elif page == "📊 Environment Setup":
        show_environment_setup()
    elif page == "🌐 Network Analysis":
        show_network_transfer_analysis()
    elif page == "🧠 AI Optimizer":
        show_optimized_recommendations()
    elif page == "🚀 Analysis & Recommendations":
        show_analysis_section()
    elif page == "📈 Results Dashboard":
        show_results_dashboard()
    elif page == "📄 Reports & Export":
        show_reports_section()
    else:
        st.markdown("## Welcome to the AWS Database Migration Tool")
        st.markdown("Please select a section from the sidebar to get started.")

if __name__ == "__main__":
    main()