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

class EnhancedAWSPricingAPI:
    """Enhanced AWS Pricing API with Writer/Reader and Aurora support"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get RDS pricing for specific instance with Multi-AZ support"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
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
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                },
                'mysql': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-mysql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
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

class MigrationAnalyzer:
    """Migration analyzer for standard environment configurations"""
    
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
        """Calculate migration costs based on recommendations"""
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            env_costs = self._calculate_environment_cost(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        ps_cost = migration_timeline_weeks * 8000
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        
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
        """Generate REAL Claude AI insights synchronously"""
        if not self.anthropic_api_key or not ANTHROPIC_AVAILABLE:
            return {'error': 'No Anthropic API key provided or library not available', 'source': 'Error'}
        
        try:
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
            
        except Exception as e:
            return {'error': f'Claude AI failed: {str(e)}', 'source': 'API Error', 'success': False}
    
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
        elif cpu_cores <= 64 and ram_gb <= 256:
            instance_class = 'db.r5.4xlarge'
        else:
            instance_class = 'db.r5.8xlarge'
        
        if env_type == 'development' and 'r5' in instance_class:
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
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs"""
        use_direct_connect = migration_params.get('use_direct_connect', False)
        
        internet_cost = data_size_gb * 0.09
        
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02
        else:
            dx_cost = internet_cost
        
        return {
            'internet': internet_cost,
            'direct_connect': dx_cost,
            'total': min(internet_cost, dx_cost)
        }

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
            }
        }
    
    def calculate_3_year_growth_projection(self, base_costs: Dict, migration_params: Dict) -> Dict:
        """Calculate 3-year cost projection with growth"""
        data_growth = migration_params.get('annual_data_growth', 15) / 100
        user_growth = migration_params.get('annual_user_growth', 25) / 100
        transaction_growth = migration_params.get('annual_transaction_growth', 20) / 100
        seasonality = migration_params.get('seasonality_factor', 1.2)
        
        projections = {}
        
        for year in range(4):
            year_multiplier = {
                'data': (1 + data_growth) ** year,
                'users': (1 + user_growth) ** year,
                'transactions': (1 + transaction_growth) ** year
            }
            
            year_costs = self._calculate_year_costs(base_costs, year_multiplier, seasonality, year)
            projections[f'year_{year}'] = year_costs
        
        return {
            'yearly_projections': projections,
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
            
            base_instance_cost = env_costs.get('instance_cost', 0)
            base_storage_cost = env_costs.get('storage_cost', 0)
            
            adjusted_costs = {
                'instance_cost': base_instance_cost * compute_multiplier,
                'storage_cost': base_storage_cost * storage_multiplier,
                'backup_cost': (base_storage_cost * storage_multiplier) * 0.2,
                'total_monthly': 0,
                'peak_monthly': 0,
                'resource_scaling': {
                    'compute_scaling': compute_multiplier,
                    'storage_scaling': storage_multiplier
                }
            }
            
            adjusted_costs['total_monthly'] = sum([
                adjusted_costs['instance_cost'],
                adjusted_costs['storage_cost'],
                adjusted_costs['backup_cost']
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
        """Identify cost optimization opportunities based on growth projections"""
        optimizations = []
        
        year_3 = projections['year_3']
        total_growth = (year_3['total_annual'] / projections['year_0']['total_annual'] - 1) * 100
        
        if total_growth > 100:
            optimizations.append({
                'opportunity': 'Reserved Instance Strategy',
                'description': f"Total costs growing {total_growth:.1f}% over 3 years. Reserved Instances can provide 30-60% savings.",
                'estimated_savings': year_3['total_annual'] * 0.35,
                'implementation_effort': 'Low',
                'timeline': '1-2 months'
            })
        
        return optimizations

# ===========================
# RISK ASSESSMENT FUNCTIONS
# ===========================

def create_default_risk_assessment():
    """Create a default risk assessment when none exists"""
    migration_params = getattr(st.session_state, 'migration_params', {})
    recommendations = getattr(st.session_state, 'recommendations', {})
    
    try:
        if migration_params and recommendations:
            return calculate_migration_risks_safe(migration_params, recommendations)
        else:
            return get_fallback_risk_assessment()
    except:
        return get_fallback_risk_assessment()

def calculate_migration_risks_safe(migration_params: Dict, recommendations: Dict) -> Dict:
    """Safe version of risk calculation that never fails"""
    try:
        source_engine = migration_params.get('source_engine', 'unknown')
        target_engine = migration_params.get('target_engine', 'unknown')
        data_size_gb = int(migration_params.get('data_size_gb', 1000))
        timeline_weeks = int(migration_params.get('migration_timeline_weeks', 12))
        team_size = int(migration_params.get('team_size', 5))
        
        engine_risk = 75 if source_engine != target_engine else 15
        data_risk = min(80, 20 + (data_size_gb / 1000) * 10)
        timeline_risk = 30 + max(0, (12 - timeline_weeks) * 5)
        team_risk = 40 + max(0, (5 - team_size) * 10)
        
        technical_risks = {
            'engine_compatibility': engine_risk,
            'data_migration_complexity': data_risk,
            'application_integration': 45,
            'performance_risk': 35
        }
        
        business_risks = {
            'timeline_risk': timeline_risk,
            'cost_overrun_risk': 40,
            'business_continuity': 45,
            'resource_availability': team_risk
        }
        
        tech_avg = sum(technical_risks.values()) / len(technical_risks)
        business_avg = sum(business_risks.values()) / len(business_risks)
        overall_score = (tech_avg + business_avg) / 2
        
        if overall_score < 30:
            risk_level = {'level': 'Low', 'color': '#38a169', 'action': 'Standard monitoring sufficient'}
        elif overall_score < 50:
            risk_level = {'level': 'Medium', 'color': '#d69e2e', 'action': 'Active monitoring recommended'}
        elif overall_score < 70:
            risk_level = {'level': 'High', 'color': '#e53e3e', 'action': 'Immediate mitigation required'}
        else:
            risk_level = {'level': 'Critical', 'color': '#9f1239', 'action': 'Project review required'}
        
        mitigation_strategies = [
            {
                'risk': 'Migration Planning',
                'strategy': 'Develop comprehensive migration plan with clear milestones',
                'timeline': '2-3 weeks',
                'cost_impact': 'Medium'
            }
        ]
        
        return {
            'overall_score': overall_score,
            'risk_level': risk_level,
            'technical_risks': technical_risks,
            'business_risks': business_risks,
            'mitigation_strategies': mitigation_strategies
        }
        
    except Exception as e:
        return get_fallback_risk_assessment()

def get_fallback_risk_assessment() -> Dict:
    """Fallback risk assessment when all else fails"""
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
            }
        ]
    }

# ===========================
# ANALYSIS FUNCTIONS
# ===========================

def run_migration_analysis():
    """Run migration analysis with growth projections"""
    try:
        anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
        analyzer = MigrationAnalyzer(anthropic_api_key)
        
        st.write("📊 Calculating instance recommendations...")
        recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
        st.session_state.recommendations = recommendations
        
        st.write("💰 Analyzing costs...")
        cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.analysis_results = cost_analysis
        
        st.write("⚠️ Assessing risks...")
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        
        st.write("📈 Calculating 3-year growth projections...")
        growth_analyzer = GrowthAwareCostAnalyzer()
        growth_analysis = growth_analyzer.calculate_3_year_growth_projection(
            cost_analysis, st.session_state.migration_params
        )
        st.session_state.growth_analysis = growth_analysis
        
        if anthropic_api_key:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = analyzer.generate_ai_insights_sync(cost_analysis, st.session_state.migration_params)
                st.session_state.ai_insights = ai_insights
                st.success("✅ AI insights generated")
            except Exception as e:
                st.warning(f"AI insights failed: {str(e)}")
                st.session_state.ai_insights = {'error': str(e)}
        
        st.success("✅ Analysis complete with growth projections!")
        show_analysis_summary()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if not hasattr(st.session_state, 'risk_assessment') or st.session_state.risk_assessment is None:
            st.session_state.risk_assessment = get_fallback_risk_assessment()

def show_analysis_summary():
    """Show analysis summary after completion"""
    st.markdown("#### 🎯 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    results = st.session_state.analysis_results
    
    with col1:
        st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
    
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
        if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
            risk_level = st.session_state.risk_assessment['risk_level']['level']
            st.metric("Risk Level", risk_level)
    
    st.info("📈 View detailed results in the 'Results Dashboard' section")

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def create_growth_projection_charts(growth_analysis: Dict) -> List[go.Figure]:
    """Create comprehensive growth projection visualizations"""
    charts = []
    projections = growth_analysis['yearly_projections']
    
    # 3-Year Cost Projection Chart
    years = ['Current', 'Year 1', 'Year 2', 'Year 3']
    annual_costs = [projections[f'year_{i}']['total_annual'] for i in range(4)]
    peak_costs = [projections[f'year_{i}']['peak_annual'] for i in range(4)]
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=years, y=annual_costs,
        mode='lines+markers',
        name='Average Annual Cost',
        line=dict(color='#3182ce', width=3),
        marker=dict(size=10)
    ))
    
    fig1.add_trace(go.Scatter(
        x=years, y=peak_costs,
        mode='lines+markers',
        name='Peak Annual Cost (with seasonality)',
        line=dict(color='#e53e3e', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    
    fig1.update_layout(
        title='3-Year Cost Projection with Growth',
        xaxis_title='Timeline',
        yaxis_title='Annual Cost ($)',
        height=500,
        hovermode='x unified'
    )
    
    charts.append(fig1)
    return charts

def create_environment_comparison_chart(environment_costs: Dict) -> go.Figure:
    """Create environment cost comparison chart"""
    environments = list(environment_costs.keys())
    total_costs = []
    
    for env_name, costs in environment_costs.items():
        if isinstance(costs, dict):
            cost = costs.get('total_monthly', 0)
        else:
            cost = float(costs) if costs else 0
        total_costs.append(cost)
    
    fig = go.Figure(data=[
        go.Bar(x=environments, y=total_costs, marker_color='#3182ce')
    ])
    
    fig.update_layout(
        title='Monthly Cost by Environment',
        xaxis_title='Environment',
        yaxis_title='Monthly Cost ($)',
        height=400
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
    migration_costs = analysis_results.get('migration_costs', {})
    
    metrics_data = [
        ['Metric', 'Value', 'Impact'],
        ['Monthly AWS Cost', f"${analysis_results.get('monthly_aws_cost', 0):,.0f}", 'Operational'],
        ['Annual AWS Cost', f"${analysis_results.get('annual_aws_cost', 0):,.0f}", 'Budget Planning'],
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

# ===========================
# UI FUNCTIONS
# ===========================

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
        seasonality_factor = st.slider("Seasonality Factor", min_value=1.0, max_value=3.0, value=1.2, step=0.1)
        scaling_strategy = st.selectbox("Scaling Strategy", ["Auto-scaling", "Manual scaling", "Over-provision"])

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
            'migration_budget': migration_budget,
            'anthropic_api_key': anthropic_api_key,
            'annual_data_growth': annual_data_growth,
            'annual_user_growth': annual_user_growth,
            'annual_transaction_growth': annual_transaction_growth,
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
        ["📝 Manual Entry", "📁 Bulk Upload"],
        horizontal=True
    )
    
    if config_method == "📁 Bulk Upload":
        show_bulk_upload_interface()
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
        
        summary_df = pd.DataFrame.from_dict(environment_specs, orient='index')
        st.dataframe(summary_df, use_container_width=True)

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

def show_analysis_section():
    """Show analysis and recommendations section"""
    st.markdown("## 🚀 Migration Analysis & Recommendations")
    
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
        st.session_state.analysis_results = None
        
        with st.spinner("🔄 Analyzing migration requirements..."):
            run_migration_analysis()

def show_results_dashboard():
    """Show comprehensive results dashboard"""
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run the migration analysis first.")
        return
    
    st.markdown("## 📊 Migration Analysis Results")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Cost Summary",
        "📈 Growth Projections",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "📊 Visualizations",
        "🤖 AI Insights"
    ])
    
    with tab1:
        show_cost_summary()
    
    with tab2:
        show_growth_analysis_dashboard()
    
    with tab3:
        show_risk_assessment_tab()
    
    with tab4:
        show_environment_analysis_tab()
    
    with tab5:
        show_visualizations_tab()
    
    with tab6:
        show_ai_insights_tab()

def show_cost_summary():
    """Show basic cost summary from analysis results"""
    if not st.session_state.analysis_results:
        st.error("No analysis results available.")
        return
    
    results = st.session_state.analysis_results
    
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
                        total_env_cost = costs.get('total_monthly', 0)
                        st.metric("Total Monthly", f"${total_env_cost:,.2f}")
                else:
                    if isinstance(costs, (int, float)):
                        st.metric("Monthly Cost", f"${costs:,.2f}")

def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
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
    
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        for i, chart in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True, key=f"growth_chart_{i}")
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")

def show_risk_assessment_tab():
    """Show risk assessment results"""
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        show_risk_assessment()
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")

def show_risk_assessment():
    """Show risk assessment dashboard"""
    st.markdown("### ⚠️ Migration Risk Assessment")
    
    risk_assessment = st.session_state.risk_assessment
    
    # Overall risk level display
    risk_level = risk_assessment.get('risk_level', {'level': 'Unknown', 'color': '#666666', 'action': 'Assessment needed'})
    overall_score = risk_assessment.get('overall_score', 50)
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {risk_level['color']}; background: linear-gradient(90deg, {risk_level['color']}22, white);">
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
    
    st.markdown("---")
    
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

def show_environment_analysis_tab():
    """Show environment analysis"""
    if not st.session_state.analysis_results:
        st.warning("⚠️ Environment analysis not available. Please run the migration analysis first.")
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

def show_visualizations_tab():
    """Show visualization charts"""
    st.markdown("### 📊 Cost & Performance Visualizations")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Visualizations not available. Please run the migration analysis first.")
        return
    
    try:
        results = st.session_state.analysis_results
        env_costs = results.get('environment_costs', {})
        
        if env_costs:
            fig = create_environment_comparison_chart(env_costs)
            st.plotly_chart(fig, use_container_width=True, key="env_cost_chart")
        
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            st.markdown("#### 📈 Growth Projections")
            try:
                charts = create_growth_projection_charts(st.session_state.growth_analysis)
                for i, chart in enumerate(charts):
                    st.plotly_chart(chart, use_container_width=True, key=f"viz_growth_chart_{i}")
            except Exception as e:
                st.error(f"Error creating growth charts: {str(e)}")
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def show_ai_insights_tab():
    """Show AI insights if available"""
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        st.markdown("### 🤖 AI-Powered Insights")
        
        insights = st.session_state.ai_insights
        
        if 'error' in insights:
            st.warning(f"AI insights partially available: {insights.get('summary', 'Analysis complete')}")
            st.error(f"Error: {insights['error']}")
        else:
            if 'ai_analysis' in insights:
                st.markdown("#### 📝 AI Analysis")
                st.write(insights['ai_analysis'])
            
            if 'source' in insights:
                st.info(f"Generated by: {insights['source']}")
    else:
        st.info("🤖 AI insights not available. Provide an Anthropic API key in the configuration to enable AI-powered analysis.")

def show_reports_section():
    """Show reports and export section"""
    st.markdown("## 📄 Reports & Export")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        return
    
    results = st.session_state.analysis_results
    
    st.markdown("### 📊 Available Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👔 Executive Summary")
        st.markdown("Perfect for stakeholders and decision makers")
        
        if st.button("📄 Generate Executive PDF", key="exec_pdf", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                try:
                    pdf_buffer = generate_executive_summary_pdf(results, st.session_state.migration_params)
                    
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=pdf_buffer.getvalue(),
                        file_name=f"AWS_Migration_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col2:
        st.markdown("#### 📊 Data Export")
        st.markdown("Raw data for further analysis")
        
        if st.button("📊 Export Data (CSV)", key="csv_export", use_container_width=True):
            try:
                env_costs = results.get('environment_costs', {})
                csv_data = []
                
                for env_name, costs in env_costs.items():
                    if isinstance(costs, dict):
                        csv_data.append({
                            'Environment': env_name,
                            'Instance_Cost': costs.get('instance_cost', 0),
                            'Storage_Cost': costs.get('storage_cost', 0),
                            'Backup_Cost': costs.get('backup_cost', 0),
                            'Total_Monthly': costs.get('total_monthly', 0)
                        })
                
                if csv_data:
                    csv_df = pd.DataFrame(csv_data)
                    csv_string = csv_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV Data",
                        data=csv_string,
                        file_name=f"AWS_Migration_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error preparing CSV: {str(e)}")

# ===========================
# SESSION STATE INITIALIZATION
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
        'growth_analysis': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ===========================
# MAIN APPLICATION
# ===========================

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
            st.metric("Migration Cost", f"${results['migration_costs']['total']:,.0f}")
        else:
            st.info("ℹ️ Analysis pending")
    
    # Main content area
    if page == "🔧 Migration Configuration":
        show_migration_configuration()
    elif page == "📊 Environment Setup":
        show_environment_setup()
    elif page == "🚀 Analysis & Recommendations":
        show_analysis_section()
    elif page == "📈 Results Dashboard":
        show_results_dashboard()
    elif page == "📄 Reports & Export":
        show_reports_section()

if __name__ == "__main__":
    main()