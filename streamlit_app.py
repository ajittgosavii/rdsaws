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
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ ReportLab not installed. PDF generation will be limited.")

# Optional imports for enhanced functionality
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
        
        # Additional complexity for heterogeneous migrations
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


class AWSPricingAPI:
    """AWS Pricing API for real-time cost calculation"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        try:
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        except Exception as e:
            self.pricing_client = None
    
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get RDS pricing from AWS Pricing API or fallback data"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try real AWS pricing first
        try:
            if self.pricing_client:
                real_pricing = self._fetch_real_aws_pricing(region, engine, instance_class, multi_az)
                if real_pricing:
                    self.cache[cache_key] = real_pricing
                    return real_pricing
        except Exception as e:
            pass
        
        # Fallback to static pricing
        result = self._get_fallback_pricing(region, engine, instance_class, multi_az)
        self.cache[cache_key] = result
        return result
    
    def _fetch_real_aws_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool) -> Optional[Dict]:
        """Fetch real pricing from AWS Pricing API"""
        if not self.pricing_client:
            return None
        
        try:
            # Simplified pricing fetch - would need more complex logic for production
            return None  # Placeholder for real implementation
        except Exception:
            return None
    
    def _get_fallback_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool) -> Dict:
        """Fallback pricing when API is unavailable"""
        
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92},
                    'db.r5.4xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84},
                },
                'mysql': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96},
                },
                'aurora-postgresql': {
                    'db.r5.large': {'hourly': 0.29, 'storage_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.58, 'storage_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 1.16, 'storage_gb': 0.10},
                },
                'aurora-mysql': {
                    'db.r5.large': {'hourly': 0.29, 'storage_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.58, 'storage_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 1.16, 'storage_gb': 0.10},
                }
            }
        }
        
        engine_pricing = pricing_data.get(region, {}).get(engine, {})
        instance_pricing = engine_pricing.get(instance_class, {
            'hourly': 0.5, 
            'hourly_multi_az': 1.0
        })
        
        hourly_cost = instance_pricing.get('hourly_multi_az' if multi_az else 'hourly', 0.5)
        
        return {
            'hourly': hourly_cost,
            'storage_gb': 0.115,
            'iops_gb': 0.10,
            'io_request': 0.20,
            'is_aurora': 'aurora' in engine,
            'multi_az': multi_az,
            'source': 'Fallback Pricing'
        }


class MigrationAnalyzer:
    """Main migration analyzer class"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = AWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key
    
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations"""
        
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
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # Calculate migration costs
        migration_costs = self._calculate_migration_service_costs(data_size_gb, timeline_weeks, migration_params)
        
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
        
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )
        
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
            'total_monthly': total_monthly,
            'pricing_source': pricing.get('source', 'Unknown')
        }
    
    def _calculate_migration_service_costs(self, data_size_gb: int, timeline_weeks: int, migration_params: Dict) -> Dict:
        """Calculate migration service costs"""
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * timeline_weeks
        
        # Data transfer costs
        use_direct_connect = migration_params.get('use_direct_connect', False)
        transfer_cost_per_gb = 0.02 if use_direct_connect else 0.09
        data_transfer_cost = data_size_gb * transfer_cost_per_gb
        
        # Professional services
        ps_cost = timeline_weeks * 8000
        
        base_cost = dms_instance_cost + data_transfer_cost + ps_cost
        contingency = base_cost * 0.2
        
        return {
            'dms_instance': dms_instance_cost,
            'data_transfer': data_transfer_cost,
            'professional_services': ps_cost,
            'contingency': contingency,
            'total': base_cost + contingency
        }


class GrowthAwareCostAnalyzer:
    """Cost analyzer with 3-year growth projections"""
    
    def __init__(self):
        self.growth_factors = {
            'compute': {'user_growth_factor': 0.8, 'transaction_factor': 0.9, 'data_factor': 0.3},
            'storage': {'data_growth_factor': 1.0, 'user_growth_factor': 0.2, 'transaction_factor': 0.4},
            'iops': {'transaction_factor': 0.95, 'user_growth_factor': 0.7, 'data_factor': 0.5}
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
            
            year_costs = self._calculate_year_costs(base_costs, year_multiplier, seasonality, year)
            projections[f'year_{year}'] = year_costs
        
        return {
            'yearly_projections': projections,
            'growth_summary': self._generate_growth_summary(projections),
            'scaling_recommendations': self._generate_scaling_recommendations(projections),
            'cost_optimization_opportunities': self._identify_growth_optimizations(projections)
        }
    
    def _calculate_year_costs(self, base_costs: Dict, multipliers: Dict, seasonality: float, year: int) -> Dict:
        """Calculate costs for a specific year"""
        
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
            
            year_costs[env_name] = adjusted_costs
        
        total_monthly = sum([env['total_monthly'] for env in year_costs.values()])
        
        return {
            'environment_costs': year_costs,
            'total_monthly': total_monthly,
            'total_annual': total_monthly * 12,
            'peak_monthly': total_monthly * seasonality,
            'peak_annual': total_monthly * 12 * seasonality,
            'year': year
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
    
    def _generate_scaling_recommendations(self, projections: Dict) -> List[Dict]:
        """Generate scaling recommendations"""
        
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
        
        return recommendations
    
    def _identify_growth_optimizations(self, projections: Dict) -> List[Dict]:
        """Identify cost optimization opportunities"""
        
        optimizations = []
        year_3 = projections['year_3']
        
        if year_3['total_annual'] > 100000:  # $100k threshold
            optimizations.append({
                'opportunity': 'Reserved Instance Strategy',
                'description': 'High annual costs justify Reserved Instance purchases for 30-40% savings',
                'estimated_savings': year_3['total_annual'] * 0.35,
                'implementation_effort': 'Low',
                'timeline': '1-2 months'
            })
        
        return optimizations


# ===========================
# RISK ASSESSMENT
# ===========================

def create_default_risk_assessment() -> Dict:
    """Create a comprehensive risk assessment"""
    
    try:
        migration_params = getattr(st.session_state, 'migration_params', {})
        recommendations = getattr(st.session_state, 'recommendations', {})
        
        if migration_params and recommendations:
            return calculate_migration_risks(migration_params, recommendations)
        else:
            return get_fallback_risk_assessment()
    except:
        return get_fallback_risk_assessment()


def calculate_migration_risks(migration_params: Dict, recommendations: Dict) -> Dict:
    """Calculate comprehensive migration risks"""
    
    source_engine = migration_params.get('source_engine', 'unknown')
    target_engine = migration_params.get('target_engine', 'unknown')
    data_size_gb = migration_params.get('data_size_gb', 1000)
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    team_size = migration_params.get('team_size', 5)
    
    # Technical risks
    engine_risk = 75 if source_engine != target_engine else 25
    data_risk = min(80, 20 + (data_size_gb / 1000) * 10)
    app_risk = min(80, migration_params.get('num_applications', 1) * 15)
    
    technical_risks = {
        'engine_compatibility': engine_risk,
        'data_migration_complexity': data_risk,
        'application_integration': app_risk,
        'performance_risk': 40
    }
    
    # Business risks
    timeline_risk = 60 if timeline_weeks < 12 else 40
    cost_risk = 50  # Default
    continuity_risk = 45
    resource_risk = 60 if team_size < 5 else 35
    
    business_risks = {
        'timeline_risk': timeline_risk,
        'cost_overrun_risk': cost_risk,
        'business_continuity': continuity_risk,
        'resource_availability': resource_risk
    }
    
    # Calculate overall score
    overall_score = (sum(technical_risks.values()) + sum(business_risks.values())) / 8
    
    # Generate risk level
    if overall_score < 30:
        risk_level = {'level': 'Low', 'color': '#38a169', 'action': 'Standard monitoring sufficient'}
    elif overall_score < 50:
        risk_level = {'level': 'Medium', 'color': '#d69e2e', 'action': 'Active monitoring recommended'}
    elif overall_score < 70:
        risk_level = {'level': 'High', 'color': '#e53e3e', 'action': 'Immediate mitigation required'}
    else:
        risk_level = {'level': 'Critical', 'color': '#9f1239', 'action': 'Project review required'}
    
    return {
        'overall_score': overall_score,
        'risk_level': risk_level,
        'technical_risks': technical_risks,
        'business_risks': business_risks,
        'mitigation_strategies': generate_mitigation_strategies(technical_risks, business_risks)
    }


def generate_mitigation_strategies(technical_risks: Dict, business_risks: Dict) -> List[Dict]:
    """Generate mitigation strategies"""
    
    strategies = []
    
    if technical_risks.get('engine_compatibility', 0) > 60:
        strategies.append({
            'risk': 'Engine Compatibility',
            'strategy': 'Use AWS Schema Conversion Tool and conduct thorough testing',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    if business_risks.get('timeline_risk', 0) > 50:
        strategies.append({
            'risk': 'Timeline Risk',
            'strategy': 'Add parallel workstreams and increase team capacity',
            'timeline': 'Immediate',
            'cost_impact': 'High'
        })
    
    # Default strategy
    if not strategies:
        strategies.append({
            'risk': 'General Migration Management',
            'strategy': 'Follow AWS migration best practices and maintain regular checkpoints',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    return strategies


def get_fallback_risk_assessment() -> Dict:
    """Fallback risk assessment"""
    
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
# AI INSIGHTS
# ===========================

def generate_ai_insights_sync(cost_analysis: Dict, migration_params: Dict) -> Dict:
    """Generate AI insights synchronously"""
    
    if not ANTHROPIC_AVAILABLE:
        return {
            'error': 'Anthropic library not available',
            'fallback_insights': get_fallback_ai_insights(cost_analysis, migration_params)
        }
    
    api_key = migration_params.get('anthropic_api_key')
    if not api_key:
        return {
            'error': 'No Anthropic API key provided',
            'fallback_insights': get_fallback_ai_insights(cost_analysis, migration_params)
        }
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        context = f"""
        Migration Analysis:
        - Source: {migration_params.get('source_engine', 'Unknown')} 
        - Target: {migration_params.get('target_engine', 'Unknown')}
        - Data Size: {migration_params.get('data_size_gb', 0):,} GB
        - Timeline: {migration_params.get('migration_timeline_weeks', 0)} weeks
        - Monthly Cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
        - Migration Cost: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}
        
        Provide 3-4 key insights and recommendations for this database migration.
        """
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": context}]
        )
        
        return {
            'ai_analysis': message.content[0].text,
            'source': 'Claude AI',
            'model': 'claude-3-sonnet-20240229',
            'success': True
        }
        
    except Exception as e:
        return {
            'error': f'Claude AI failed: {str(e)}',
            'fallback_insights': get_fallback_ai_insights(cost_analysis, migration_params),
            'success': False
        }


def get_fallback_ai_insights(cost_analysis: Dict, migration_params: Dict) -> Dict:
    """Fallback AI insights when API is unavailable"""
    
    monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    data_size = migration_params.get('data_size_gb', 0)
    
    return {
        'cost_analysis': f"Monthly AWS cost of ${monthly_cost:,.0f} appears reasonable for this migration scale. Consider Reserved Instances for potential 30-40% savings.",
        'timeline_analysis': f"Timeline of {timeline_weeks} weeks allows for proper testing and validation phases for {data_size:,} GB of data.",
        'risk_assessment': "Migration complexity is manageable with proper planning and established AWS best practices.",
        'recommendations': [
            "Start with non-production environments for validation",
            "Implement comprehensive backup and rollback procedures", 
            "Use AWS DMS for minimal downtime migration",
            "Plan for performance optimization post-migration"
        ],
        'source': 'Enhanced Fallback Analysis'
    }


# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def create_growth_projection_charts(growth_analysis: Dict) -> List[go.Figure]:
    """Create growth projection visualizations"""
    
    charts = []
    projections = growth_analysis['yearly_projections']
    
    # 3-Year Cost Projection Chart
    years = ['Current', 'Year 1', 'Year 2', 'Year 3']
    annual_costs = [projections[f'year_{i}']['total_annual'] for i in range(4)]
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=years, y=annual_costs,
        mode='lines+markers',
        name='Annual Cost',
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
    
    # Cost component breakdown if available
    try:
        if projections['year_3']['environment_costs']:
            year_3_env = list(projections['year_3']['environment_costs'].values())[0]
            
            components = ['Compute', 'Storage', 'Backup']
            component_costs = [
                year_3_env.get('instance_cost', 0),
                year_3_env.get('storage_cost', 0),
                year_3_env.get('backup_cost', 0)
            ]
            
            if sum(component_costs) > 0:
                fig2 = go.Figure(data=[go.Pie(
                    labels=components,
                    values=component_costs,
                    hole=0.4,
                    textinfo='label+percent'
                )])
                
                fig2.update_layout(
                    title='Year 3 Cost Distribution',
                    height=400
                )
                
                charts.append(fig2)
    except Exception:
        pass
    
    return charts


def create_cost_breakdown_chart(analysis_results: Dict) -> go.Figure:
    """Create cost breakdown chart"""
    
    env_costs = analysis_results.get('environment_costs', {})
    
    if not env_costs:
        return None
    
    # Simple cost comparison
    env_names = list(env_costs.keys())
    monthly_costs = []
    
    for env_name in env_names:
        costs = env_costs[env_name]
        if isinstance(costs, dict):
            total_cost = costs.get('total_monthly', sum([
                costs.get('instance_cost', 0),
                costs.get('storage_cost', 0),
                costs.get('backup_cost', 0)
            ]))
        else:
            total_cost = float(costs) if costs else 0
        monthly_costs.append(total_cost)
    
    fig = go.Figure(data=[
        go.Bar(x=env_names, y=monthly_costs, marker_color='#3182ce')
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

def generate_executive_summary_pdf(analysis_results: Dict, migration_params: Dict) -> Optional[io.BytesIO]:
    """Generate executive summary PDF"""
    
    if not REPORTLAB_AVAILABLE:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("AWS Database Migration Analysis", styles['Title']))
        story.append(Paragraph("Executive Summary Report", styles['Heading2']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key metrics
        migration_costs = analysis_results.get('migration_costs', {})
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Monthly AWS Cost', f"${analysis_results.get('monthly_aws_cost', 0):,.0f}"],
            ['Annual AWS Cost', f"${analysis_results.get('annual_aws_cost', 0):,.0f}"],
            ['Migration Investment', f"${migration_costs.get('total', 0):,.0f}"],
            ['Timeline', f"{migration_params.get('migration_timeline_weeks', 0)} weeks"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Key Recommendations", styles['Heading2']))
        recommendations = [
            "Proceed with phased migration approach",
            "Implement AWS DMS for data synchronization",
            "Plan comprehensive testing for each environment",
            "Consider Aurora for production workloads"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
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
        'growth_analysis': None,
        'enhanced_analysis_results': None,
        'enhanced_recommendations': None
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
    
    # Configuration method
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


def show_bulk_upload_interface():
    """Show bulk upload interface"""
    
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
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
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
        
        count = 0
        for env_name, specs in envs.items():
            if count < 4:
                cpu_cores = specs.get('cpu_cores', 'N/A')
                ram_gb = specs.get('ram_gb', 'N/A')
                st.markdown(f"• **{env_name}:** {cpu_cores} cores, {ram_gb} GB RAM")
                count += 1
        
        if len(envs) > 4:
            st.markdown(f"• ... and {len(envs) - 4} more environments")
    
    # Run analysis
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        st.session_state.analysis_results = None
        
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
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: Growth Analysis
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
                ai_insights = generate_ai_insights_sync(cost_analysis, st.session_state.migration_params)
                st.session_state.ai_insights = ai_insights
                
                if ai_insights.get('success'):
                    st.success("✅ AI insights generated successfully!")
                else:
                    st.warning("⚠️ Using fallback insights")
                    
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
        st.session_state.risk_assessment = get_fallback_risk_assessment()


def show_analysis_summary():
    """Show analysis summary"""
    
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
            st.metric("Annual Cost", f"${results['annual_aws_cost']:,.0f}")
    
    with col4:
        if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
            risk_level = st.session_state.risk_assessment['risk_level']['level']
            st.metric("Risk Level", risk_level)
        else:
            st.metric("Environments", len(st.session_state.environment_specs))
    
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
        "📈 Growth Projections",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "📊 Visualizations",
        "🤖 AI Insights"
    ])
    
    with tab1:
        show_basic_cost_summary()
    
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


def show_basic_cost_summary():
    """Show basic cost summary"""
    
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
                if isinstance(costs, dict):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Instance Cost", f"${costs.get('instance_cost', 0):,.2f}/month")
                    
                    with col2:
                        st.metric("Storage Cost", f"${costs.get('storage_cost', 0):,.2f}/month")
                    
                    with col3:
                        st.metric("Total Monthly", f"${costs.get('total_monthly', 0):,.2f}")


def show_growth_analysis_dashboard():
    """Show growth analysis dashboard"""
    
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
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
        for i, chart in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True, key=f"growth_chart_{i}")
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")


def show_risk_assessment_tab():
    """Show risk assessment results"""
    
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        st.markdown("### ⚠️ Migration Risk Assessment")
        
        risk_assessment = st.session_state.risk_assessment
        risk_level = risk_assessment.get('risk_level', {'level': 'Unknown', 'color': '#666666'})
        overall_score = risk_assessment.get('overall_score', 50)
        
        # Overall risk display
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {risk_level['color']};">
            <div style="font-size: 2rem; font-weight: bold; color: {risk_level['color']};">
                {risk_level['level']} Risk
            </div>
            <div style="font-size: 1.1rem; color: #666;">
                Overall Score: {overall_score:.1f}/100
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Technical Risks")
            tech_risks = risk_assessment.get('technical_risks', {})
            for risk_name, score in tech_risks.items():
                st.metric(risk_name.replace('_', ' ').title(), f"{score:.0f}/100")
        
        with col2:
            st.markdown("#### 💼 Business Risks")
            business_risks = risk_assessment.get('business_risks', {})
            for risk_name, score in business_risks.items():
                st.metric(risk_name.replace('_', ' ').title(), f"{score:.0f}/100")
        
        # Mitigation strategies
        st.markdown("#### 🛡️ Mitigation Strategies")
        strategies = risk_assessment.get('mitigation_strategies', [])
        
        for strategy in strategies:
            with st.expander(f"🎯 {strategy.get('risk', 'Strategy')}"):
                st.write(f"**Strategy:** {strategy.get('strategy', 'Not specified')}")
                st.write(f"**Timeline:** {strategy.get('timeline', 'TBD')}")
                st.write(f"**Cost Impact:** {strategy.get('cost_impact', 'Unknown')}")
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")


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
                    
                    # Show recommendations if available
                    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
                        rec = st.session_state.recommendations.get(env_name, {})
                        if rec:
                            st.write(f"Recommended Instance: {rec.get('instance_class', 'N/A')}")
                            st.write(f"Multi-AZ: {'Yes' if rec.get('multi_az', False) else 'No'}")


def show_visualizations_tab():
    """Show visualization charts"""
    
    st.markdown("### 📊 Cost & Performance Visualizations")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Visualizations not available. Please run the migration analysis first.")
        return
    
    try:
        # Cost breakdown chart
        cost_chart = create_cost_breakdown_chart(st.session_state.analysis_results)
        if cost_chart:
            st.plotly_chart(cost_chart, use_container_width=True, key="cost_breakdown_viz")
        
        # Growth visualization if available
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            st.markdown("#### 📈 Growth Projections")
            try:
                charts = create_growth_projection_charts(st.session_state.growth_analysis)
                for i, chart in enumerate(charts):
                    st.plotly_chart(chart, use_container_width=True, key=f"viz_growth_{i}")
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
            st.warning(f"Using enhanced fallback insights")
            st.error(f"AI Error: {insights['error']}")
            
            # Show fallback insights
            fallback = insights.get('fallback_insights', {})
            if fallback:
                if 'cost_analysis' in fallback:
                    st.markdown("#### 💰 Cost Analysis")
                    st.write(fallback['cost_analysis'])
                
                if 'recommendations' in fallback:
                    st.markdown("#### 💡 Recommendations")
                    for rec in fallback['recommendations']:
                        st.markdown(f"• {rec}")
        else:
            # Show real AI insights
            if 'ai_analysis' in insights:
                st.markdown("#### 🧠 AI Analysis")
                st.write(insights['ai_analysis'])
            
            if insights.get('source') == 'Claude AI':
                st.success("✅ Generated by Claude AI")
            else:
                st.info("💡 Enhanced analysis")
    else:
        st.info("🤖 AI insights not available. Provide an Anthropic API key in the configuration to enable AI-powered analysis.")


def show_reports_section():
    """Show reports and export section"""
    
    st.markdown("## 📄 Reports & Export")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        return
    
    # Report options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👔 Executive Summary")
        st.markdown("Perfect for stakeholders and decision makers")
        
        if st.button("📄 Generate Executive PDF", key="exec_pdf", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                try:
                    pdf_buffer = generate_executive_summary_pdf(
                        st.session_state.analysis_results,
                        st.session_state.migration_params
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Executive Summary",
                            data=pdf_buffer.getvalue(),
                            file_name=f"AWS_Migration_Executive_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("PDF generation failed (ReportLab not available)")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col2:
        st.markdown("#### 📊 Data Export")
        st.markdown("Raw data for further analysis")
        
        if st.button("📊 Export Data (CSV)", key="csv_export", use_container_width=True):
            try:
                # Prepare CSV data
                results = st.session_state.analysis_results
                env_costs = results.get('environment_costs', {})
                
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
                else:
                    st.error("No data available for export")
            except Exception as e:
                st.error(f"Error preparing CSV: {str(e)}")
    
    with col3:
        st.markdown("#### ℹ️ Analysis Summary")
        st.markdown("Key metrics and insights")
        
        results = st.session_state.analysis_results
        
        st.metric("Monthly Cost", f"${results.get('monthly_aws_cost', 0):,.0f}")
        st.metric("Migration Cost", f"${results.get('migration_costs', {}).get('total', 0):,.0f}")
        
        if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
            risk_level = st.session_state.risk_assessment['risk_level']['level']
            st.metric("Risk Level", risk_level)


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
    
    # Main content based on selected page
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