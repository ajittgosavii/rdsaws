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
    
    # Main content
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