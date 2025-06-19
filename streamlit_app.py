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

# --- AWS Pricing API Class ---
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

        # Enhanced pricing data with Multi-AZ and Aurora support (simplified for example)
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

# --- Migration Analyzer Class ---
class MigrationAnalyzer:
    """Basic migration analyzer for standard environment configurations"""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = EnhancedAWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key

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

    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs (placeholder for now)"""
        # This is a simplified calculation. Real transfer costs are more complex.
        region = migration_params.get('region', 'us-east-1')
        egress_cost_per_gb = 0.09  # Example: US East (N. Virginia) to internet
        total_transfer_cost = data_size_gb * egress_cost_per_gb
        return {'total': total_transfer_cost, 'details': f'{data_size_gb} GB @ ${egress_cost_per_gb}/GB'}

    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate cost for a single environment"""
        # Get pricing
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )

        # Calculate monthly hours
        daily_usage_hours = rec.get('daily_usage_hours', 24)
        monthly_hours = daily_usage_hours * 30.44  # Average days in a month

        instance_cost = pricing['hourly'] * monthly_hours
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        # Assuming IOPS cost is per GB, simplified
        iops_cost = rec['storage_gb'] * pricing['iops_gb']
        # Backup cost simplified as a percentage of storage
        backup_cost = storage_cost * 0.2

        total_monthly = instance_cost + storage_cost + iops_cost + backup_cost

        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'iops_cost': iops_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly
        }


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

        # Migration service costs
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)

        # DMS costs (simplified)
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks  # t3.large instance

        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)

        # Professional services (simplified)
        ps_cost = migration_timeline_weeks * 8000  # $8k per week

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
        if not self.anthropic_api_key:
            return {'error': 'No Anthropic API key provided', 'source': 'Error'}
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
        except ImportError:
            return {'error': 'Run: pip install anthropic', 'source': 'Library Error', 'success': False}
        except Exception as e:
            return {'error': f'Claude AI failed: {str(e)}', 'source': 'API Error', 'success': False}

# --- Placeholder for GrowthAwareCostAnalyzer ---
class GrowthAwareCostAnalyzer:
    """Placeholder for GrowthAwareCostAnalyzer class."""
    def calculate_3_year_growth_projection(self, analysis_results: Dict, migration_params: Dict) -> Dict:
        """
        Calculates a simplified 3-year growth projection.
        In a real scenario, this would involve more complex growth models.
        """
        annual_growth_rate = migration_params.get('annual_data_growth', 15) / 100
        current_annual_cost = analysis_results.get('annual_aws_cost', 0)

        # Simple compounding growth
        year1_cost = current_annual_cost * (1 + annual_growth_rate)
        year2_cost = year1_cost * (1 + annual_growth_rate)
        year3_cost = year2_cost * (1 + annual_growth_rate)

        total_3_year_growth_percent = ((year3_cost / current_annual_cost) - 1) * 100 if current_annual_cost else 0

        return {
            'growth_summary': {
                'total_3_year_growth_percent': total_3_year_growth_percent,
                'year_0_cost': current_annual_cost,
                'year_1_cost': year1_cost,
                'year_2_cost': year2_cost,
                'year_3_cost': year3_cost,
                'last_updated': datetime.now().isoformat()
            },
            'detailed_projection': [
                {'year': 0, 'cost': current_annual_cost},
                {'year': 1, 'cost': year1_cost},
                {'year': 2, 'cost': year2_cost},
                {'year': 3, 'cost': year3_cost},
            ]
        }

# --- Cost Calculation and Refresh Functions ---
def refresh_cost_calculations():
    """
    Main function to refresh all cost calculations and update the dollar values:
    - Monthly AWS Cost
    - Annual AWS Cost
    - Migration Cost
    - 3-Year Growth
    """
    try:
        if not st.session_state.get('migration_params'):
            st.error("❌ Migration parameters required. Please configure migration settings first.")
            return False
        if not st.session_state.get('environment_specs'):
            st.error("❌ Environment specifications required. Please configure environments first.")
            return False

        with st.spinner("🔄 Refreshing cost calculations..."):
            monthly_cost, annual_cost, migration_cost = refresh_basic_costs()
            growth_percentage = refresh_growth_analysis(monthly_cost, annual_cost)
            update_cost_session_state(monthly_cost, annual_cost, migration_cost, growth_percentage)
            display_refreshed_metrics(monthly_cost, annual_cost, migration_cost, growth_percentage)
            st.success("✅ Cost calculations refreshed successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Error refreshing costs: {str(e)}")
        return False

def refresh_basic_costs():
    """Refresh basic AWS and migration cost calculations"""
    analyzer = MigrationAnalyzer(st.session_state.migration_params.get('anthropic_api_key'))
    recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
    st.session_state.recommendations = recommendations
    cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
    st.session_state.analysis_results = cost_analysis

    monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
    annual_cost = cost_analysis.get('annual_aws_cost', monthly_cost * 12)
    migration_cost = cost_analysis.get('migration_costs', {}).get('total', 0)
    return monthly_cost, annual_cost, migration_cost

def refresh_growth_analysis(monthly_cost: float, annual_cost: float):
    """Refresh 3-year growth analysis and calculate growth percentage"""
    try:
        growth_analyzer = GrowthAwareCostAnalyzer()
        growth_analysis = growth_analyzer.calculate_3_year_growth_projection(
            st.session_state.analysis_results,
            st.session_state.migration_params
        )
        st.session_state.growth_analysis = growth_analysis
        growth_percentage = growth_analysis['growth_summary']['total_3_year_growth_percent']
        return growth_percentage
    except Exception as e:
        st.warning(f"Growth analysis failed, using default: {str(e)}")
        annual_growth_rate = st.session_state.migration_params.get('annual_data_growth', 15)
        growth_percentage = ((1 + annual_growth_rate/100) ** 3 - 1) * 100
        return growth_percentage

def update_cost_session_state(monthly_cost: float, annual_cost: float,
                             migration_cost: float, growth_percentage: float):
    """Update session state with refreshed cost values"""
    if not st.session_state.get('analysis_results'):
        st.session_state.analysis_results = {}
    st.session_state.analysis_results.update({
        'monthly_aws_cost': monthly_cost,
        'annual_aws_cost': annual_cost,
        'migration_costs': {
            'total': migration_cost,
            'last_updated': datetime.now().isoformat()
        }
    })
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.session_state.growth_analysis = {
            'growth_summary': {
                'total_3_year_growth_percent': growth_percentage,
                'year_0_cost': annual_cost,
                'last_updated': datetime.now().isoformat()
            }
        }
    else:
        st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent'] = growth_percentage

def display_refreshed_metrics(monthly_cost: float, annual_cost: float,
                             migration_cost: float, growth_percentage: float):
    """Display the refreshed cost metrics in a formatted layout"""
    st.markdown("### 💰 Refreshed Cost Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Monthly AWS Cost",
            value=f"${monthly_cost:,.0f}",
            delta=f"Updated {datetime.now().strftime('%H:%M')}"
        )
    with col2:
        st.metric(
            label="Annual AWS Cost",
            value=f"${annual_cost:,.0f}",
            delta=f"${monthly_cost * 12:,.0f}/year"
        )
    with col3:
        st.metric(
            label="Migration Cost",
            value=f"${migration_cost:,.0f}",
            delta="One-time investment"
        )
    with col4:
        st.metric(
            label="3-Year Growth",
            value=f"{growth_percentage:.1f}%",
            delta="Projected growth"
        )

def refresh_specific_environment_costs(environment_name: str):
    """Refresh costs for a specific environment"""
    if environment_name not in st.session_state.get('environment_specs', {}):
        st.error(f"Environment '{environment_name}' not found")
        return

    env_specs = {environment_name: st.session_state.environment_specs[environment_name]}
    analyzer = MigrationAnalyzer()
    recommendations = analyzer.calculate_instance_recommendations(env_specs)
    cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)

    if st.session_state.get('analysis_results'):
        if 'environment_costs' not in st.session_state.analysis_results:
            st.session_state.analysis_results['environment_costs'] = {}
        st.session_state.analysis_results['environment_costs'][environment_name] = \
            cost_analysis['environment_costs'][environment_name]
    st.success(f"✅ Refreshed costs for {environment_name}")

def auto_refresh_costs():
    """Automatic cost refresh with real-time pricing"""
    st.markdown("### 🔄 Auto-Refresh Cost Analysis")
    auto_refresh = st.checkbox("Enable Auto-Refresh (every 30 seconds)", value=False)
    if auto_refresh:
        import time
        placeholder = st.empty()
        while auto_refresh:
            with placeholder.container():
                refresh_cost_calculations()
                st.write(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(30)

def export_refreshed_costs():
    """Export the refreshed cost data to CSV"""
    if not st.session_state.get('analysis_results'):
        st.warning("No cost data available to export")
        return

    export_data = {
        'Metric': ['Monthly AWS Cost', 'Annual AWS Cost', 'Migration Cost', '3-Year Growth'],
        'Value': [
            f"${st.session_state.analysis_results.get('monthly_aws_cost', 0):,.0f}",
            f"${st.session_state.analysis_results.get('annual_aws_cost', 0):,.0f}",
            f"${st.session_state.analysis_results.get('migration_costs', {}).get('total', 0):,.0f}",
            f"{st.session_state.get('growth_analysis', {}).get('growth_summary', {}).get('total_3_year_growth_percent', 0):.1f}%"
        ],
        'Last_Updated': [datetime.now().isoformat()] * 4
    }
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Refreshed Costs (CSV)",
        data=csv,
        file_name=f"refreshed_costs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# --- Streamlit UI Integration Functions ---
def integrate_cost_refresh_ui():
    """Add cost refresh UI elements to the main application"""
    st.markdown("---")
    st.markdown("### 🔄 Cost Refresh Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh All Costs", type="primary", use_container_width=True):
            refresh_cost_calculations()
    with col2:
        if st.button("📊 Refresh Growth Analysis", use_container_width=True):
            if st.session_state.get('analysis_results'):
                monthly_cost = st.session_state.analysis_results.get('monthly_aws_cost', 0)
                annual_cost = st.session_state.analysis_results.get('annual_aws_cost', 0)
                growth_percentage = refresh_growth_analysis(monthly_cost, annual_cost)
                st.success(f"✅ Growth updated: {growth_percentage:.1f}%")
            else:
                st.warning("Please run full analysis first")
    with col3:
        if st.button("📥 Export Costs", use_container_width=True):
            export_refreshed_costs()

    if st.session_state.get('environment_specs'):
        st.markdown("#### 🏢 Environment-Specific Refresh")
        selected_env = st.selectbox(
            "Select Environment to Refresh",
            list(st.session_state.environment_specs.keys())
        )
        if st.button(f"🔄 Refresh {selected_env}", use_container_width=True):
            refresh_specific_environment_costs(selected_env)

def main_cost_refresh_section():
    """Main section to be added to your Streamlit app"""
    st.markdown("## 💰 Cost Analysis & Refresh")
    if not st.session_state.get('analysis_results'):
        st.info("👆 Please run the migration analysis first to see cost data")
        return

    results = st.session_state.analysis_results
    st.markdown("#### 📊 Current Cost Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        monthly_cost = results.get('monthly_aws_cost', 0)
        st.metric("Monthly AWS Cost", f"${monthly_cost:,.0f}")
    with col2:
        annual_cost = results.get('annual_aws_cost', monthly_cost * 12)
        st.metric("Annual AWS Cost", f"${annual_cost:,.0f}")
    with col3:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    with col4:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_pct = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_pct:.1f}%")
        else:
            st.metric("3-Year Growth", "Not calculated")
    integrate_cost_refresh_ui()

def show_enhanced_environment_analysis():
    """Show enhanced environment analysis with Writer/Reader details"""
    st.markdown("### 🏢 Enhanced Environment Analysis")
    recommendations = st.session_state.get('enhanced_recommendations', {})
    environment_specs = st.session_state.get('environment_specs', {})

    env_comparison_data = []
    for env_name, rec in recommendations.items():
        specs = environment_specs.get(env_name, {})
        writer_config = f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})"
        reader_count = rec['readers']['count']
        reader_config = f"{reader_count} x {rec['readers']['instance_class']}" if reader_count > 0 else "No readers"

        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec['environment_type'].title(),
            'Current Resources': f"{specs.get('cpu_cores', 'N/A')} cores, {specs.get('ram_gb', 'N/A')} GB RAM",
            'Writer Instance': writer_config,
            'Read Replicas': reader_config,
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'Workload Pattern': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })

    if env_comparison_data:
        env_df = pd.DataFrame(env_comparison_data)
        st.dataframe(env_df, use_container_width=True)
    else:
        st.info("No enhanced environment analysis data available.")

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

# --- Placeholder for VRopsMetricsAnalyzer and associated functions ---
class VRopsMetricsAnalyzer:
    """Placeholder for VRopsMetricsAnalyzer class."""
    def __init__(self):
        pass

def create_vrops_sample_template() -> pd.DataFrame:
    """Creates a sample vROps metrics template dataframe."""
    data = {
        'VM_Name': ['DB-PROD-01', 'DB-PROD-02'],
        'Environment': ['Production', 'Production'],
        'vCPU_Cores_Avg': [8, 16],
        'RAM_GB_Avg': [32, 64],
        'Storage_GB_Allocated': [500, 1000],
        'IOPS_Read_Avg': [500, 1000],
        'IOPS_Write_Avg': [200, 400],
        'Network_Throughput_Mbps_Avg': [100, 200],
        'DB_Engine': ['PostgreSQL', 'PostgreSQL'],
        'Connections_Peak': [500, 1000]
    }
    return pd.DataFrame(data)

def process_vrops_data(df: pd.DataFrame, analyzer: VRopsMetricsAnalyzer) -> Dict:
    """Processes vROps data to extract environment specifications."""
    processed_environments = {}
    for index, row in df.iterrows():
        env_name = row['VM_Name']
        processed_environments[env_name] = {
            'cpu_cores': row.get('vCPU_Cores_Avg', 0),
            'ram_gb': row.get('RAM_GB_Avg', 0),
            'storage_gb': row.get('Storage_GB_Allocated', 0),
            'iops_requirement': row.get('IOPS_Read_Avg', 0) + row.get('IOPS_Write_Avg', 0),
            'peak_connections': row.get('Connections_Peak', 0),
            'daily_usage_hours': 24, # Assume 24/7 for now
            'network_throughput_mbps': row.get('Network_Throughput_Mbps_Avg', 0),
            'database_engine': row.get('DB_Engine', 'Unknown')
        }
    return processed_environments

def show_vrops_processing_summary(processed_environments: Dict, analyzer: VRopsMetricsAnalyzer):
    """Shows a summary of processed vROps environments."""
    st.markdown("#### ✅ Processed Environments Summary")
    summary_data = []
    for env_name, specs in processed_environments.items():
        summary_data.append({
            'Environment': env_name,
            'CPU (Cores)': specs['cpu_cores'],
            'RAM (GB)': specs['ram_gb'],
            'Storage (GB)': specs['storage_gb'],
            'Peak IOPS': specs['iops_requirement']
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# --- Enhanced Environment Setup UI ---
def show_enhanced_environment_setup_with_vrops():
    """Enhanced environment setup with vROps integration"""
    st.markdown("## 📊 Enhanced Environment Configuration")
    if not st.session_state.get('migration_params'):
        st.warning("⚠️ Please complete Migration Configuration first.")
        return

    if 'vrops_analyzer' not in st.session_state:
        st.session_state.vrops_analyzer = VRopsMetricsAnalyzer()
    analyzer = st.session_state.vrops_analyzer

    config_method = st.radio(
        "Choose configuration method:",
        [
            "📊 vROps Metrics Import",
            "📝 Manual Detailed Entry", # Placeholder for manual entry
            "📁 Bulk CSV Upload", # Renamed for clarity
            "🔄 Simple Configuration (Legacy)" # Placeholder for simple config
        ],
        horizontal=True
    )

    if config_method == "📊 vROps Metrics Import":
        show_vrops_import_interface(analyzer)
    elif config_method == "📁 Bulk CSV Upload":
        show_enhanced_bulk_upload(analyzer)
    elif config_method == "📝 Manual Detailed Entry":
        st.info("Manual detailed entry interface will be here.")
    elif config_method == "🔄 Simple Configuration (Legacy)":
        st.info("Simple configuration interface will be here.")

def show_vrops_import_interface(analyzer: VRopsMetricsAnalyzer):
    """Show vROps metrics import interface"""
    st.markdown("### 📊 vROps Metrics Import")
    with st.expander("📋 Download vROps Export Template", expanded=False):
        st.markdown("""
        **vROps Data Collection Instructions:**
        1. **Export Performance Data** from vROps for your database VMs
        2. **Time Period**: Minimum 30 days, recommended 90 days
        3. **Metrics to Include**: Use the template below or export all available metrics
        4. **Format**: CSV export with hourly or daily aggregation
        """)
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

    uploaded_file = st.file_uploader(
        "Upload vROps Export File", type=['csv', 'xlsx'], help="Upload your vROps performance metrics export"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            st.markdown("#### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("#### 🔗 Map vROps Metrics to Standard Fields")
            processed_environments = process_vrops_data(df, analyzer)
            if processed_environments:
                st.session_state.environment_specs = processed_environments
                st.success(f"✅ Successfully processed {len(processed_environments)} environments!")
                show_vrops_processing_summary(processed_environments, analyzer)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.code(str(e))

def show_enhanced_bulk_upload(analyzer: VRopsMetricsAnalyzer):
    """Show enhanced bulk upload with comprehensive template"""
    st.markdown("### 📁 Enhanced Bulk Upload")
    with st.expander("📋 Download Comprehensive Template", expanded=False):
        template_data = create_vrops_sample_template() # Re-using vROps template for consistency
        csv_data = template_data.to_csv(index=False)
        st.dataframe(template_data, use_container_width=True)
        st.download_button(
            label="📥 Download Performance Metrics Template",
            data=csv_data,
            file_name="performance_metrics_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    uploaded_file = st.file_uploader(
        "Upload Performance Data", type=['csv', 'xlsx'], help="Upload CSV or Excel file with performance metrics"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
            environments = process_vrops_data(df, analyzer)
            if environments:
                st.session_state.environment_specs = environments
                st.success(f"✅ Processed {len(environments)} environments!")
                show_vrops_processing_summary(environments, analyzer)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# --- Risk Assessment and Analysis Summary (Simplified) ---
def get_fallback_risk_assessment() -> Dict:
    """Returns a fallback risk assessment."""
    return {
        'technical_risks': {'data_corruption': 30, 'downtime': 40, 'performance_degradation': 35},
        'business_risks': {'cost_overruns': 45, 'timeline_delays': 50, 'resource_availability': 30}
    }

def show_analysis_summary():
    """Show analysis summary after completion"""
    st.markdown("#### 🎯 Analysis Summary")
    col1, col2, col3 = st.columns(3)

    results = st.session_state.get('enhanced_analysis_results', st.session_state.get('analysis_results'))

    if results:
        with col1:
            st.metric("Monthly Cost", f"${results.get('monthly_aws_cost', 0):,.0f}")
        with col2:
            migration_cost = results.get('migration_costs', {}).get('total', 0)
            st.metric("Migration Cost", f"${migration_cost:,.0f}")
        with col3:
            if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
                growth_pct = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
                st.metric("3-Year Growth", f"{growth_pct:.1f}%")
            else:
                st.metric("3-Year Growth", "Not calculated")

    if not hasattr(st.session_state, 'risk_assessment') or st.session_state.risk_assessment is None:
        st.session_state.risk_assessment = get_fallback_risk_assessment()

    risk_assessment = st.session_state.risk_assessment
    st.markdown("#### ⚠️ Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚙️ Technical Risks")
        technical_risks = risk_assessment.get('technical_risks', {})
        for risk_name, score in technical_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#48bb78'
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd;">
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
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#48bb78'
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd;">
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

def add_realtime_cost_widget():
    """Add a real-time cost monitoring widget to any page"""
    if st.session_state.get('analysis_results'):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 3])
            with col1:
                auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=False)
            with col2:
                if st.button("🔄 Refresh Now", type="primary"):
                    refresh_cost_calculations()
                    st.experimental_rerun() # This will rerun the entire app
            with col3:
                last_updated = st.session_state.analysis_results.get('migration_costs', {}).get('last_updated', 'Unknown')
                if last_updated != 'Unknown':
                    last_time = datetime.fromisoformat(last_updated).strftime('%H:%M:%S')
                    st.caption(f"Updated: {last_time}")
            if auto_refresh:
                import time
                time.sleep(30)
                refresh_cost_calculations()
                st.experimental_rerun()

# --- PDF Report Generation Helper Class and Functions ---
class PDFReportGenerator:
    """Helper class for generating various sections of the PDF report."""
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=colors.HexColor('#2d3748')
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            fontName='Helvetica-Bold',
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2d3748')
        ))
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            fontName='Helvetica-Bold',
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#4a5568')
        ))
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            spaceAfter=6,
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name='KeyMetric',
            parent=self.styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2b6cb0'),
            spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Helvetica-Bold',
            textColor=colors.whitesmoke,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            leftIndent=20,
            firstLineIndent=-10,
            spaceAfter=4,
            bulletText='•'
        ))
        self.chart_width = 400
        self.chart_height = 250
        self._verify_styles()

    def _verify_styles(self):
        """Verify that all required styles are available and create fallbacks if not."""
        required_styles = [
            'ReportTitle', 'SectionHeader', 'SubsectionHeader', 'BodyText',
            'KeyMetric', 'TableHeader', 'BulletPoint'
        ]
        for style_name in required_styles:
            if style_name not in self.styles.byName:
                print(f"Warning: Style '{style_name}' not found. Creating fallback.")
                self._create_fallback_style(style_name)

    def _create_fallback_style(self, style_name):
        """Create a fallback style if the original creation failed"""
        try:
            base_style = self.styles['Normal']
            if style_name == 'ReportTitle':
                self.styles.add(ParagraphStyle(
                    name='ReportTitle', parent=self.styles['Title'], fontSize=24,
                    fontName='Helvetica-Bold', alignment=TA_CENTER
                ))
            elif style_name == 'SectionHeader':
                self.styles.add(ParagraphStyle(
                    name='SectionHeader', parent=self.styles['Heading1'], fontSize=16,
                    fontName='Helvetica-Bold', spaceBefore=20, spaceAfter=12
                ))
            else: # Generic fallback
                self.styles.add(ParagraphStyle(
                    name=style_name, parent=base_style, fontSize=12, fontName='Helvetica'
                ))
        except Exception as e:
            print(f"Error creating fallback style {style_name}: {e}")

    def _create_title_page(self, analysis_mode: str, analysis_results: Dict):
        """Create the title page for the PDF report."""
        story = []
        story.append(Paragraph("AWS Database Migration Analysis Report", self.styles['ReportTitle']))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['BodyText']))
        story.append(Spacer(1, 0.5 * inch))

        story.append(Paragraph("Analysis Mode:", self.styles['SubsectionHeader']))
        story.append(Paragraph(f"<b>{analysis_mode.replace('_', ' ').title()} Analysis</b>", self.styles['KeyMetric']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Key Financial Summary:", self.styles['SubsectionHeader']))
        monthly_cost = analysis_results.get('monthly_aws_cost', 0)
        annual_cost = analysis_results.get('annual_aws_cost', monthly_cost * 12)
        migration_total_cost = analysis_results.get('migration_costs', {}).get('total', 0)

        story.append(Paragraph(f"Estimated Monthly AWS Cost: <b>${monthly_cost:,.0f}</b>", self.styles['BodyText']))
        story.append(Paragraph(f"Estimated Annual AWS Cost: <b>${annual_cost:,.0f}</b>", self.styles['BodyText']))
        story.append(Paragraph(f"Total One-Time Migration Investment: <b>${migration_total_cost:,.0f}</b>", self.styles['BodyText']))
        story.append(Spacer(1, 1 * inch))
        return story

    def _create_executive_summary(self, analysis_results: Dict, analysis_mode: str, ai_insights: Optional[Dict]):
        """Create the executive summary section."""
        story = []
        story.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        story.append(Paragraph(
            "This report provides a comprehensive analysis of the proposed AWS database migration, "
            "including cost projections, environmental recommendations, and strategic insights.",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("1.1. Key Metrics Overview", self.styles['SubsectionHeader']))
        monthly_aws_cost = analysis_results.get('monthly_aws_cost', 0)
        annual_aws_cost = analysis_results.get('annual_aws_cost', 0)
        migration_total_cost = analysis_results.get('migration_costs', {}).get('total', 0)
        growth_pct = st.session_state.get('growth_analysis', {}).get('growth_summary', {}).get('total_3_year_growth_percent', 0)

        metrics_data = [
            ['Metric', 'Value'],
            ['Monthly AWS Cost', f"${monthly_aws_cost:,.0f}"],
            ['Annual AWS Cost', f"${annual_aws_cost:,.0f}"],
            ['Migration Investment', f"${migration_total_cost:,.0f}"],
            ['3-Year Projected Growth', f"{growth_pct:.1f}%"]
        ]
        metrics_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a0aec0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ])
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(metrics_table_style)
        story.append(metrics_table)
        story.append(Spacer(1, 0.2 * inch))

        # Add cost projection chart
        cost_chart = self.create_cost_projection_chart(analysis_results, st.session_state.get('growth_analysis', {}))
        if cost_chart:
            story.append(Paragraph("Annual Cost Projection", self.styles['SubsectionHeader']))
            story.append(cost_chart)
            story.append(Spacer(1, 0.2 * inch))

        if ai_insights and ai_insights.get('success'):
            story.append(Paragraph("1.2. AI-Powered Insights", self.styles['SubsectionHeader']))
            formatted_insights = self.format_ai_insights_text(ai_insights['ai_analysis'])
            for paragraph_text in formatted_insights:
                story.append(Paragraph(paragraph_text, self.styles['BodyText']))
                story.append(Spacer(1, 0.05 * inch))
        return story

    def _create_technical_analysis(self, results: Dict, recommendations: Dict, migration_params: Dict):
        """Create the technical analysis section."""
        story = []
        story.append(Paragraph("2. Technical Analysis & Recommendations", self.styles['SectionHeader']))

        story.append(Paragraph("2.1. Environment Specifications & Recommendations", self.styles['SubsectionHeader']))
        if hasattr(st.session_state, 'environment_specs') and st.session_state.environment_specs:
            for env_name, specs in st.session_state.environment_specs.items():
                story.append(Paragraph(f"Environment: <b>{env_name}</b>", self.styles['KeyMetric']))
                story.append(Paragraph(f"Current CPU Cores: {specs.get('cpu_cores', 'N/A')}", self.styles['BodyText']))
                story.append(Paragraph(f"Current RAM (GB): {specs.get('ram_gb', 'N/A')}", self.styles['BodyText']))
                story.append(Paragraph(f"Current Storage (GB): {specs.get('storage_gb', 'N/A')}", self.styles['BodyText']))

                rec = recommendations.get(env_name, {})
                if rec:
                    story.append(Paragraph("Recommended AWS Configuration:", self.styles['BodyText']))
                    if 'writer' in rec: # Enhanced recommendation format
                        story.append(Paragraph(f"  Writer Instance: {rec['writer'].get('instance_class', 'N/A')} ({'Multi-AZ' if rec['writer'].get('multi_az') else 'Single-AZ'})", self.styles['BodyText']))
                        if rec['readers'].get('count', 0) > 0:
                            story.append(Paragraph(f"  Read Replicas: {rec['readers']['count']} x {rec['readers'].get('instance_class', 'N/A')}", self.styles['BodyText']))
                        story.append(Paragraph(f"  Storage: {rec['storage'].get('size_gb', 'N/A')} GB {rec['storage'].get('type', '').upper()}", self.styles['BodyText']))
                    else: # Standard recommendation format
                        story.append(Paragraph(f"  Instance Type: {rec.get('instance_class', 'N/A')}", self.styles['BodyText']))
                        story.append(Paragraph(f"  Multi-AZ Deployment: {'Yes' if rec.get('multi_az') else 'No'}", self.styles['BodyText']))
                story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(Paragraph("No environment specifications available.", self.styles['BodyText']))
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("2.2. Detailed Cost Breakdown", self.styles['SubsectionHeader']))
        env_costs_data = [['Environment', 'Instance Cost (Monthly)', 'Storage Cost (Monthly)', 'Backup Cost (Monthly)', 'Total Monthly Cost']]
        for env_name, costs in results.get('environment_costs', {}).items():
            instance_cost = costs.get('instance_cost', costs.get('writer_instance_cost', 0)) + costs.get('reader_costs', 0)
            env_costs_data.append([
                env_name,
                f"${instance_cost:,.0f}",
                f"${costs.get('storage_cost', 0):,.0f}",
                f"${costs.get('backup_cost', 0):,.0f}",
                f"${costs.get('total_monthly', 0):,.0f}"
            ])
        if len(env_costs_data) > 1:
            cost_breakdown_table = Table(env_costs_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.3*inch])
            cost_breakdown_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#edf2f7')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a0aec0')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            story.append(cost_breakdown_table)
        else:
            story.append(Paragraph("No detailed environment cost breakdown available.", self.styles['BodyText']))
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("2.3. Migration Specific Costs", self.styles['SubsectionHeader']))
        migration_costs = results.get('migration_costs', {})
        story.append(Paragraph(f"DMS Instance Cost: <b>${migration_costs.get('dms_instance', 0):,.0f}</b>", self.styles['BodyText']))
        story.append(Paragraph(f"Data Transfer Cost: <b>${migration_costs.get('data_transfer', 0):,.0f}</b>", self.styles['BodyText']))
        story.append(Paragraph(f"Professional Services: <b>${migration_costs.get('professional_services', 0):,.0f}</b>", self.styles['BodyText']))
        story.append(Paragraph(f"Contingency: <b>${migration_costs.get('contingency', 0):,.0f}</b>", self.styles['BodyText']))
        story.append(Paragraph(f"<b>Total One-Time Migration Investment: ${migration_costs.get('total', 0):,.0f}</b>", self.styles['KeyMetric']))
        story.append(Spacer(1, 0.3 * inch))
        return story

    def _create_migration_timeline(self, migration_params: Dict):
        """Create the migration timeline section."""
        story = []
        story.append(Paragraph("3. Migration Timeline & Strategy", self.styles['SectionHeader']))
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        story.append(Paragraph(f"Projected migration timeline: <b>{timeline_weeks} weeks</b>.", self.styles['BodyText']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("3.1. Key Migration Phases", self.styles['SubsectionHeader']))
        phases = [
            {'name': 'Discovery & Planning', 'duration': 2, 'description': 'Assess current environment, define scope, and plan migration strategy.'},
            {'name': 'Schema Migration', 'duration': 3, 'description': 'Convert and migrate database schemas to AWS RDS/Aurora.'},
            {'name': 'Data Migration', 'duration': 4, 'description': 'Migrate data using AWS DMS or other methods.'},
            {'name': 'Application Testing', 'duration': 3, 'description': 'Thorough testing of applications with migrated databases.'},
            {'name': 'User Acceptance Testing (UAT)', 'duration': 2, 'description': 'Business user validation of migrated applications.'},
            {'name': 'Go-Live Preparation', 'duration': 1, 'description': 'Final checks and readiness for production cutover.'},
            {'name': 'Production Cutover', 'duration': 1, 'description': 'Switch production traffic to AWS databases.'}
        ]

        # Create a simple timeline chart using Plotly
        timeline_fig = go.Figure()
        current_week = 0
        for i, phase in enumerate(phases):
            timeline_fig.add_trace(go.Bar(
                name=phase['name'],
                x=[phase['name']],
                y=[phase['duration']],
                text=f"{phase['duration']} weeks",
                textposition='auto',
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            ))
        timeline_fig.update_layout(
            title_text='Migration Phase Duration',
            xaxis_title='Phase',
            yaxis_title='Duration (weeks)',
            showlegend=False,
            height=350,
            xaxis_tickangle=-45,
            margin=dict(l=50, r=50, b=100, t=50)
        )
        timeline_chart_img = self.create_plotly_chart_image(timeline_fig, width=600, height=350)
        if timeline_chart_img:
            story.append(timeline_chart_img)
            story.append(Spacer(1, 0.2 * inch))

        for phase in phases:
            story.append(Paragraph(f"<b>{phase['name']} ({phase['duration']} weeks):</b> {phase['description']}", self.styles['BodyText']))
            story.append(Spacer(1, 0.05 * inch))

        story.append(Spacer(1, 0.3 * inch))
        return story

    def create_plotly_chart_image(self, fig, width=600, height=400):
        """Convert Plotly figure to image for PDF inclusion with improved quality"""
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file.flush()
                img = Image(tmp_file.name, width=self.chart_width, height=self.chart_height)
            os.unlink(tmp_file.name) # Clean up the temporary file
            return img
        except Exception as e:
            print(f"Error creating chart image: {e}")
            return None

    def create_cost_projection_chart(self, analysis_results: Dict, growth_analysis: Dict):
        """Create a Plotly chart for annual cost projection."""
        if not growth_analysis or 'detailed_projection' not in growth_analysis:
            return None

        df_growth = pd.DataFrame(growth_analysis['detailed_projection'])
        df_growth['year_label'] = df_growth['year'].apply(lambda x: f"Year {x}")

        fig = px.line(
            df_growth,
            x='year_label',
            y='cost',
            title='Projected Annual AWS Cost Over 3 Years',
            labels={'year_label': 'Year', 'cost': 'Annual Cost ($)'},
            markers=True,
            height=400
        )
        fig.update_traces(mode='lines+markers', line=dict(color='blue', width=2), marker=dict(size=8, color='red'))
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Annual Cost ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return self.create_plotly_chart_image(fig, width=700, height=400)

    def format_ai_insights_text(self, ai_text, max_line_length=90):
        """Format AI insights text for better PDF display, ensuring proper breaking."""
        if not ai_text:
            return ["No AI insights available."]

        cleaned_text = ai_text.replace('...', '')
        paragraphs = []
        current_paragraph = []

        for line in cleaned_text.split('\n'):
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                continue

            words = line.split(' ')
            temp_line = []
            for word in words:
                if len(" ".join(temp_line) + " " + word) > max_line_length and temp_line:
                    current_paragraph.append(" ".join(temp_line))
                    temp_line = [word]
                else:
                    temp_line.append(word)
            if temp_line:
                current_paragraph.append(" ".join(temp_line))

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        # Join lines within a paragraph to form coherent blocks
        final_paragraphs = []
        for p in paragraphs:
            # Simple heuristic: if a paragraph is very short and follows a bullet, treat as continuation
            if p.strip().startswith(('1.', '2.', '3.', '4.', '•', '-')) or len(p.split()) > 5:
                final_paragraphs.append(p)
            elif final_paragraphs:
                final_paragraphs[-1] += " " + p
            else:
                final_paragraphs.append(p)
        return final_paragraphs

def generate_executive_summary_pdf_robust(results: Dict, migration_params: Dict) -> Optional[io.BytesIO]:
    """Generate executive summary PDF - ROBUST VERSION (consolidated)"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        report_gen = PDFReportGenerator()
        story = []

        # Add title page
        story.extend(report_gen._create_title_page("Executive Summary", results))
        story.append(PageBreak())

        # Add executive summary content
        ai_insights = None # Assuming AI insights are not directly passed here, need to fetch from session_state
        if st.session_state.get('ai_insights'):
            ai_insights = st.session_state.ai_insights
        story.extend(report_gen._create_executive_summary(results, "Executive Summary", ai_insights))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error generating executive summary PDF: {e}")
        return None

def generate_technical_report_pdf_robust(results: Dict, recommendations: Dict, migration_params: Dict) -> Optional[io.BytesIO]:
    """Generate technical report PDF - ROBUST VERSION (placeholder)"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        report_gen = PDFReportGenerator()
        story = []

        # Title Page
        story.extend(report_gen._create_title_page("Technical Analysis", results))
        story.append(PageBreak())

        # Technical Analysis Section
        story.extend(report_gen._create_technical_analysis(results, recommendations, migration_params))
        story.append(PageBreak())

        # Migration Timeline Section
        story.extend(report_gen._create_migration_timeline(migration_params))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error generating technical report PDF: {e}")
        return None

def prepare_csv_export_data(results: Dict, recommendations: Dict) -> Optional[pd.DataFrame]:
    """Prepare CSV data for export"""
    try:
        env_costs = results.get('environment_costs', {})
        if not env_costs:
            return None
        csv_data = []
        for env_name, costs in env_costs.items():
            instance_cost = costs.get('instance_cost', costs.get('writer_instance_cost', 0))
            storage_cost = costs.get('storage_cost', 0)
            backup_cost = costs.get('backup_cost', 0)
            total_monthly = costs.get('total_monthly', 0)

            reader_costs = costs.get('reader_costs', 0)
            reader_count = costs.get('reader_count', 0)

            # Get recommendation details for this environment
            rec = recommendations.get(env_name, {})
            recommended_instance = rec.get('instance_class', 'N/A')
            if 'writer' in rec: # Enhanced recommendation format
                recommended_instance = rec['writer'].get('instance_class', 'N/A')
                if rec['readers'].get('count', 0) > 0:
                    recommended_instance += f" + {rec['readers']['count']}x {rec['readers'].get('instance_class', 'N/A')} Readers"
            multi_az = 'Yes' if rec.get('multi_az', False) else 'No'
            env_type = rec.get('environment_type', 'Unknown').title()


            csv_data.append({
                'Environment': env_name,
                'Environment_Type': env_type,
                'Recommended_Instance': recommended_instance,
                'Multi_AZ': multi_az,
                'Instance_Cost_Monthly': instance_cost,
                'Reader_Costs_Monthly': reader_costs,
                'Reader_Count': reader_count,
                'Storage_Cost_Monthly': storage_cost,
                'Backup_Cost_Monthly': backup_cost,
                'Total_Monthly_Cost': total_monthly,
                'Total_Annual_Cost': total_monthly * 12
            })
        if csv_data:
            return pd.DataFrame(csv_data)
        else:
            return None
    except Exception as e:
        print(f"Error preparing CSV data: {e}")
        return None

def create_bulk_reports_zip(results: Dict, recommendations: Dict, migration_params: Dict) -> Optional[io.BytesIO]:
    """Create a ZIP file containing all generated reports."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Executive Summary PDF
        exec_pdf = generate_executive_summary_pdf_robust(results, migration_params)
        if exec_pdf:
            zf.writestr(f"AWS_Migration_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf", exec_pdf.getvalue())
        # Technical Report PDF
        tech_pdf = generate_technical_report_pdf_robust(results, recommendations, migration_params)
        if tech_pdf:
            zf.writestr(f"AWS_Migration_Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf", tech_pdf.getvalue())
        # CSV Data
        csv_data_df = prepare_csv_export_data(results, recommendations)
        if csv_data_df is not None:
            csv_string = csv_data_df.to_csv(index=False)
            zf.writestr(f"AWS_Migration_Analysis_Data_{datetime.now().strftime('%Y%m%d')}.csv", csv_string)
    zip_buffer.seek(0)
    return zip_buffer

# --- Main Streamlit App Functions (Integration Points) ---
def validate_growth_analysis_functions():
    """Validate that all required growth analysis functions are defined."""
    required_functions = [
        'refresh_cost_calculations',
        'refresh_growth_analysis',
        'update_cost_session_state',
        'display_refreshed_metrics',
        'integrate_cost_refresh_ui',
        'main_cost_refresh_section',
        # Add other functions as they become part of the growth analysis workflow
        'show_analysis_summary', # Assuming this is part of dashboard for metrics
        # 'show_vrops_dashboard', # This seems to be a UI function, not a calculation
        'show_risk_assessment_tab', # UI function
        'show_environment_analysis_tab', # UI function
        'show_visualizations_tab', # UI function
        'show_ai_insights_tab', # UI function
        'show_timeline_analysis_tab', # UI function
        'create_growth_projection_charts' # Needs to be defined or removed
    ]

    missing_functions = []
    # This check needs to be more robust, as 'globals()' might not contain all functions
    # due to how Streamlit re-executes scripts. For a real app, this check might be
    # more about ensuring class methods or specific helper functions are callable.
    # For now, we'll check directly in this scope.
    defined_functions = globals()
    for func_name in required_functions:
        if func_name not in defined_functions or not callable(defined_functions[func_name]):
            missing_functions.append(func_name)

    if missing_functions:
        st.error(f"❌ Missing functions: {', '.join(missing_functions)}. Please ensure all parts of the application are loaded.")
        return False
    else:
        st.success("✅ All growth analysis functions are properly defined!")
        return True

# Dummy functions for validation or future implementation
def show_vrops_dashboard():
    st.markdown("### vROps Dashboard (Placeholder)")
    st.info("This section would display vROps related dashboards.")

def show_risk_assessment_tab():
    st.markdown("### Risk Assessment (Placeholder)")
    show_analysis_summary() # Re-using existing function for content

def show_environment_analysis_tab():
    st.markdown("### Environment Analysis (Placeholder)")
    show_enhanced_environment_analysis() # Re-using existing function for content

def show_visualizations_tab():
    st.markdown("### Visualizations (Placeholder)")
    st.info("Charts and graphs related to migration analysis would be displayed here.")
    if st.session_state.get('analysis_results') and st.session_state.get('growth_analysis'):
        report_gen = PDFReportGenerator()
        cost_chart = report_gen.create_cost_projection_chart(st.session_state.analysis_results, st.session_state.growth_analysis)
        if cost_chart:
            st.markdown("#### Annual Cost Projection")
            st.image(cost_chart.width, cost_chart.height) # Render the image bytes if possible, or embed directly

def show_ai_insights_tab():
    st.markdown("### AI Insights (Placeholder)")
    if st.session_state.get('ai_insights') and st.session_state.ai_insights.get('success'):
        st.write(st.session_state.ai_insights['ai_analysis'])
    else:
        st.info("No AI insights available or generation failed.")

def show_timeline_analysis_tab():
    st.markdown("### Timeline Analysis (Placeholder)")
    if st.session_state.get('migration_params'):
        report_gen = PDFReportGenerator()
        # To display, we'd need to recreate the plotly figure directly in Streamlit context
        # For simplicity, showing text summary
        st.markdown(f"Projected migration timeline: **{st.session_state.migration_params.get('migration_timeline_weeks', 0)} weeks**.")
        st.markdown("Phases:")
        phases = [
            {'name': 'Discovery & Planning', 'duration': 2},
            {'name': 'Schema Migration', 'duration': 3},
            {'name': 'Data Migration', 'duration': 4},
            {'name': 'Application Testing', 'duration': 3},
            {'name': 'User Acceptance Testing (UAT)', 'duration': 2},
            {'name': 'Go-Live Preparation', 'duration': 1},
            {'name': 'Production Cutover', 'duration': 1}
        ]
        for phase in phases:
            st.markdown(f"- {phase['name']}: {phase['duration']} weeks")
    else:
        st.info("Please configure migration parameters to see timeline.")

def create_growth_projection_charts():
    st.markdown("### Growth Projection Charts (Placeholder)")
    st.info("This section would contain detailed growth projection charts.")
    if st.session_state.get('analysis_results') and st.session_state.get('growth_analysis'):
        report_gen = PDFReportGenerator()
        cost_chart = report_gen.create_cost_projection_chart(st.session_state.analysis_results, st.session_state.growth_analysis)
        if cost_chart:
            st.markdown("#### Annual Cost Projection")
            st.image(cost_chart.width, cost_chart.height)

# --- Main Application Setup (assuming a `streamlit_app.py` context) ---
def setup_initial_session_state():
    if 'migration_params' not in st.session_state:
        st.session_state.migration_params = {
            'source_engine': 'PostgreSQL',
            'target_engine': 'PostgreSQL',
            'data_size_gb': 1000,
            'migration_timeline_weeks': 12,
            'annual_data_growth': 15,
            'region': 'us-east-1',
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_API_KEY')
        }
    if 'environment_specs' not in st.session_state:
        st.session_state.environment_specs = {
            'Prod_DB': {'cpu_cores': 8, 'ram_gb': 32, 'storage_gb': 1000, 'daily_usage_hours': 24, 'peak_connections': 500},
            'Dev_DB': {'cpu_cores': 2, 'ram_gb': 8, 'storage_gb': 200, 'daily_usage_hours': 8, 'peak_connections': 50}
        }
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'enhanced_recommendations' not in st.session_state:
        st.session_state.enhanced_recommendations = {}
    if 'growth_analysis' not in st.session_state:
        st.session_state.growth_analysis = None
    if 'risk_assessment' not in st.session_state:
        st.session_state.risk_assessment = None
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = None

def run_analysis_and_recommendations():
    st.markdown("## 🚀 Analysis & Recommendations")
    if st.button("🚀 Run Comprehensive Analysis"):
        with st.spinner("Running comprehensive analysis..."):
            try:
                # Ensure analyzer is initialized with API key
                analyzer = MigrationAnalyzer(st.session_state.migration_params.get('anthropic_api_key'))

                # Step 1: Calculate Enhanced Instance Recommendations (e.g., with Writer/Reader)
                # This part assumes a more complex 'calculate_instance_recommendations' that
                # returns 'writer', 'readers', 'storage' details directly.
                # For this simplified version, let's create a dummy enhanced_recommendations based on basic ones.
                basic_recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
                enhanced_recs = {}
                for env_name, rec in basic_recommendations.items():
                    enhanced_recs[env_name] = {
                        'environment_type': rec['environment_type'],
                        'workload_pattern': 'mixed', # Dummy
                        'read_write_ratio': 70, # Dummy
                        'connections': rec['peak_connections'],
                        'writer': {
                            'instance_class': rec['instance_class'],
                            'multi_az': rec['multi_az'],
                            'cpu_cores': rec['cpu_cores'],
                            'ram_gb': rec['ram_gb']
                        },
                        'readers': {
                            'count': 1 if rec['environment_type'] == 'production' else 0, # Add a reader for production
                            'instance_class': rec['instance_class'],
                            'multi_az': rec['multi_az']
                        },
                        'storage': {
                            'size_gb': rec['storage_gb'],
                            'type': 'gp3',
                            'iops': rec['storage_gb'] * 3, # Dummy IOPS
                            'encrypted': True,
                            'backup_retention_days': 7
                        }
                    }
                st.session_state.enhanced_recommendations = enhanced_recs

                # Step 2: Calculate migration costs with enhanced recommendations
                # The existing calculate_migration_costs would need to be updated to consume the enhanced_recs format
                # For now, we'll keep it as is, or adjust it to work with a simplified enhanced recs structure.
                # Let's use the basic recommendations for cost calculation if the MigrationAnalyzer expects that.
                cost_analysis = analyzer.calculate_migration_costs(basic_recommendations, st.session_state.migration_params)
                st.session_state.enhanced_analysis_results = cost_analysis # Store as enhanced results

                # Step 3: Run growth analysis
                growth_analyzer = GrowthAwareCostAnalyzer()
                growth_analysis = growth_analyzer.calculate_3_year_growth_projection(
                    st.session_state.enhanced_analysis_results, st.session_state.migration_params
                )
                st.session_state.growth_analysis = growth_analysis

                # Step 4: Generate AI Insights
                ai_insights = analyzer.generate_ai_insights_sync(
                    st.session_state.enhanced_analysis_results, st.session_state.migration_params
                )
                st.session_state.ai_insights = ai_insights

                # Step 5: Generate Risk Assessment (using fallback for now)
                st.session_state.risk_assessment = get_fallback_risk_assessment()

                st.success("✅ Comprehensive analysis completed!")
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    show_analysis_summary()

# Main function structure for Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="AWS Migration Planner")
    st.title("☁️ AWS Database Migration Planner")

    setup_initial_session_state()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "📊 Environment Configuration",
            "🚀 Analysis & Recommendations",
            "💰 Cost Refresh & Monitor",
            "📄 Reports & Export"
        ]
    )

    if page == "Home":
        st.markdown("Welcome to the AWS Database Migration Planner. Use this tool to:")
        st.markdown("- Configure your on-premises database environments.")
        st.markdown("- Get AWS instance recommendations and migration cost estimates.")
        st.markdown("- Analyze growth projections and risks.")
        st.markdown("- Generate comprehensive reports for stakeholders.")
        st.image("https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80", caption="Cloud Migration")

    elif page == "📊 Environment Configuration":
        st.markdown("## 📊 Environment Configuration")
        st.info("Define your existing on-premises database environments here. You can use vROps metrics, bulk upload, or manual entry.")
        show_enhanced_environment_setup_with_vrops()

    elif page == "🚀 Analysis & Recommendations":
        run_analysis_and_recommendations()
        st.markdown("---")
        st.markdown("### Detailed Insights & Visualizations")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Environment Analysis", "Risk Assessment", "Visualizations", "AI Insights", "Timeline Analysis"
        ])
        with tab1:
            show_environment_analysis_tab()
        with tab2:
            show_risk_assessment_tab()
        with tab3:
            show_visualizations_tab()
        with tab4:
            show_ai_insights_tab()
        with tab5:
            show_timeline_analysis_tab()

    elif page == "💰 Cost Refresh & Monitor":
        main_cost_refresh_section()
        auto_refresh_costs() # Add auto-refresh option

    elif page == "📄 Reports & Export":
        show_reports_section()

def show_reports_section():
    """Show reports and export section - ROBUST VERSION (consolidated)"""
    st.markdown("## 📄 Reports & Export")

    has_regular_results = st.session_state.get('analysis_results') is not None
    has_enhanced_results = st.session_state.get('enhanced_analysis_results') is not None

    if not has_regular_results and not has_enhanced_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        st.info("👆 Go to 'Analysis & Recommendations' section and click '🚀 Run Comprehensive Analysis'")
        return

    st.markdown("### 📊 Current Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        config_status = "✅ Complete" if st.session_state.get('migration_params') else "❌ Missing"
        st.metric("Migration Config", config_status)
    with col2:
        env_status = "✅ Complete" if st.session_state.get('environment_specs') else "❌ Missing"
        st.metric("Environment Setup", env_status)
    with col3:
        analysis_status = "✅ Complete" if (has_regular_results or has_enhanced_results) else "❌ Pending"
        st.metric("Analysis Results", analysis_status)

    results_to_use = None
    recommendations_to_use = None
    if has_enhanced_results:
        results_to_use = st.session_state.enhanced_analysis_results
        recommendations_to_use = st.session_state.enhanced_recommendations
        st.info("📊 Using Enhanced Analysis Results for reports.")
    elif has_regular_results:
        results_to_use = st.session_state.analysis_results
        recommendations_to_use = st.session_state.get('recommendations', {})
        st.info("📊 Using Standard Analysis Results for reports.")

    if results_to_use is None:
        st.error("Could not find analysis results to generate reports.")
        return

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
                    pdf_buffer = generate_executive_summary_pdf_robust(results_to_use, st.session_state.migration_params)
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
                    pdf_buffer = generate_technical_report_pdf_robust(results_to_use, recommendations_to_use, st.session_state.migration_params)
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
                csv_data = prepare_csv_export_data(results_to_use, recommendations_to_use)
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

    st.markdown("---")
    st.markdown("### 📦 Bulk Download")
    if st.button("📊 Generate All Reports", key="bulk_reports", use_container_width=True):
        with st.spinner("Generating all reports... This may take a moment..."):
            try:
                zip_buffer = create_bulk_reports_zip(results_to_use, recommendations_to_use, st.session_state.migration_params)
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

if __name__ == "__main__":
    main()