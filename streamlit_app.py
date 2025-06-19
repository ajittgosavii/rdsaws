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

def refresh_cost_calculations():
    """
    Main function to refresh all cost calculations and update the dollar values:
    - Monthly AWS Cost
    - Annual AWS Cost
    - Migration Cost
    - 3-Year Growth
    """

    try:
        # Check if required data exists
        if not st.session_state.migration_params:
            st.error("❌ Migration parameters required. Please configure migration settings first.")
            return False

        if not st.session_state.environment_specs:
            st.error("❌ Environment specifications required. Please configure environments first.")
            return False

        with st.spinner("🔄 Refreshing cost calculations..."):

            # Step 1: Refresh basic cost analysis
            monthly_cost, annual_cost, migration_cost = refresh_basic_costs()

            # Step 2: Refresh growth analysis for 3-year projections
            growth_percentage = refresh_growth_analysis(monthly_cost, annual_cost)

            # Step 3: Update session state with new values
            update_cost_session_state(monthly_cost, annual_cost, migration_cost, growth_percentage)

            # Step 4: Display refreshed values
            display_refreshed_metrics(monthly_cost, annual_cost, migration_cost, growth_percentage)

            st.success("✅ Cost calculations refreshed successfully!")
            return True

    except Exception as e:
        st.error(f"❌ Error refreshing costs: {str(e)}")
        return False

def refresh_basic_costs():
    """Refresh basic AWS and migration cost calculations"""

    # Initialize analyzer
    analyzer = MigrationAnalyzer(st.session_state.migration_params.get('anthropic_api_key'))

    # Recalculate instance recommendations
    recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
    st.session_state.recommendations = recommendations

    # Recalculate costs
    cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
    st.session_state.analysis_results = cost_analysis

    # Extract key values
    monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
    annual_cost = cost_analysis.get('annual_aws_cost', monthly_cost * 12)
    migration_cost = cost_analysis.get('migration_costs', {}).get('total', 0)

    return monthly_cost, annual_cost, migration_cost

def refresh_growth_analysis(monthly_cost: float, annual_cost: float):
    """Refresh 3-year growth analysis and calculate growth percentage"""

    try:
        # Initialize growth analyzer
        growth_analyzer = GrowthAwareCostAnalyzer()

        # Calculate 3-year growth projection
        growth_analysis = growth_analyzer.calculate_3_year_growth_projection(
            st.session_state.analysis_results,
            st.session_state.migration_params
        )
        st.session_state.growth_analysis = growth_analysis

        # Extract 3-year growth percentage
        growth_percentage = growth_analysis['growth_summary']['total_3_year_growth_percent']

        return growth_percentage

    except Exception as e:
        st.warning(f"Growth analysis failed, using default: {str(e)}")
        # Fallback calculation based on migration parameters
        annual_growth_rate = st.session_state.migration_params.get('annual_data_growth', 15)
        growth_percentage = ((1 + annual_growth_rate/100) ** 3 - 1) * 100
        return growth_percentage

def update_cost_session_state(monthly_cost: float, annual_cost: float,
                             migration_cost: float, growth_percentage: float):
    """Update session state with refreshed cost values"""

    # Update main analysis results
    if not st.session_state.analysis_results:
        st.session_state.analysis_results = {}

    st.session_state.analysis_results.update({
        'monthly_aws_cost': monthly_cost,
        'annual_aws_cost': annual_cost,
        'migration_costs': {
            'total': migration_cost,
            'last_updated': datetime.now().isoformat()
        }
    })

    # Update growth analysis summary
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

    # Create columns for metrics display
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

    if environment_name not in st.session_state.environment_specs:
        st.error(f"Environment '{environment_name}' not found")
        return

    # Get current environment specs
    env_specs = {environment_name: st.session_state.environment_specs[environment_name]}

    # Recalculate for this environment only
    analyzer = MigrationAnalyzer()

    recommendations = analyzer.calculate_instance_recommendations(env_specs)
    cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)

    # Update environment-specific costs
    if st.session_state.analysis_results:
        st.session_state.analysis_results['environment_costs'][environment_name] = \
            cost_analysis['environment_costs'][environment_name]

    st.success(f"✅ Refreshed costs for {environment_name}")

def auto_refresh_costs():
    """Automatic cost refresh with real-time pricing"""

    st.markdown("### 🔄 Auto-Refresh Cost Analysis")

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Enable Auto-Refresh (every 30 seconds)", value=False)

    if auto_refresh:
        # Use Streamlit's auto-refresh capability
        import time

        placeholder = st.empty()

        while auto_refresh:
            with placeholder.container():
                refresh_cost_calculations()
                st.write(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            time.sleep(30)  # Refresh every 30 seconds

def export_refreshed_costs():
    """Export the refreshed cost data to CSV"""

    if not st.session_state.analysis_results:
        st.warning("No cost data available to export")
        return

    # Prepare export data
    export_data = {
        'Metric': ['Monthly AWS Cost', 'Annual AWS Cost', 'Migration Cost', '3-Year Growth'],
        'Value': [
            f"${st.session_state.analysis_results.get('monthly_aws_cost', 0):,.0f}",
            f"${st.session_state.analysis_results.get('annual_aws_cost', 0):,.0f}",
            f"${st.session_state.analysis_results.get('migration_costs', {}).get('total', 0):,.0f}",
            f"{st.session_state.growth_analysis.get('growth_summary', {}).get('total_3_year_growth_percent', 0):.1f}%"
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

# Integration function for the main Streamlit app
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
            if st.session_state.analysis_results:
                monthly_cost = st.session_state.analysis_results.get('monthly_aws_cost', 0)
                annual_cost = st.session_state.analysis_results.get('annual_aws_cost', 0)
                growth_percentage = refresh_growth_analysis(monthly_cost, annual_cost)
                st.success(f"✅ Growth updated: {growth_percentage:.1f}%")
            else:
                st.warning("Please run full analysis first")

    with col3:
        if st.button("📥 Export Costs", use_container_width=True):
            export_refreshed_costs()

    # Environment-specific refresh
    if st.session_state.environment_specs:
        st.markdown("#### 🏢 Environment-Specific Refresh")

        selected_env = st.selectbox(
            "Select Environment to Refresh",
            list(st.session_state.environment_specs.keys())
        )

        if st.button(f"🔄 Refresh {selected_env}", use_container_width=True):
            refresh_specific_environment_costs(selected_env)

# Usage example for the main application
def main_cost_refresh_section():
    """Main section to be added to your Streamlit app"""

    st.markdown("## 💰 Cost Analysis & Refresh")

    # Check if analysis has been run
    if not st.session_state.analysis_results:
        st.info("👆 Please run the migration analysis first to see cost data")
        return

    # Display current values
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

    # Add refresh controls
    integrate_cost_refresh_ui()

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

    def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate REAL Claude AI insights synchronously"""

        if not self.anthropic_api_key:
            return {'error': 'No Anthropic API key provided', 'source': 'Error'}

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
            return {'error': 'Run: pip install anthropic', 'source': 'Library Error', 'success': False}
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

    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate the monthly cost for a single environment, including storage and IOPS."""
        
        # Get pricing for the recommended instance
        pricing = self.pricing_api.get_rds_pricing(
            region,
            target_engine,
            rec['instance_class'],
            rec['multi_az']
        )
        
        instance_cost_hourly = pricing['hourly']
        storage_cost_gb_monthly = pricing['storage_gb']
        iops_cost_gb_monthly = pricing['iops_gb']
        
        # Calculate instance cost
        instance_monthly_cost = instance_cost_hourly * rec['daily_usage_hours'] * 30.44  # Average days in a month
        
        # Calculate storage cost
        storage_monthly_cost = rec['storage_gb'] * storage_cost_gb_monthly
        
        # Calculate IOPS cost (assuming provisioned IOPS based on storage for simplicity, or add a specific IOPS field)
        # For a more accurate model, you'd need actual IOPS requirements from `specs`
        iops_monthly_cost = rec['storage_gb'] * iops_cost_gb_monthly # Placeholder, refine with actual IOPS if available
        
        total_monthly = instance_monthly_cost + storage_monthly_cost + iops_monthly_cost
        
        return {
            'instance_monthly_cost': instance_monthly_cost,
            'storage_monthly_cost': storage_monthly_cost,
            'iops_monthly_cost': iops_monthly_cost,
            'total_monthly': total_monthly
        }

# FIXED: Streamlit-compatible analysis function
def run_streamlit_migration_analysis():
    """Run migration analysis synchronously for Streamlit"""
    try:
        # Check if this is enhanced environment data
        is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
        if is_enhanced:
            st.write("Running enhanced cluster analysis...")
            # Use EnhancedMigrationAnalyzer for enhanced data
            analyzer = EnhancedMigrationAnalyzer()
            recommendations = analyzer.calculate_enhanced_instance_recommendations(st.session_state.environment_specs)
            st.session_state.enhanced_recommendations = recommendations # Store enhanced recommendations
            cost_analysis = analyzer.calculate_enhanced_migration_costs(recommendations, st.session_state.migration_params)
            st.session_state.enhanced_analysis_results = cost_analysis # Store enhanced results
            st.session_state.analysis_results = cost_analysis # Also update general results for other tabs
        else:
            st.write("Running standard migration analysis...")
            # Use standard MigrationAnalyzer
            analyzer = MigrationAnalyzer(st.session_state.migration_params.get('anthropic_api_key'))
            recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
            st.session_state.recommendations = recommendations
            cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
            st.session_state.analysis_results = cost_analysis

        # Step 3: Generate risk assessment using appropriate data
        st.write("⚠️ Assessing risks...")
        if is_enhanced:
            risk_assessment = calculate_migration_risks_enhanced(st.session_state.migration_params, recommendations)
        else:
            risk_assessment = calculate_migration_risks(st.session_state.migration_params, recommendations)
        st.session_state.risk_assessment = risk_assessment

        # Step 4: Generate cost comparison
        st.write("📈 Generating cost comparisons...")
        generate_enhanced_cost_visualizations() # This function should handle both enhanced/standard data

        st.success("✅ Analysis complete!")
        # Show summary
        show_analysis_summary() # This function should also handle both enhanced/standard data

        # Provide navigation hint
        st.info("📈 View detailed results in the 'Results Dashboard' section")

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.code(str(e))
        # Provide troubleshooting info
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("If the error persists:")
        st.markdown("1. Check that all environment fields are properly filled")
        st.markdown("2. Verify that numerical values are within valid ranges")
        st.markdown("3. Try using the 'Simple Configuration' option instead")


def is_enhanced_environment_data(environment_specs: Dict) -> bool:
    """Check if the environment data contains enhanced (cluster-specific) fields."""
    if not environment_specs:
        return False
    # Check the first environment for enhanced fields
    first_env_spec = next(iter(environment_specs.values()), {})
    return 'writer' in first_env_spec or 'readers' in first_env_spec


# ===========================
# ENHANCED STREAMLIT INTERFACE
# ===========================
def show_enhanced_environment_setup_with_vrops():
    """Enhanced environment setup with vROps integration"""
    st.markdown("## 📊 Enhanced Environment Configuration")
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

def show_vrops_import_interface(analyzer: Any): # VRopsMetricsAnalyzer type hint
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
        "Upload vROps Export File", type=['csv', 'xlsx'],
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

def process_vrops_data(df: pd.DataFrame, analyzer: Any) -> Dict: # VRopsMetricsAnalyzer type hint
    """Process uploaded vROps data into environment specifications"""
    st.markdown("##### 🔗 Column Mapping")
    # Get required metrics
    required_metrics = analyzer.required_metrics # Access directly from instance

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

    with col2:
        mappings['avg_memory_usage_percent'] = st.selectbox("Avg Memory Usage %", available_columns, key="avg_mem_col")
        mappings['memory_allocated_gb'] = st.selectbox("Memory Allocated GB", available_columns, key="mem_alloc_col")
        mappings['memory_balloon_gb'] = st.selectbox("Memory Balloon GB", available_columns, key="mem_balloon_col")
        mappings['memory_swapped_gb'] = st.selectbox("Memory Swapped GB", available_columns, key="mem_swap_col")

        st.markdown("**Disk/Storage Metrics**")
        mappings['max_iops_total'] = st.selectbox("Max IOPS Total", available_columns, key="max_iops_col")
        mappings['avg_iops_total'] = st.selectbox("Avg IOPS Total", available_columns, key="avg_iops_col")
        mappings['max_disk_latency_ms'] = st.selectbox("Max Disk Latency ms", available_columns, key="max_latency_col")
        mappings['storage_allocated_gb'] = st.selectbox("Storage Allocated GB", available_columns, key="storage_alloc_col")
        mappings['storage_used_gb'] = st.selectbox("Storage Used GB", available_columns, key="storage_used_col")

    # Add a button to process
    if st.button("🔄 Process vROps Data", type="primary", use_container_width=True):
        if not all(mappings.values()):
            st.error("Please map all required vROps metrics to proceed.")
            return {}
        try:
            processed_data = analyzer.process_vrops_dataframe(df, mappings) # Call instance method
            return processed_data
        except Exception as e:
            st.error(f"Error processing data with current mapping: {str(e)}")
            return {}
    return {}

def show_vrops_processing_summary(processed_environments: Dict, analyzer: Any): # VRopsMetricsAnalyzer type hint
    """Show a summary of processed vROps environments."""
    st.markdown("#### ✅ Processed Environments Summary")
    summary_data = []
    for vm_name, specs in processed_environments.items():
        summary_data.append({
            "VM Name": vm_name,
            "Environment Type": specs.get('environment_type', 'N/A'),
            "CPU Cores": specs.get('cpu_cores', 'N/A'),
            "RAM GB": specs.get('ram_gb', 'N/A'),
            "Storage GB": specs.get('storage_gb', 'N/A'),
            "Avg IOPS": specs.get('avg_iops', 'N/A'),
            "Peak Connections": specs.get('peak_connections', 'N/A')
        })
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)

    # Optional: Display more detailed insights from the analyzer
    if hasattr(analyzer, 'vrops_insights') and analyzer.vrops_insights:
        st.markdown("##### 💡 vROps Insights")
        for vm, insights in analyzer.vrops_insights.items():
            with st.expander(f"Insights for {vm}"):
                for insight in insights:
                    st.write(f"- {insight}")

class VRopsMetricsAnalyzer:
    """Analyzes vROps performance metrics to derive environment specifications."""
    def __init__(self):
        self.required_metrics = [
            'vm_name', 'environment', 'max_cpu_usage_percent', 'avg_cpu_usage_percent',
            'cpu_cores_allocated', 'max_memory_usage_percent', 'avg_memory_usage_percent',
            'memory_allocated_gb', 'max_iops_total', 'avg_iops_total',
            'max_disk_latency_ms', 'storage_allocated_gb', 'storage_used_gb',
            'max_network_throughput_mbps', 'database_connections_max', 'database_size_gb'
        ]
        self.vrops_insights = {}

    def process_vrops_dataframe(self, df: pd.DataFrame, mappings: Dict) -> Dict:
        """Processes the vROps DataFrame based on user mappings to extract environment specs."""
        processed_environments = {}
        self.vrops_insights = {} # Reset insights for new processing

        # Validate essential mappings
        for key in ['vm_name', 'cpu_cores_allocated', 'memory_allocated_gb', 'storage_allocated_gb']:
            if not mappings.get(key) or mappings[key] not in df.columns:
                raise ValueError(f"Mapped column '{key}' not found in the uploaded data. Please check mappings.")

        for index, row in df.iterrows():
            try:
                vm_name = row[mappings['vm_name']]
                environment_type = row.get(mappings.get('environment', 'Environment'), 'Production') # Default to Production if not mapped
                cpu_cores = row[mappings['cpu_cores_allocated']]
                ram_gb = row[mappings['memory_allocated_gb']]
                storage_gb = row[mappings['storage_allocated_gb']]

                # Utilize performance metrics for more dynamic sizing or recommendations
                max_cpu_usage = row.get(mappings.get('max_cpu_usage_percent'), 0)
                avg_cpu_usage = row.get(mappings.get('avg_cpu_usage_percent'), 0)
                max_mem_usage = row.get(mappings.get('max_memory_usage_percent'), 0)
                avg_mem_usage = row.get(mappings.get('avg_memory_usage_percent'), 0)
                max_iops = row.get(mappings.get('max_iops_total'), 0)
                avg_iops = row.get(mappings.get('avg_iops_total'), 0)
                max_latency = row.get(mappings.get('max_disk_latency_ms'), 0)
                network_throughput = row.get(mappings.get('max_network_throughput_mbps'), 0)
                peak_connections = row.get(mappings.get('database_connections_max'), 0)
                db_size = row.get(mappings.get('database_size_gb'), storage_gb) # Use allocated if specific DB size not given
                mem_balloon = row.get(mappings.get('memory_balloon_gb'), 0)
                mem_swapped = row.get(mappings.get('memory_swapped_gb'), 0)

                # Basic validation and type conversion
                cpu_cores = int(cpu_cores)
                ram_gb = int(ram_gb)
                storage_gb = int(storage_gb)

                # Derive workload pattern (simple example)
                workload_pattern = 'mixed'
                if avg_iops > 1000 and network_throughput < 50:
                    workload_pattern = 'io_heavy'
                elif network_throughput > 100 and avg_iops < 500:
                    workload_pattern = 'network_heavy'

                # Derive read/write ratio (placeholder, ideally from vROps or estimation)
                read_write_ratio = 50 # Default to 50/50

                # Determine multi_az based on environment type or performance
                multi_az = environment_type.lower() == 'production' or (max_cpu_usage > 70 and max_mem_usage > 70)

                # Collect insights
                vm_insights = []
                if max_cpu_usage > 80:
                    vm_insights.append(f"High peak CPU usage ({max_cpu_usage:.1f}%). Consider larger instance or scaling group.")
                if max_mem_usage > 80:
                    vm_insights.append(f"High peak memory usage ({max_mem_usage:.1f}%). Consider larger instance.")
                if mem_balloon > 0 or mem_swapped > 0:
                    vm_insights.append(f"Memory ballooning ({mem_balloon} GB) or swapping ({mem_swapped} GB) detected. Indicates potential memory contention.")
                if max_latency > 20:
                    vm_insights.append(f"High disk latency ({max_latency:.1f} ms). Suggests potential storage bottleneck.")
                if avg_iops == 0 and max_iops == 0 and db_size > 0:
                    vm_insights.append("No IOPS data found. Unable to accurately assess storage performance.")

                self.vrops_insights[vm_name] = vm_insights

                processed_environments[vm_name] = {
                    'environment_type': environment_type.lower(),
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'iops_requirement': max_iops, # Use max IOPS as requirement
                    'peak_connections': peak_connections,
                    'workload_pattern': workload_pattern,
                    'read_write_ratio': read_write_ratio,
                    'multi_az': multi_az,
                    'avg_cpu_usage_percent': avg_cpu_usage,
                    'avg_memory_usage_percent': avg_mem_usage,
                    'avg_iops': avg_iops,
                    'database_size_gb': db_size,
                    'network_throughput_mbps': network_throughput # Added for completeness
                }
            except KeyError as ke:
                st.warning(f"Skipping row {index} due to missing mapped column: {ke}. Please check your mappings and data.")
            except ValueError as ve:
                st.warning(f"Skipping row {index} due to data conversion error: {ve}. Ensure numerical fields contain valid numbers.")
            except Exception as e:
                st.error(f"An unexpected error occurred while processing row {index} for VM {row.get(mappings.get('vm_name', 'N/A'))}: {e}")

        return processed_environments

def show_manual_detailed_entry(analyzer: Any): # VRopsMetricsAnalyzer type hint
    """Show manual detailed entry for environment configuration"""
    st.markdown("### 📝 Manual Detailed Environment Entry")
    st.warning("This section is under development. Please use Simple Configuration or CSV upload for now.")

def show_enhanced_bulk_upload(analyzer: Any): # VRopsMetricsAnalyzer type hint
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
        "Upload Cluster Configuration", type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with your detailed cluster configurations."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")

            # Process the DataFrame into enhanced environment specs
            processed_environments = process_enhanced_cluster_data(df)
            if processed_environments:
                st.session_state.environment_specs = processed_environments
                st.success(f"✅ Successfully processed {len(processed_environments)} cluster environments!")
                show_cluster_configuration_preview(processed_environments)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.code(str(e))

def process_enhanced_cluster_data(df: pd.DataFrame) -> Dict:
    """Processes DataFrame for enhanced cluster configurations."""
    # This is a placeholder. Real implementation would parse the DF into the nested structure
    # expected by EnhancedMigrationAnalyzer.calculate_enhanced_instance_recommendations.
    # For now, it assumes a flat structure that matches basic environment_specs.
    st.warning("Processing of enhanced cluster data from CSV is a placeholder. Ensure your CSV columns match expected fields for writer/reader/storage if manual processing is needed.")
    processed = {}
    for idx, row in df.iterrows():
        env_name = row.get('EnvironmentName', f'Environment_{idx+1}')
        processed[env_name] = {
            'cpu_cores': row.get('Writer_CPU_Cores', 0),
            'ram_gb': row.get('Writer_RAM_GB', 0),
            'storage_gb': row.get('Storage_Size_GB', 0),
            'environment_type': row.get('EnvironmentType', 'production'),
            'iops_requirement': row.get('Storage_IOPS', 0),
            'peak_connections': row.get('Connections_Peak', 0),
            # These fields are crucial for enhanced analysis
            'writer': {
                'instance_class': row.get('Writer_Instance_Class', 'db.r5.large'),
                'multi_az': row.get('Writer_MultiAZ', True),
                'cpu_cores': row.get('Writer_CPU_Cores', 0),
                'ram_gb': row.get('Writer_RAM_GB', 0),
            },
            'readers': {
                'count': row.get('Reader_Count', 0),
                'instance_class': row.get('Reader_Instance_Class', 'db.r5.large'),
                'multi_az': row.get('Reader_MultiAZ', False),
            },
            'storage': {
                'size_gb': row.get('Storage_Size_GB', 0),
                'type': row.get('Storage_Type', 'gp3'),
                'iops': row.get('Storage_IOPS', 0),
                'encrypted': row.get('Storage_Encrypted', True),
                'backup_retention_days': row.get('Backup_Retention_Days', 7)
            },
            'workload_pattern': row.get('Workload_Pattern', 'mixed'),
            'read_write_ratio': row.get('Read_Write_Ratio', 50),
            'connections': row.get('Connections_Peak', 0),
        }
    return processed

def create_enhanced_cluster_template() -> pd.DataFrame:
    """Create a sample DataFrame for enhanced cluster configuration template."""
    data = {
        'EnvironmentName': ['ProdCluster1', 'DevCluster1'],
        'EnvironmentType': ['production', 'development'],
        'Writer_CPU_Cores': [16, 4],
        'Writer_RAM_GB': [64, 16],
        'Writer_Instance_Class': ['db.r5.4xlarge', 'db.t3.large'],
        'Writer_MultiAZ': [True, False],
        'Reader_Count': [3, 0],
        'Reader_Instance_Class': ['db.r5.large', 'N/A'],
        'Reader_MultiAZ': [False, False],
        'Storage_Size_GB': [2000, 500],
        'Storage_Type': ['gp3', 'gp3'],
        'Storage_IOPS': [6000, 1500],
        'Storage_Encrypted': [True, True],
        'Backup_Retention_Days': [30, 7],
        'Workload_Pattern': ['oltp', 'development'],
        'Read_Write_Ratio': [70, 30],
        'Connections_Peak': [1000, 150]
    }
    return pd.DataFrame(data)

def show_cluster_configuration_preview(recommendations: Dict):
    """Show a preview of the processed cluster configurations."""
    st.markdown("#### 📊 Cluster Configuration Preview")
    preview_data = []
    for env_name, rec in recommendations.items():
        preview_data.append({
            "Environment": env_name,
            "Type": rec['environment_type'].title(),
            "Writer Instance": f"{rec['writer']['instance_class']} (Multi-AZ: {rec['writer']['multi_az']})",
            "Readers": f"{rec['readers']['count']} x {rec['readers']['instance_class']}",
            "Storage": f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()} ({rec['storage']['iops']} IOPS)",
            "Workload": f"{rec['workload_pattern']} ({rec['read_write_ratio']}% Read)"
        })
    st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

class EnhancedMigrationAnalyzer(MigrationAnalyzer):
    """Enhanced migration analyzer with support for Aurora clusters (Writer/Reader)"""
    
    def calculate_enhanced_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations for Aurora environments with writer/reader."""
        
        enhanced_recommendations = {}
        for env_name, specs in environment_specs.items():
            # Assume specs already contain 'writer', 'readers', 'storage' breakdown
            # This method acts more as a pass-through and validation/enrichment
            # if the input `environment_specs` are already in the enhanced format.
            # If not, you'd add logic to infer them from basic specs.

            # For now, let's assume `specs` is already in the enhanced format from
            # `process_enhanced_cluster_data` or manual entry.
            
            # Simple validation/defaulting for critical nested keys
            writer_specs = specs.get('writer', {})
            readers_specs = specs.get('readers', {})
            storage_specs = specs.get('storage', {})

            # Categorize environment
            environment_type = self._categorize_environment(env_name)

            # Assign default values if not provided for instance class, or infer based on CPU/RAM
            writer_instance_class = writer_specs.get('instance_class') or \
                                    self._calculate_instance_class(writer_specs.get('cpu_cores', 0), 
                                                                   writer_specs.get('ram_gb', 0), 
                                                                   environment_type)
            reader_instance_class = readers_specs.get('instance_class') or \
                                    self._calculate_instance_class(readers_specs.get('cpu_cores', 0), # Readers might have different specs
                                                                   readers_specs.get('ram_gb', 0), 
                                                                   'read_replica') # Special env type for readers

            enhanced_recommendations[env_name] = {
                'environment_type': environment_type,
                'writer': {
                    'instance_class': writer_instance_class,
                    'multi_az': writer_specs.get('multi_az', True),
                    'cpu_cores': writer_specs.get('cpu_cores', 0),
                    'ram_gb': writer_specs.get('ram_gb', 0),
                },
                'readers': {
                    'count': readers_specs.get('count', 0),
                    'instance_class': reader_instance_class,
                    'multi_az': readers_specs.get('multi_az', False),
                },
                'storage': {
                    'size_gb': storage_specs.get('size_gb', 0),
                    'type': storage_specs.get('type', 'gp3'),
                    'iops': storage_specs.get('iops', 0),
                    'encrypted': storage_specs.get('encrypted', True),
                    'backup_retention_days': storage_specs.get('backup_retention_days', 7)
                },
                'workload_pattern': specs.get('workload_pattern', 'mixed'),
                'read_write_ratio': specs.get('read_write_ratio', 50),
                'connections': specs.get('peak_connections', 100), # Using connections for consistency
            }
        return enhanced_recommendations
        
    def calculate_enhanced_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs for enhanced (Aurora cluster) recommendations."""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'aurora-postgresql') # Aurora default
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            env_costs = self._calculate_enhanced_environment_cost(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
            
        # Inherit common migration costs from parent (DMS, data transfer, professional services)
        # Call the parent's method to calculate these common parts
        parent_migration_costs_data = super().calculate_migration_costs(
            self._flatten_recommendations_for_base_cost(recommendations), # Convert enhanced to basic for this call
            migration_params
        )
        
        migration_costs = parent_migration_costs_data['migration_costs']
        transfer_costs = parent_migration_costs_data['transfer_costs']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }

    def _flatten_recommendations_for_base_cost(self, enhanced_recommendations: Dict) -> Dict:
        """Flattens enhanced recommendations to a basic format for common cost calculations."""
        flattened = {}
        for env_name, rec in enhanced_recommendations.items():
            # Create a simplified representation for the base MigrationAnalyzer
            # This assumes that the primary instance (writer) will drive the base cost
            # and readers will be factored in by the enhanced cost calculation.
            flattened[env_name] = {
                'environment_type': rec['environment_type'],
                'instance_class': rec['writer']['instance_class'],
                'cpu_cores': rec['writer']['cpu_cores'],
                'ram_gb': rec['writer']['ram_gb'],
                'storage_gb': rec['storage']['size_gb'], # Use writer's primary storage
                'multi_az': rec['writer']['multi_az'],
                'daily_usage_hours': 24, # Assume 24/7 for production clusters
                'peak_connections': rec['connections']
            }
        return flattened

    def _calculate_enhanced_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate the monthly cost for an enhanced (Aurora cluster) environment, including writer, readers, storage, and I/O."""
        
        writer_pricing = self.pricing_api.get_rds_pricing(
            region,
            target_engine,
            rec['writer']['instance_class'],
            rec['writer']['multi_az']
        )
        
        # Writer cost
        writer_monthly_cost = writer_pricing['hourly'] * 24 * 30.44
        
        # Readers cost
        readers_monthly_cost = 0
        if rec['readers']['count'] > 0:
            reader_pricing = self.pricing_api.get_rds_pricing(
                region,
                target_engine,
                rec['readers']['instance_class'],
                rec['readers']['multi_az'] # Multi-AZ for readers usually false
            )
            readers_monthly_cost = rec['readers']['count'] * reader_pricing['hourly'] * 24 * 30.44
            
        # Storage cost (Aurora storage is consumption-based, often includes 3 copies)
        # Simplistic: direct Gb cost. Real Aurora pricing is tiered and includes I/O.
        storage_cost_gb_monthly = writer_pricing['storage_gb'] # Using writer's storage cost metric
        storage_monthly_cost = rec['storage']['size_gb'] * storage_cost_gb_monthly
        
        # I/O cost (Aurora I/O requests)
        # This is an estimation. Real I/O pricing depends on millions of requests.
        # Assuming a base level of I/O operations proportional to storage size or workload.
        # For simplicity, using a factor based on average IOPS from vROps or a default.
        estimated_iops_per_month = rec['storage']['iops'] * 30.44 * 24 * 60 * 60 / 1000000 # Convert IOPS to millions per month
        io_request_cost_per_million = writer_pricing['io_request'] # Price per million I/O requests
        io_monthly_cost = estimated_iops_per_month * io_request_cost_per_million
        
        total_monthly = writer_monthly_cost + readers_monthly_cost + storage_monthly_cost + io_monthly_cost
        
        return {
            'writer_monthly_cost': writer_monthly_cost,
            'readers_monthly_cost': readers_monthly_cost,
            'storage_monthly_cost': storage_monthly_cost,
            'io_monthly_cost': io_monthly_cost,
            'total_monthly': total_monthly
        }

def show_simple_configuration():
    """Show simple environment configuration interface (legacy)"""
    st.markdown("### 🔄 Simple Environment Configuration (Legacy)")
    st.info("This is a simplified configuration option. For detailed setup, use 'Manual Detailed Entry' or 'Bulk CSV Upload'.")

    # Initialize session state for environments if not present
    if 'environment_specs' not in st.session_state:
        st.session_state.environment_specs = {}

    environment_name = st.text_input("Environment Name", "Production", key="simple_env_name")
    cpu_cores = st.number_input("CPU Cores", min_value=1, value=4, key="simple_cpu")
    ram_gb = st.number_input("RAM (GB)", min_value=1, value=16, key="simple_ram")
    storage_gb = st.number_input("Storage (GB)", min_value=100, value=500, step=100, key="simple_storage")
    iops_requirement = st.number_input("IOPS Requirement", min_value=0, value=3000, step=500, key="simple_iops")
    peak_connections = st.number_input("Peak Database Connections", min_value=1, value=500, step=50, key="simple_connections")

    # Add environment button
    if st.button("➕ Add/Update Environment", key="add_simple_env", type="primary", use_container_width=True):
        st.session_state.environment_specs[environment_name] = {
            'cpu_cores': cpu_cores,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'iops_requirement': iops_requirement,
            'peak_connections': peak_connections
        }
        st.success(f"Environment '{environment_name}' added/updated!")
    
    # Display current environments
    if st.session_state.environment_specs:
        st.markdown("#### 📊 Configured Environments")
        env_df = pd.DataFrame.from_dict(st.session_state.environment_specs, orient='index')
        st.dataframe(env_df, use_container_width=True)


# ===========================
# RISK ASSESSMENT MODULE
# ===========================

def calculate_migration_risks(migration_params: Dict, recommendations: Dict) -> Dict:
    """Calculate migration risks based on parameters and recommendations."""
    
    risks = []
    overall_risk_score = 0
    
    # Risk 1: Data Size vs. Timeline
    data_size_gb = migration_params.get('data_size_gb', 0)
    migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 0)
    
    if data_size_gb > 1000 and migration_timeline_weeks < 8:
        risks.append("High volume of data with short timeline. Risk of data transfer bottlenecks and incomplete migration. **Mitigation**: Extend timeline, use AWS Direct Connect/Snowball, optimize DMS tasks.")
        overall_risk_score += 3
    elif data_size_gb > 500 and migration_timeline_weeks < 4:
        risks.append("Moderate data with very short timeline. Consider data transfer optimizations. **Mitigation**: Use DMS, assess network bandwidth.")
        overall_risk_score += 2
        
    # Risk 2: Source/Target Engine Compatibility
    source_engine = migration_params.get('source_engine', '').lower()
    target_engine = migration_params.get('target_engine', '').lower()
    
    if source_engine != target_engine:
        if source_engine == 'oracle' and 'postgres' in target_engine:
            risks.append("Heterogeneous migration (Oracle to PostgreSQL). High risk of schema/code conversion challenges, data type mismatches, and application re-platforming. **Mitigation**: Utilize AWS SCT, detailed code analysis, extensive testing, specialized migration tools.")
            overall_risk_score += 5
        elif source_engine in ['sqlserver', 'mysql'] and 'postgres' in target_engine:
            risks.append("Heterogeneous migration. Moderate risk for schema/code conversion. **Mitigation**: Use AWS SCT for conversion, thorough testing.")
            overall_risk_score += 3
        else:
            risks.append("Heterogeneous migration. Potential for compatibility issues. **Mitigation**: Thorough schema and code analysis.")
            overall_risk_score += 2
    else:
        risks.append("Homogeneous migration. Lower compatibility risk, focus on data integrity and downtime minimization. **Mitigation**: Use native tools, logical replication.")
        
    # Risk 3: Application Downtime Tolerance
    downtime_tolerance = migration_params.get('downtime_tolerance', 'high').lower()
    if downtime_tolerance == 'low':
        risks.append("Low downtime tolerance. Requires advanced migration strategies like DMS CDC, blue/green deployments, or minimal cutover windows. **Mitigation**: Implement CDC replication, detailed cutover plan, extensive dry runs.")
        overall_risk_score += 4
    elif downtime_tolerance == 'medium':
        risks.append("Medium downtime tolerance. Can use standard DMS or backup/restore with planned outage. **Mitigation**: Schedule during off-peak hours, clear communication plan.")
        overall_risk_score += 2
        
    # Risk 4: Complexity of Environment (e.g., number of environments, multi-region)
    num_environments = len(recommendations)
    if num_environments > 5:
        risks.append(f"Large number of environments ({num_environments}). Increased management overhead and coordination complexity. **Mitigation**: Phased migration approach, automation, dedicated migration team.")
        overall_risk_score += 3
    
    # Risk 5: Lack of Multi-AZ for Production (if applicable and not recommended)
    for env_name, rec in recommendations.items():
        if rec.get('environment_type') == 'production' and not rec.get('multi_az', True): # Default to True for safer check
            risks.append(f"Production environment '{env_name}' not configured with Multi-AZ. Risk of single point of failure and higher RTO/RPO. **Mitigation**: Enable Multi-AZ or implement read replicas/DR strategy.")
            overall_risk_score += 4
            break # Only add once for production environments
            
    # Risk 6: Lack of AI Key (if AI insights are a feature)
    if not migration_params.get('anthropic_api_key'):
        risks.append("Anthropic API Key not provided. AI-driven insights for risk mitigation and optimization will be unavailable. **Mitigation**: Provide API key for enhanced analysis.")
        overall_risk_score += 1
            
    # Determine overall risk level
    if overall_risk_score >= 10:
        overall_risk_level = "High"
    elif overall_risk_score >= 5:
        overall_risk_level = "Medium"
    else:
        overall_risk_level = "Low"
        
    return {
        'risks': risks,
        'overall_risk_score': overall_risk_score,
        'overall_risk_level': overall_risk_level
    }

def calculate_migration_risks_enhanced(migration_params: Dict, recommendations: Dict) -> Dict:
    """Calculate migration risks specifically for enhanced/cluster recommendations."""
    
    risks = []
    overall_risk_score = 0
    
    # Inherit base risks from standard function
    base_risks_data = calculate_migration_risks(migration_params, recommendations) # Pass enhanced recs, it will handle it
    risks.extend(base_risks_data['risks'])
    overall_risk_score += base_risks_data['overall_risk_score']

    # Additional risks for enhanced/cluster environments
    for env_name, rec in recommendations.items():
        # Risk: Reader count vs. Read workload
        if rec.get('workload_pattern') == 'read_heavy' and rec['readers']['count'] == 0:
            risks.append(f"Environment '{env_name}' is read-heavy but has no read replicas. Risk of performance bottlenecks on writer instance. **Mitigation**: Add read replicas to offload read traffic.")
            overall_risk_score += 2
        
        # Risk: Storage type vs. IOPS requirement (if specific IOPS is high but storage type is not optimized)
        if rec['storage']['iops'] > 5000 and rec['storage']['type'] in ['gp2', 'gp3'] and rec['environment_type'] == 'production':
            risks.append(f"Production environment '{env_name}' has high IOPS requirement ({rec['storage']['iops']}) but uses GP2/GP3 storage. Risk of IOPS throttling or higher cost than io2. **Mitigation**: Consider io2 storage for consistent high performance and predictable IOPS.")
            overall_risk_score += 3
        
        # Risk: Multi-AZ for Writer (critical for production)
        if rec['environment_type'] == 'production' and not rec['writer'].get('multi_az', False):
            risks.append(f"Production environment '{env_name}' writer is not Multi-AZ. High risk of downtime during failures. **Mitigation**: Enable Multi-AZ for writer instance.")
            overall_risk_score += 5

    # Re-evaluate overall risk level after adding enhanced risks
    if overall_risk_score >= 12: # Higher threshold for enhanced high risk
        overall_risk_level = "High"
    elif overall_risk_score >= 7: # Higher threshold for enhanced medium risk
        overall_risk_level = "Medium"
    else:
        overall_risk_level = "Low"

    return {
        'risks': list(set(risks)), # Use set to remove duplicates
        'overall_risk_score': overall_risk_score,
        'overall_risk_level': overall_risk_level
    }

def show_risk_assessment_tab():
    """Show the risk assessment tab with insights."""
    st.markdown("## ⚠️ Migration Risk Assessment")

    if 'risk_assessment' not in st.session_state or not st.session_state.risk_assessment:
        st.info("👆 Run the migration analysis first to generate risk assessment.")
        return

    risk_data = st.session_state.risk_assessment

    st.markdown(f"### Overall Risk Level: **{risk_data['overall_risk_level']}** (Score: {risk_data['overall_risk_score']})")

    if risk_data['overall_risk_level'] == "High":
        st.error("🚨 High-Risk Migration Detected! Careful planning and mitigation required.")
    elif risk_data['overall_risk_level'] == "Medium":
        st.warning("🔶 Medium-Risk Migration. Address key concerns for smoother transition.")
    else:
        st.success("✅ Low-Risk Migration. Proceed with confidence, but remain vigilant.")

    st.markdown("---")
    st.markdown("### Detailed Risks & Mitigations")

    if risk_data['risks']:
        for i, risk in enumerate(risk_data['risks']):
            st.markdown(f"**Risk {i+1}:** {risk}")
            st.markdown("---")
    else:
        st.info("No significant risks identified based on current parameters.")


# ===========================
# VISUALIZATIONS MODULE
# ===========================

def generate_enhanced_cost_visualizations():
    """Generate comprehensive cost visualizations (bar, pie, growth)."""
    st.markdown("### 📊 Cost Visualizations")

    if 'analysis_results' not in st.session_state:
        st.warning("No analysis results to visualize. Please run the analysis first.")
        return

    analysis_results = st.session_state.analysis_results
    migration_params = st.session_state.migration_params

    # 1. Monthly AWS Cost Breakdown (Bar Chart)
    if 'environment_costs' in analysis_results:
        env_costs_data = {env: details['total_monthly'] for env, details in analysis_results['environment_costs'].items()}
        env_df = pd.DataFrame(list(env_costs_data.items()), columns=['Environment', 'Monthly Cost'])
        fig_env = px.bar(env_df, x='Environment', y='Monthly Cost', title='Monthly AWS Cost by Environment',
                         hover_data={'Monthly Cost': ':.2f'},
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_env, use_container_width=True)

    # 2. Migration Cost Breakdown (Pie Chart)
    if 'migration_costs' in analysis_results and analysis_results['migration_costs']['total'] > 0:
        migration_breakdown = {k.replace('_', ' ').title(): v for k, v in analysis_results['migration_costs'].items() if k != 'total'}
        migration_df = pd.DataFrame(list(migration_breakdown.items()), columns=['Category', 'Cost'])
        fig_mig = px.pie(migration_df, values='Cost', names='Category', title='Migration Cost Breakdown',
                         hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_mig, use_container_width=True)

    # 3. 3-Year Growth Projection (Line Chart)
    if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
        growth_data = st.session_state.growth_analysis['yearly_projection']
        growth_df = pd.DataFrame(growth_data)
        fig_growth = px.line(growth_df, x='year', y='cost', title='3-Year Cost Projection with Growth',
                             markers=True, hover_data={'cost': ':.2f'})
        fig_growth.update_layout(xaxis_title="Year", yaxis_title="Annual Cost ($)")
        st.plotly_chart(fig_growth, use_container_width=True)

    # 4. CPU/RAM Utilization vs. Recommended (Bar Chart - requires detailed recommendations)
    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
        data = []
        for env_name, rec in st.session_state.recommendations.items():
            # For enhanced recommendations, you'd aggregate writer/reader CPU/RAM
            if 'writer' in rec: # Enhanced structure
                current_cpu = st.session_state.environment_specs[env_name]['cpu_cores']
                current_ram = st.session_state.environment_specs[env_name]['ram_gb']
                recommended_cpu = rec['writer']['cpu_cores'] + (rec['readers']['count'] * rec['readers']['cpu_cores'])
                recommended_ram = rec['writer']['ram_gb'] + (rec['readers']['count'] * rec['readers']['ram_gb'])
            else: # Standard structure
                current_cpu = st.session_state.environment_specs[env_name]['cpu_cores']
                current_ram = st.session_state.environment_specs[env_name]['ram_gb']
                recommended_cpu = rec['cpu_cores']
                recommended_ram = rec['ram_gb']

            data.append({'Environment': env_name, 'Type': 'Current CPU', 'Value': current_cpu})
            data.append({'Environment': env_name, 'Type': 'Recommended CPU', 'Value': recommended_cpu})
            data.append({'Environment': env_name, 'Type': 'Current RAM', 'Value': current_ram})
            data.append({'Environment': env_name, 'Type': 'Recommended RAM', 'Value': recommended_ram})

        df_res = pd.DataFrame(data)
        fig_res = px.bar(df_res, x='Environment', y='Value', color='Type', barmode='group',
                         title='Current vs. Recommended Resources (CPU Cores & RAM GB)',
                         facet_col='Type', facet_col_wrap=2,
                         color_discrete_map={'Current CPU': 'lightgray', 'Recommended CPU': 'darkblue',
                                             'Current RAM': 'lightgray', 'Recommended RAM': 'darkgreen'})
        fig_res.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
        st.plotly_chart(fig_res, use_container_width=True)

def create_migration_timeline_chart(migration_params: Dict):
    """Create a detailed migration timeline Gantt chart."""
    # This is a simplified bar chart instead of a full Gantt for less dependency
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    
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

    # Adjust durations to fit timeline_weeks proportionally if needed
    total_default_duration = sum(p['duration'] for p in phases)
    scale_factor = timeline_weeks / total_default_duration if total_default_duration > 0 else 1

    chart_data = []
    current_week = 0
    for phase in phases:
        duration_scaled = max(1, round(phase['duration'] * scale_factor)) # Ensure at least 1 week
        start_week = current_week
        end_week = current_week + duration_scaled
        chart_data.append(dict(Task=phase['name'], Start=start_week, Finish=end_week, Duration=duration_scaled))
        current_week = end_week

    df_gantt = pd.DataFrame(chart_data)

    fig = ff.create_gantt(df_gantt, colors=px.colors.qualitative.Pastel, index_col='Task',
                          show_colorbar=False, bar_width=0.4, showgrid_x=True, showgrid_y=True,
                          title='Migration Project Timeline')
    
    fig.update_xaxes(title_text="Weeks from Start", tickvals=list(range(0, int(current_week) + 2, 2)))
    fig.update_layout(autosize=True, height=450,
                      hovermode='closest')
    return fig

def show_visualizations_tab():
    """Show various visualizations for the migration analysis."""
    st.markdown("## 📈 Data Visualizations")
    
    if not st.session_state.get('analysis_results') and not st.session_state.get('enhanced_analysis_results'):
        st.info("👆 Please run the migration analysis first to generate visualizations.")
        return

    # Check which results are available
    has_regular_results = st.session_state.get('analysis_results') is not None
    has_enhanced_results = st.session_state.get('enhanced_analysis_results') is not None

    if has_enhanced_results:
        st.info("📊 Displaying visualizations based on Enhanced Cluster Analysis Results.")
    elif has_regular_results:
        st.info("📊 Displaying visualizations based on Standard Analysis Results.")
    else:
        st.warning("No analysis results found to generate visualizations.")
        return
        
    generate_enhanced_cost_visualizations() # This function should dynamically pick enhanced/standard
    
    st.markdown("---")
    st.markdown("### 🗓️ Migration Timeline")
    if st.session_state.get('migration_params'):
        timeline_fig = create_migration_timeline_chart(st.session_state.migration_params)
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("Configure migration parameters to see the timeline.")

def show_ai_insights_tab():
    """Show AI-generated insights and recommendations."""
    st.markdown("## 🤖 AI-Powered Insights")

    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
        st.info("👆 Please run the migration analysis first to generate AI insights.")
        return

    if 'ai_analysis_results' not in st.session_state or not st.session_state.ai_analysis_results.get('success'):
        st.warning("AI insights not yet generated or failed. Click the button below.")
        if st.button("✨ Generate AI Insights", type="primary"):
            with st.spinner("Generating insights with Claude AI..."):
                analyzer = MigrationAnalyzer(st.session_state.migration_params.get('anthropic_api_key'))
                ai_results = analyzer.generate_ai_insights_sync(
                    st.session_state.analysis_results, st.session_state.migration_params
                )
                st.session_state.ai_analysis_results = ai_results
                if ai_results.get('success'):
                    st.success("AI insights generated successfully!")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to generate AI insights: {ai_results.get('error', 'Unknown error')}")
        return

    ai_results = st.session_state.ai_analysis_results

    if ai_results.get('success'):
        st.markdown("### Summary of AI Analysis")
        st.markdown(ai_results['ai_analysis'])
        st.caption(f"Source: {ai_results.get('source', 'N/A')} | Model: {ai_results.get('model', 'N/A')}")
    else:
        st.error(f"Error generating AI insights: {ai_results.get('error', 'No error message')}")
        if "pip install anthropic" in ai_results.get('error', ''):
            st.code("pip install anthropic")

def show_timeline_analysis_tab():
    """Show a detailed migration timeline analysis."""
    st.markdown("## 🗓️ Detailed Timeline Analysis")

    if not st.session_state.get('migration_params'):
        st.info("👆 Please configure migration parameters to see the timeline analysis.")
        return

    migration_params = st.session_state.migration_params
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)

    st.markdown(f"### Project Timeline: **{timeline_weeks} weeks**")

    # Dynamic phase breakdown (example)
    phases = [
        {'phase': 'Discovery & Assessment', 'weeks': 2, 'description': 'Analyze current environment, define scope, identify risks.'},
        {'phase': 'Planning & Design', 'weeks': 3, 'description': 'Develop migration strategy, architecture design, resource planning.'},
        {'phase': 'Data Migration & Replication', 'weeks': 4, 'description': 'Set up AWS DMS, initial data load, continuous replication (CDC).'},
        {'phase': 'Application Refactoring & Testing', 'weeks': 3, 'description': 'Modify applications for AWS, comprehensive testing of functionality and performance.'},
        {'phase': 'Cutover & Validation', 'weeks': 1, 'description': 'Final data sync, DNS cutover, post-migration validation.'},
        {'phase': 'Optimization & Handover', 'weeks': 1, 'description': 'Performance tuning, cost optimization, documentation, team training.'}
    ]

    # Adjust durations to fit total timeline_weeks proportionally
    total_default_duration = sum(p['weeks'] for p in phases)
    scale_factor = timeline_weeks / total_default_duration if total_default_duration > 0 else 1

    current_week = 0
    st.markdown("#### 🗺️ Phase Breakdown")
    for phase in phases:
        duration = max(1, round(phase['weeks'] * scale_factor)) # Ensure at least 1 week
        start_week = current_week
        end_week = current_week + duration
        st.markdown(f"##### {phase['phase']} (Weeks {int(start_week)+1}-{int(end_week)})")
        st.write(f"- Duration: {duration} weeks")
        st.write(f"- Description: {phase['description']}")
        current_week = end_week

        # Progress bar for current phase (illustrative)
        progress = (current_week / timeline_weeks) if timeline_weeks > 0 else 0
        st.progress(min(progress, 1.0)) # Cap at 1.0

    st.markdown("---")

    # Key milestones
    st.markdown("#### 🎯 Key Milestones")
    milestones = [
        f"Week {int(phases[0]['weeks'] * scale_factor)}: Assessment Complete",
        f"Week {int(sum(p['weeks'] for p in phases[:2]) * scale_factor)}: Design Complete",
        f"Week {int(sum(p['weeks'] for p in phases[:3]) * scale_factor)}: Data Migration Ready",
        f"Week {int(sum(p['weeks'] for p in phases[:4]) * scale_factor)}: Application Tested",
        f"Week {timeline_weeks}: Go-Live"
    ]
    for milestone in milestones:
        st.markdown(f"• {milestone}")

    st.markdown("---")

    # Team and resources
    st.markdown("#### 👥 Team & Resources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Team Configuration:**")
        st.write(f"Team Size: {migration_params.get('team_size', 5)} people")
        st.write(f"Expertise Level: {migration_params.get('team_expertise', 'medium').title()}")
        st.write(f"Timeline: {timeline_weeks} weeks")
    with col2:
        st.markdown("**Migration Budget:**")
        budget = migration_params.get('migration_budget', 500000)
        st.write(f"Total Budget: ${budget:,.0f}")
        weekly_budget = budget / timeline_weeks if timeline_weeks > 0 else 0
        st.write(f"Weekly Budget: ${weekly_budget:,.0f}")

def create_growth_projection_charts(growth_analysis: Dict):
    """Create interactive charts for growth projection."""
    st.markdown("### 📈 Growth Projection Visuals")

    if not growth_analysis or 'yearly_projection' not in growth_analysis:
        st.warning("No growth analysis data available to visualize.")
        return

    yearly_df = pd.DataFrame(growth_analysis['yearly_projection'])

    # Line chart for annual cost projection
    fig_line = px.line(yearly_df, x='year', y='cost',
                       title='Projected Annual Cost Over 3 Years',
                       markers=True, text='cost')
    fig_line.update_traces(texttemplate='$%{text:,.0f}', textposition='top center')
    fig_line.update_layout(xaxis_title="Year", yaxis_title="Projected Annual Cost ($)",
                           hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar chart for year-over-year growth
    yearly_df['growth_amount'] = yearly_df['cost'].diff().fillna(0)
    yearly_df['growth_percent'] = yearly_df['cost'].pct_change().fillna(0) * 100
    
    fig_bar = px.bar(yearly_df, x='year', y='growth_amount',
                     title='Year-over-Year Cost Increase',
                     color='growth_percent',
                     color_continuous_scale=px.colors.sequential.Viridis,
                     text='growth_amount')
    fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_bar.update_layout(xaxis_title="Year", yaxis_title="Cost Increase ($)",
                           hovermode="x unified")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Display growth summary metrics
    st.markdown("#### 📊 Growth Summary")
    growth_summary = growth_analysis['growth_summary']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total 3-Year Growth", f"{growth_summary['total_3_year_growth_percent']:.1f}%")
    with col2:
        st.metric("Year 0 Annual Cost", f"${growth_summary['year_0_cost']:,.0f}")
    with col3:
        st.metric("Year 3 Projected Cost", f"${yearly_df[yearly_df['year'] == 3]['cost'].iloc[0]:,.0f}")

class GrowthAwareCostAnalyzer:
    """
    Analyzes and projects AWS migration costs over 3 years,
    incorporating an annual growth rate.
    """
    def __init__(self):
        pass

    def calculate_3_year_growth_projection(self, analysis_results: Dict, migration_params: Dict) -> Dict:
        """
        Calculates the 3-year cost projection based on an annual growth rate.

        Args:
            analysis_results (Dict): The initial cost analysis results.
            migration_params (Dict): Migration parameters including annual_data_growth.

        Returns:
            Dict: A dictionary containing the growth projection data.
        """
        annual_growth_rate = migration_params.get('annual_data_growth', 15) / 100
        initial_annual_cost = analysis_results.get('annual_aws_cost', 0)

        yearly_projection = []
        current_annual_cost = initial_annual_cost

        for year in range(4): # Years 0, 1, 2, 3
            yearly_projection.append({'year': year, 'cost': current_annual_cost})
            current_annual_cost *= (1 + annual_growth_rate)

        # Calculate total 3-year growth percentage (from year 0 to year 3)
        year_0_cost = yearly_projection[0]['cost']
        year_3_cost = yearly_projection[3]['cost']
        
        total_3_year_growth_percent = ((year_3_cost - year_0_cost) / year_0_cost) * 100 if year_0_cost else 0

        return {
            'yearly_projection': yearly_projection,
            'growth_summary': {
                'initial_annual_cost': initial_annual_cost,
                'annual_growth_rate': annual_growth_rate,
                'total_3_year_growth_percent': total_3_year_growth_percent,
                'year_0_cost': year_0_cost # Explicitly add year 0 cost
            }
        }

def validate_growth_analysis_functions():
    """
    Validates that all required functions for growth analysis are defined.
    """
    required_functions = [
        'refresh_cost_calculations',
        'refresh_basic_costs',
        'refresh_growth_analysis',
        'update_cost_session_state',
        'display_refreshed_metrics',
        'refresh_specific_environment_costs',
        'auto_refresh_costs',
        'export_refreshed_costs',
        'integrate_cost_refresh_ui',
        'main_cost_refresh_section',
        'create_growth_projection_charts'
    ]
    
    missing_functions = []
    
    # Use globals() to check for function existence
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        st.error(f"❌ Missing functions: {', '.join(missing_functions)}")
        return False
    else:
        st.success("✅ All growth analysis functions are properly defined!")
        return True

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
            st.warning("⚠️ Growth analysis data not found in session state. Run analysis first.")

# ===========================
# PDF REPORT GENERATION
# ===========================

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self._verify_styles_exist()

    def safe_get_style(self, style_name: str, fallback_parent_style: str = 'Normal') -> ParagraphStyle:
        """Safely gets a style, creating a fallback if it doesn't exist."""
        if style_name in self.styles:
            return self.styles[style_name]
        else:
            st.warning(f"Style '{style_name}' not found. Using fallback '{fallback_parent_style}'.")
            return self.styles[fallback_parent_style]

    def _verify_styles_exist(self):
        """Verify that all required styles are available"""
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
            else:
                # Generic fallback
                self.styles.add(ParagraphStyle(
                    name=style_name, parent=base_style, fontSize=12, fontName='Helvetica'
                ))
        except Exception as e:
            print(f"Error creating fallback style {style_name}: {e}")

    def setup_custom_styles(self):
        """Setup improved custom styles for the report, checking for existence."""
        try:
            # Report Title Style
            if 'ReportTitle' not in self.styles.byName:
                self.styles.add(ParagraphStyle(
                    name='ReportTitle', parent=self.styles['Title'], fontSize=24,
                    spaceBefore=0, spaceAfter=20, textColor=colors.darkblue,
                    alignment=TA_CENTER, fontName='Helvetica-Bold'
                ))
            # Section Header Style
            if 'SectionHeader' not in self.styles.byName:
                self.styles.add(ParagraphStyle(
                    name='SectionHeader', parent=self.styles['Heading1'], fontSize=16,
                    spaceBefore=20, spaceAfter=12, textColor=colors.darkblue,
                    fontName='Helvetica-Bold', borderWidth=1, borderColor=colors.darkblue,
                    borderPadding=5
                ))
            # Subsection Header Style
            if 'SubsectionHeader' not in self.styles.byName:
                self.styles.add(ParagraphStyle(
                    name='SubsectionHeader', parent=self.styles['Heading2'], fontSize=14,
                    spaceBefore=15, spaceAfter=8, textColor=colors.darkgreen,
                    fontName='Helvetica-Bold'
                ))
            # Body Text Style
            if 'BodyText' not in self.styles.byName: # Crucial fix for the KeyError
                self.styles.add(ParagraphStyle(
                    name='BodyText', parent=self.styles['Normal'], fontSize=11,
                    spaceBefore=6, spaceAfter=6, textColor=colors.black,
                    fontName='Helvetica', leading=14
                ))
            # Key Metric Style
            if 'KeyMetric' not in self.styles.byName:
                self.styles.add(ParagraphStyle(
                    name='KeyMetric', parent=self.styles['Normal'], fontSize=12,
                    textColor=colors.darkred, fontName='Helvetica-Bold', alignment=TA_CENTER
                ))
            # Table Header Style
            if 'TableHeader' not in self.styles.byName:
                self.styles.add(ParagraphStyle(
                    name='TableHeader', parent=self.styles['Normal'], fontSize=10,
                    textColor=colors.white, fontName='Helvetica-Bold', alignment=TA_CENTER
                ))
            # Bullet Point Style
            if 'BulletPoint' not in self.styles.byName:
                self.styles.add(ParagraphStyle(
                    name='BulletPoint', parent=self.styles['Normal'], fontSize=11,
                    spaceBefore=4, spaceAfter=4, leftIndent=20, bulletIndent=10,
                    fontName='Helvetica'
                ))
        except Exception as e:
            print(f"Error setting up custom styles: {e}")
            self._create_basic_styles() # Fallback

    def _create_basic_styles(self):
        """Create basic styles as fallback"""
        basic_styles = {
            'ReportTitle': ('Title', 24),
            'SectionHeader': ('Heading1', 16),
            'SubsectionHeader': ('Heading2', 14),
            'BodyText': ('Normal', 11),
            'KeyMetric': ('Normal', 12),
            'TableHeader': ('Normal', 10),
            'BulletPoint': ('Normal', 11)
        }
        for style_name, (parent, size) in basic_styles.items():
            try:
                if style_name not in self.styles.byName:
                    self.styles.add(ParagraphStyle(
                        name=style_name, parent=self.styles[parent], fontSize=size,
                        fontName='Helvetica'
                    ))
            except Exception as e:
                print(f"Error creating basic style {style_name}: {e}")

    def generate_executive_summary_pdf_robust(self, analysis_results: Dict, migration_params: Dict) -> io.BytesIO:
        """Generate a robust executive summary PDF report with improved styling and content."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        
        story = []

        # Title and header
        story.append(Paragraph("AWS Database Migration Analysis", self.safe_get_style('ReportTitle')))
        story.append(Paragraph("Executive Summary Report", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", self.safe_get_style('BodyText')))
        story.append(Spacer(1, 20))

        # Key metrics table
        cost_analysis = analysis_results
        migration_costs = cost_analysis.get('migration_costs', {})
        
        # Include 3-year growth if available
        growth_projection_cost = 0
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_data = st.session_state.growth_analysis['yearly_projection']
            if len(growth_data) > 3: # Year 3 data
                growth_projection_cost = growth_data[3]['cost']

        metrics_data = [
            ['Metric', 'Value', 'Impact'],
            ['Monthly AWS Cost', f"${cost_analysis.get('monthly_aws_cost', 0):,.0f}", 'Operational Efficiency'],
            ['Annual AWS Cost', f"${cost_analysis.get('annual_aws_cost', 0):,.0f}", 'Annual Budget Planning'],
            ['Migration Investment', f"${migration_costs.get('total', 0):,.0f}", 'One-time Project Cost'],
            ['3-Year Projected Cost', f"${growth_projection_cost:,.0f}", 'Long-term Financial Outlook'],
            ['Risk Level', st.session_state.get('risk_assessment', {}).get('overall_risk_level', 'N/A'), 'Project Risk Management']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Overview and key findings
        story.append(Paragraph("1. Project Overview & Key Findings", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(f"This report outlines the proposed migration of your database infrastructure from **{migration_params.get('source_engine', 'N/A')}** to **{migration_params.get('target_engine', 'N/A')}** on AWS. The analysis covers cost projections, performance considerations, and potential risks associated with the transition.", self.safe_get_style('BodyText')))
        story.append(Paragraph("Key findings include:", self.safe_get_style('BodyText')))
        
        findings = [
            f"Optimized AWS monthly operational cost of **${cost_analysis.get('monthly_aws_cost', 0):,.0f}** after migration.",
            f"A one-time migration investment of **${migration_costs.get('total', 0):,.0f}** covering DMS, data transfer, and professional services.",
            f"Projected **{st.session_state.get('growth_analysis', {}).get('growth_summary', {}).get('total_3_year_growth_percent', 0):.1f}%** cost growth over 3 years, necessitating a growth-aware strategy.",
            f"Overall migration risk assessed as **{st.session_state.get('risk_assessment', {}).get('overall_risk_level', 'N/A')}**, with specific mitigations identified for critical areas."
        ]
        for finding in findings:
            story.append(Paragraph(f"• {finding}", self.safe_get_style('BulletPoint')))
        story.append(Spacer(1, 15))

        # Recommendations
        story.append(Paragraph("2. Strategic Recommendations", self.safe_get_style('SectionHeader')))
        recommendations = [
            "**Phased Migration Approach:** Start with non-production environments to refine the process and minimize risks for critical systems.",
            "**AWS DMS Utilization:** Leverage AWS Database Migration Service for efficient and continuous data replication, minimizing downtime.",
            "**Performance Optimization:** For high-traffic environments, consider AWS Aurora with read replicas for enhanced scalability and performance.",
            "**Cost Monitoring & Optimization:** Implement AWS Cost Explorer and Reserved Instances/Savings Plans post-migration to manage ongoing expenses.",
            "**Comprehensive Testing:** Conduct rigorous application and performance testing in the AWS environment before cutover to ensure stability and user experience."
        ]
        for rec in recommendations:
            story.append(Paragraph(rec, self.safe_get_style('BulletPoint')))
        story.append(Spacer(1, 15))

        # Financial Impact & ROI
        story.append(Paragraph("3. Financial Impact & ROI", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(
            "The migration is expected to yield significant long-term benefits, including reduced operational overhead, improved scalability, and enhanced security. The initial migration investment is projected to be offset by operational savings over time, leading to a positive Return on Investment (ROI) within an estimated timeframe.", self.safe_get_style('BodyText')))
        
        # Add a simple line chart for cost projection if available
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_data = st.session_state.growth_analysis['yearly_projection']
            yearly_df = pd.DataFrame(growth_data)
            fig_line = px.line(yearly_df, x='year', y='cost',
                               title='Projected Annual Cost Over 3 Years',
                               markers=True)
            fig_line.update_layout(xaxis_title="Year", yaxis_title="Annual Cost ($)",
                                   showlegend=False)
            
            # Save chart to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_line.write_image(tmpfile.name)
                image_path = tmpfile.name
            
            try:
                img = Image(image_path, width=400, height=250)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 10))
                story.append(Paragraph("Figure 1: 3-Year Annual Cost Projection", self.safe_get_style('BodyText'), alignment=TA_CENTER))
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"Error embedding chart: {e}")
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path) # Clean up temp file

        story.append(Spacer(1, 15))
        story.append(Paragraph("4. Next Steps", self.safe_get_style('SectionHeader')))
        next_steps = [
            "**Stakeholder Alignment:** Present this summary to key stakeholders for approval and resource allocation.",
            "**Detailed Planning:** Develop a granular migration plan, including specific timelines, responsibilities, and success metrics.",
            "**Pilot Program:** Initiate a pilot migration with non-critical databases to validate the strategy and identify unforeseen challenges."
        ]
        for step in next_steps:
            story.append(Paragraph(f"• {step}", self.safe_get_style('BulletPoint')))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

    def generate_technical_report_pdf_robust(self, analysis_results: Dict, migration_params: Dict, environment_specs: Dict, recommendations: Dict) -> io.BytesIO:
        """Generate a robust technical PDF report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        story = []

        # Title Page
        story.append(Paragraph("AWS Database Migration Project", self.safe_get_style('ReportTitle')))
        story.append(Paragraph("Technical Deep Dive Report", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(f"For: {migration_params.get('source_engine', 'N/A')} to {migration_params.get('target_engine', 'N/A')} Migration", self.safe_get_style('SubsectionHeader')))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", self.safe_get_style('BodyText')))
        story.append(PageBreak())

        # Table of Contents
        story.append(Paragraph("Table of Contents", self.safe_get_style('SectionHeader')))
        story.append(Paragraph("1. Executive Summary", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("2. Current Environment Analysis", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("3. AWS Recommended Architecture", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("4. Cost Analysis Breakdown", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("5. Migration Strategy & Timeline", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("6. Risk Assessment & Mitigation", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("7. Performance & Scalability Considerations", self.safe_get_style('BulletPoint')))
        story.append(Paragraph("8. Appendix: Detailed Specifications", self.safe_get_style('BulletPoint')))
        story.append(PageBreak())

        # 1. Executive Summary (Brief)
        story.append(Paragraph("1. Executive Summary", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(
            "This document provides a technical deep dive into the proposed AWS database migration, detailing current environment analysis, recommended AWS architecture, comprehensive cost breakdown, migration strategy, risk assessment, and performance considerations. It serves as a technical blueprint for the migration project.", self.safe_get_style('BodyText')))
        story.append(Spacer(1, 10))

        # 2. Current Environment Analysis
        story.append(Paragraph("2. Current Environment Analysis", self.safe_get_style('SectionHeader')))
        story.append(Paragraph("A detailed analysis of the existing database environments, including resource utilization and key characteristics:", self.safe_get_style('BodyText')))
        story.append(Spacer(1, 10))

        env_analysis_data = []
        # Check if environment_specs is structured for enhanced or simple
        is_enhanced = False
        if environment_specs:
            first_env_key = next(iter(environment_specs))
            if 'writer' in environment_specs[first_env_key]:
                is_enhanced = True
        
        if is_enhanced:
            env_analysis_data.append(['Environment', 'Type', 'CPU Cores (Writer)', 'RAM GB (Writer)', 'Readers (Count)', 'Storage GB', 'IOPS Req', 'Connections'])
            for env_name, specs in environment_specs.items():
                writer = specs.get('writer', {})
                readers = specs.get('readers', {})
                storage = specs.get('storage', {})
                env_analysis_data.append([
                    env_name,
                    specs.get('environment_type', 'N/A').title(),
                    writer.get('cpu_cores', 'N/A'),
                    writer.get('ram_gb', 'N/A'),
                    readers.get('count', 0),
                    storage.get('size_gb', 'N/A'),
                    storage.get('iops', 'N/A'),
                    specs.get('peak_connections', 'N/A')
                ])
        else: # Simple environment specs
            env_analysis_data.append(['Environment', 'CPU Cores', 'RAM GB', 'Storage GB', 'IOPS Req', 'Connections'])
            for env_name, specs in environment_specs.items():
                env_analysis_data.append([
                    env_name,
                    specs.get('cpu_cores', 'N/A'),
                    specs.get('ram_gb', 'N/A'),
                    specs.get('storage_gb', 'N/A'),
                    specs.get('iops_requirement', 'N/A'),
                    specs.get('peak_connections', 'N/A')
                ])

        env_table = Table(env_analysis_data)
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#edf2f7')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#cbd5e0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(env_table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

        # 3. AWS Recommended Architecture
        story.append(Paragraph("3. AWS Recommended Architecture", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(
            "Based on the current environment analysis and migration parameters, the following AWS RDS/Aurora architecture is recommended:", self.safe_get_style('BodyText')))
        story.append(Spacer(1, 10))

        for env_name, rec in recommendations.items():
            story.append(Paragraph(f"**Environment: {env_name} ({rec.get('environment_type', 'N/A').title()})**", self.safe_get_style('SubsectionHeader')))
            
            rec_data = [
                ['Component', 'Details'],
                ['Database Engine', migration_params.get('target_engine', 'N/A').title()],
                ['Writer Instance', f"{rec['writer']['instance_class']} (Multi-AZ: {'Yes' if rec['writer']['multi_az'] else 'No'})"] if 'writer' in rec else [
                    'Instance Type', f"{rec.get('instance_class', 'N/A')} (Multi-AZ: {'Yes' if rec.get('multi_az', False) else 'No'})"],
                ['Read Replicas', f"{rec['readers']['count']} x {rec['readers']['instance_class']}" if rec.get('readers', {}).get('count', 0) > 0 else "None"],
                ['Storage Configuration', f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()} ({rec['storage']['iops']} IOPS, Encrypted: {'Yes' if rec['storage']['encrypted'] else 'No'})"],
                ['Workload Pattern', f"{rec.get('workload_pattern', 'N/A')} ({rec.get('read_write_ratio', 'N/A')}% Reads)"],
                ['Backup Retention', f"{rec['storage']['backup_retention_days']} days"],
                ['Peak Connections', str(rec.get('connections', 'N/A'))]
            ]
            
            rec_table = Table(rec_data, colWidths=[2*inch, 4*inch])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e0e7ff')),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#a7b9f5')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(rec_table)
            story.append(Spacer(1, 15))
        story.append(PageBreak())

        # 4. Cost Analysis Breakdown
        story.append(Paragraph("4. Cost Analysis Breakdown", self.safe_get_style('SectionHeader')))
        story.append(Paragraph("Detailed cost projections for AWS services and migration efforts:", self.safe_get_style('BodyText')))
        story.append(Spacer(1, 10))

        # Monthly AWS Costs
        story.append(Paragraph("4.1 Monthly AWS Operational Costs", self.safe_get_style('SubsectionHeader')))
        monthly_cost_data = [['Environment', 'Instance Cost', 'Storage Cost', 'I/O Cost', 'Total Monthly']]
        if 'environment_costs' in analysis_results:
            for env, costs in analysis_results['environment_costs'].items():
                monthly_cost_data.append([
                    env,
                    f"${costs.get('writer_monthly_cost', costs.get('instance_monthly_cost', 0)):,.0f}",
                    f"${costs.get('storage_monthly_cost', 0):,.0f}",
                    f"${costs.get('io_monthly_cost', costs.get('iops_monthly_cost', 0)):,.0f}",
                    f"${costs.get('total_monthly', 0):,.0f}"
                ])
        monthly_cost_table = Table(monthly_cost_data)
        monthly_cost_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F5E9')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#C8E6C9')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(monthly_cost_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"**Total Estimated Monthly AWS Cost: ${analysis_results.get('monthly_aws_cost', 0):,.0f}**", self.safe_get_style('KeyMetric'), alignment=TA_RIGHT))
        story.append(Spacer(1, 15))

        # Migration Costs
        story.append(Paragraph("4.2 One-Time Migration Investment", self.safe_get_style('SubsectionHeader')))
        migration_costs = analysis_results.get('migration_costs', {})
        migration_cost_data = [
            ['Category', 'Cost'],
            ['DMS Instance Costs', f"${migration_costs.get('dms_instance', 0):,.0f}"],
            ['Data Transfer Costs', f"${migration_costs.get('data_transfer', 0):,.0f}"],
            ['Professional Services', f"${migration_costs.get('professional_services', 0):,.0f}"],
            ['Contingency (20%)', f"${migration_costs.get('contingency', 0):,.0f}"]
        ]
        migration_cost_table = Table(migration_cost_data, colWidths=[2.5*inch, 2*inch])
        migration_cost_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF9800')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFF3E0')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#FFCC80')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(migration_cost_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"**Total One-Time Migration Investment: ${migration_costs.get('total', 0):,.0f}**", self.safe_get_style('KeyMetric'), alignment=TA_RIGHT))
        story.append(Spacer(1, 15))
        story.append(PageBreak())

        # 5. Migration Strategy & Timeline
        story.append(Paragraph("5. Migration Strategy & Timeline", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(
            "The migration will follow a structured approach with defined phases and a projected timeline. A 'Lift and Shift' or 'Re-platforming' strategy will be adopted based on the database engine compatibility and application requirements.", self.safe_get_style('BodyText')))
        
        # Timeline chart
        if st.session_state.get('migration_params'):
            timeline_fig = create_migration_timeline_chart(st.session_state.migration_params)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                timeline_fig.write_image(tmpfile.name)
                image_path = tmpfile.name
            
            try:
                img = Image(image_path, width=500, height=300)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 10))
                story.append(Paragraph("Figure 2: Migration Project Timeline", self.safe_get_style('BodyText'), alignment=TA_CENTER))
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"Error embedding timeline chart: {e}")
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

        story.append(Spacer(1, 15))
        story.append(PageBreak())

        # 6. Risk Assessment & Mitigation
        story.append(Paragraph("6. Risk Assessment & Mitigation", self.safe_get_style('SectionHeader')))
        risk_data = st.session_state.get('risk_assessment', {})
        story.append(Paragraph(
            f"An assessment of potential risks associated with the migration has been conducted, categorizing the overall risk level as **{risk_data.get('overall_risk_level', 'N/A')}**.", self.safe_get_style('BodyText')))
        
        if risk_data.get('risks'):
            story.append(Paragraph("Identified risks and proposed mitigations:", self.safe_get_style('BodyText')))
            for risk in risk_data['risks']:
                story.append(Paragraph(f"• {risk}", self.safe_get_style('BulletPoint')))
        else:
            story.append(Paragraph("No significant risks identified.", self.safe_get_style('BodyText')))
        story.append(Spacer(1, 15))
        story.append(PageBreak())

        # 7. Performance & Scalability Considerations
        story.append(Paragraph("7. Performance & Scalability Considerations", self.safe_get_style('SectionHeader')))
        story.append(Paragraph(
            "The recommended AWS architecture ensures high performance and scalability to meet current and future demands:", self.safe_get_style('BodyText')))
        
        perf_cons = [
            "**Scalable Instances:** Utilizing AWS RDS/Aurora instances capable of scaling compute and memory independently.",
            "**Read Replicas:** Implementing read replicas to offload read-heavy workloads and improve query performance.",
            "**Optimized Storage:** Selection of appropriate EBS storage types (e.g., gp3, io2) with provisioned IOPS to meet performance requirements.",
            "**Auto-scaling:** Leveraging Aurora's auto-scaling capabilities for serverless or provisioned clusters to automatically adjust capacity.",
            "**Monitoring & Alerting:** Setting up comprehensive CloudWatch monitoring and alerting for proactive performance management."
        ]
        for item in perf_cons:
            story.append(Paragraph(f"• {item}", self.safe_get_style('BulletPoint')))
        story.append(Spacer(1, 15))

        # 8. Appendix: Detailed Specifications
        story.append(Paragraph("8. Appendix: Detailed Specifications", self.safe_get_style('SectionHeader')))
        story.append(Paragraph("For a more granular view of each environment's recommended specifications:", self.safe_get_style('BodyText')))
        
        for env_name, rec in recommendations.items():
            story.append(Paragraph(f"**Environment: {env_name}**", self.safe_get_style('SubsectionHeader')))
            
            detail_data = []
            if 'writer' in rec: # Enhanced
                detail_data.extend([
                    ['Component', 'Detail'],
                    ['Writer Instance Class', rec['writer']['instance_class']],
                    ['Writer Multi-AZ', 'Yes' if rec['writer']['multi_az'] else 'No'],
                    ['Writer CPU Cores', str(rec['writer']['cpu_cores'])],
                    ['Writer RAM GB', str(rec['writer']['ram_gb'])],
                    ['Reader Count', str(rec['readers']['count'])],
                    ['Reader Instance Class', rec['readers']['instance_class']],
                    ['Storage Size GB', str(rec['storage']['size_gb'])],
                    ['Storage Type', rec['storage']['type'].upper()],
                    ['Storage IOPS', str(rec['storage']['iops'])],
                    ['Encrypted', 'Yes' if rec['storage']['encrypted'] else 'No'],
                    ['Backup Retention Days', str(rec['storage']['backup_retention_days'])],
                    ['Workload Pattern', rec['workload_pattern']],
                    ['Read/Write Ratio', f"{rec['read_write_ratio']}%"],
                    ['Peak Connections', str(rec['connections'])]
                ])
            else: # Simple
                detail_data.extend([
                    ['Component', 'Detail'],
                    ['Instance Class', rec['instance_class']],
                    ['Multi-AZ', 'Yes' if rec['multi_az'] else 'No'],
                    ['CPU Cores', str(rec['cpu_cores'])],
                    ['RAM GB', str(rec['ram_gb'])],
                    ['Storage Size GB', str(rec['storage_gb'])],
                    ['IOPS Requirement', str(rec['iops_requirement'])],
                    ['Peak Connections', str(rec['peak_connections'])]
                ])

            detail_table = Table(detail_data, colWidths=[2*inch, 3*inch])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#757575')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#E0E0E0')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
            ]))
            story.append(detail_table)
            story.append(Spacer(1, 15))
        story.append(PageBreak())
        
        doc.build(story)
        buffer.seek(0)
        return buffer

def generate_executive_summary_pdf_robust(analysis_results: Dict, migration_params: Dict) -> io.BytesIO:
    """Wrapper to instantiate PDFReportGenerator and call its method."""
    pdf_gen = PDFReportGenerator()
    return pdf_gen.generate_executive_summary_pdf_robust(analysis_results, migration_params)

def generate_technical_report_pdf_robust(analysis_results: Dict, migration_params: Dict, environment_specs: Dict, recommendations: Dict) -> io.BytesIO:
    """Wrapper to instantiate PDFReportGenerator and call its method."""
    pdf_gen = PDFReportGenerator()
    return pdf_gen.generate_technical_report_pdf_robust(analysis_results, migration_params, environment_specs, recommendations)

# ===========================
# NETWORK TRANSFER ANALYSIS MODULE
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
                'description': 'Data transfer over public internet using AWS DMS for replication. Suitable for smaller datasets or less strict latency requirements.',
                'components': ['On-Premises Database', 'Internet', 'AWS DMS', 'Target AWS RDS/Aurora'],
                'pros': ['Cost-effective for small transfers', 'Simple setup for initial sync', 'Built-in change data capture (CDC)'],
                'cons': ['Variable latency and throughput', 'Security concerns over public internet', 'Impacted by internet congestion'],
                'security_score': 60, # Out of 100
                'reliability_score': 70,
                'bandwidth_utilization': 50 # Percentage of theoretical max
            },
            'direct_connect_dms': {
                'name': 'AWS Direct Connect + DMS',
                'description': 'Dedicated network connection (Direct Connect) for data transfer, combined with AWS DMS for secure and high-throughput replication.',
                'components': ['On-Premises Database', 'AWS Direct Connect', 'AWS DMS', 'Target AWS RDS/Aurora'],
                'pros': ['High bandwidth and consistent network performance', 'Reduced network costs for large datasets', 'Enhanced security (private connection)', 'Lower latency'],
                'cons': ['Higher initial setup cost', 'Longer setup time for Direct Connect', 'Requires physical connectivity'],
                'security_score': 90,
                'reliability_score': 95,
                'bandwidth_utilization': 80
            },
            'vpn_dms': {
                'name': 'VPN + DMS',
                'description': 'Secure IPsec VPN tunnel over the internet with AWS DMS for data replication. A balance between cost and security.',
                'components': ['On-Premises Database', 'VPN Gateway', 'Internet', 'AWS DMS', 'Target AWS RDS/Aurora'],
                'pros': ['Secure communication over public internet', 'Cost-effective compared to Direct Connect', 'Easier to set up than Direct Connect'],
                'cons': ['Limited bandwidth compared to Direct Connect', 'Performance can vary based on internet quality', 'Requires VPN device/software'],
                'security_score': 80,
                'reliability_score': 80,
                'bandwidth_utilization': 60
            },
            's3_snowball': {
                'name': 'S3 + AWS Snowball/Snowmobile',
                'description': 'Offline data transfer using AWS Snowball or Snowmobile devices for extremely large datasets or environments with limited bandwidth.',
                'components': ['On-Premises Database', 'AWS Snowball/Snowmobile', 'AWS S3', 'AWS DMS (optional)', 'Target AWS RDS/Aurora'],
                'pros': ['Extremely high capacity for data transfer', 'Bypasses internet bandwidth limitations', 'Secure physical transfer', 'Cost-effective for petabytes'],
                'cons': ['Takes longer due to shipping and processing', 'Not suitable for ongoing replication (CDC)', 'Initial setup complexity'],
                'security_score': 95,
                'reliability_score': 98,
                'bandwidth_utilization': 100 # Conceptually, as it's physical transfer
            }
        }

    def _initialize_region_bandwidth(self) -> Dict:
        """Initialize sample inter-region bandwidth figures (illustrative)"""
        # These are illustrative values, not real AWS bandwidth guarantees
        return {
            'us-east-1': {
                'us-west-2': 1000, # Mbps
                'eu-west-1': 500,
                'ap-southeast-2': 300
            },
            'us-west-2': {
                'us-east-1': 1000,
                'ap-southeast-2': 800
            }
        }

    def analyze_transfer_options(self, data_size_gb: float, migration_params: Dict) -> Dict:
        """Analyze network transfer options and recommend the best approach."""
        
        region = migration_params.get('region', 'us-east-1')
        source_location = migration_params.get('source_location', 'on-premises')
        
        analysis_results = {}
        
        for pattern_id, pattern_info in self.transfer_patterns.items():
            estimated_cost = 0
            estimated_time_days = 0
            
            # Simple cost and time estimation based on pattern
            if pattern_id == 'internet_dms':
                estimated_cost = data_size_gb * 0.09 # $0.09/GB egress
                estimated_time_days = data_size_gb / (500 * 60 * 60 * 24 / (8*1024)) # 500 Mbps for 8 hours a day, GB to Mbps conversion
                if estimated_time_days < 1: estimated_time_days = 1 # Minimum 1 day
            elif pattern_id == 'direct_connect_dms':
                estimated_cost = (data_size_gb * 0.02) + 500 # $0.02/GB egress + base DC cost
                estimated_time_days = data_size_gb / (5000 * 60 * 60 * 24 / (8*1024)) # 5000 Mbps for 8 hours a day
                if estimated_time_days < 1: estimated_time_days = 1
            elif pattern_id == 'vpn_dms':
                estimated_cost = (data_size_gb * 0.05) + 50 # $0.05/GB egress + VPN cost
                estimated_time_days = data_size_gb / (100 * 60 * 60 * 24 / (8*1024)) # 100 Mbps
                if estimated_time_days < 1: estimated_time_days = 1
            elif pattern_id == 's3_snowball':
                estimated_cost = (data_size_gb / 1000) * 1000 + 200 # $1000 per TB + device fee
                estimated_time_days = (data_size_gb / 1000) * 7 + 3 # 7 days per TB + shipping (3 days)
                
            analysis_results[pattern_id] = {
                'name': pattern_info['name'],
                'description': pattern_info['description'],
                'estimated_cost': estimated_cost,
                'estimated_time_days': estimated_time_days,
                'security_score': pattern_info['security_score'],
                'reliability_score': pattern_info['reliability_score'],
                'bandwidth_utilization': pattern_info['bandwidth_utilization']
            }
            
        return analysis_results

    def get_network_architecture_diagram(self, selected_pattern: str):
        """Generates a simplified network architecture diagram for the selected pattern."""
        fig = go.Figure()

        pattern_info = self.transfer_patterns.get(selected_pattern, {})

        # Define component positions
        positions = {
            'on_premises': (1, 3),
            'internet': (3, 4),
            'dx': (3, 2),
            'vpn': (3, 3),
            'aws_services': (5, 3),
            'target_db': (7, 3)
        }

        # Add nodes for components
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
        data_size_gb = st.number_input(
            "Total Data Size for Migration (GB)",
            min_value=1, value=st.session_state.migration_params.get('data_size_gb', 1000), step=100
        )
    with col2:
        source_location = st.selectbox(
            "Source Data Location",
            ['on-premises', 'other-cloud'],
            index=0
        )
    with col3:
        target_region = st.selectbox(
            "Target AWS Region",
            ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-2'],
            index=0
        )

    st.markdown("### 🔍 Analyze Transfer Options")
    if st.button("🚀 Analyze Network Transfer", type="primary", use_container_width=True):
        st.session_state.network_analysis_results = analyzer.analyze_transfer_options(
            data_size_gb, st.session_state.migration_params
        )
        st.success("✅ Network transfer analysis complete!")

    if 'network_analysis_results' in st.session_state:
        st.markdown("#### 📈 Network Transfer Options Summary")
        analysis_df = pd.DataFrame.from_dict(st.session_state.network_analysis_results, orient='index')
        analysis_df['estimated_cost'] = analysis_df['estimated_cost'].apply(lambda x: f"${x:,.2f}")
        analysis_df['estimated_time_days'] = analysis_df['estimated_time_days'].apply(lambda x: f"{x:.1f} days")
        st.dataframe(analysis_df[['name', 'estimated_cost', 'estimated_time_days', 'security_score', 'reliability_score']], use_container_width=True)

        st.markdown("#### 🗺️ Network Architecture Visualizer")
        selected_pattern_id = st.selectbox(
            "Select a transfer pattern to visualize its architecture:",
            list(analyzer.transfer_patterns.keys()),
            format_func=lambda x: analyzer.transfer_patterns[x]['name']
        )
        pattern_info = analyzer.transfer_patterns[selected_pattern_id]
        
        st.markdown(f"##### {pattern_info['name']}")
        st.write(pattern_info['description'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔧 Components")
            for component in pattern_info['components']:
                st.markdown(f"• {component}")
        with col2:
            st.markdown("#### 📋 Use Cases")
            for use_case in pattern_info['pros']: # Changed to pros for clarity here
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

        # Performance Metrics
        transfer_analysis = st.session_state.network_analysis_results
        if selected_pattern_id in transfer_analysis:
            metrics = transfer_analysis[selected_pattern_id]
            st.markdown("#### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bandwidth Utilization", f"{metrics['bandwidth_utilization']}%")
            with col2:
                st.metric("Reliability Score", f"{metrics['reliability_score']}/100")
            with col3:
                st.metric("Security Score", f"{metrics['security_score']}/100")

        # Diagram
        st.markdown("#### Diagram")
        fig = analyzer.get_network_architecture_diagram(selected_pattern_id)
        st.plotly_chart(fig, use_container_width=True)


# ===========================
# MAIN APP STRUCTURE
# ===========================

def initialize_session_state():
    """Initialize all necessary session state variables."""
    if 'migration_params' not in st.session_state:
        st.session_state.migration_params = {
            'source_engine': 'PostgreSQL',
            'target_engine': 'Aurora-PostgreSQL',
            'data_size_gb': 1000,
            'migration_timeline_weeks': 12,
            'downtime_tolerance': 'medium',
            'data_security_level': 'high',
            'annual_data_growth': 15, # %
            'anthropic_api_key': '',
            'region': 'us-east-1',
            'use_direct_connect': False, # For network module
            'team_size': 5, # For timeline
            'team_expertise': 'medium', # For timeline
            'migration_budget': 500000, # For timeline
        }
    if 'environment_specs' not in st.session_state:
        st.session_state.environment_specs = {} # Stores basic or enhanced specs
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None # Stores basic analysis results
    if 'enhanced_analysis_results' not in st.session_state:
        st.session_state.enhanced_analysis_results = None # Stores enhanced cluster analysis results
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = {} # Stores basic recommendations
    if 'enhanced_recommendations' not in st.session_state:
        st.session_state.enhanced_recommendations = {} # Stores enhanced cluster recommendations
    if 'risk_assessment' not in st.session_state:
        st.session_state.risk_assessment = None
    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = None
    if 'growth_analysis' not in st.session_state:
        st.session_state.growth_analysis = None
    if 'network_analysis_results' not in st.session_state:
        st.session_state.network_analysis_results = None

import streamlit as st

def show_migration_configuration_tab():
    st.header("⚙️ Migration Configuration")

    # Initialize session state for migration parameters if not already present
    if 'migration_params' not in st.session_state:
        st.session_state.migration_params = {
            'source_engine': None,
            'target_engine': None,
            'region': 'us-east-1',
            'data_size_gb': 100,
            'migration_timeline_weeks': 4,
            'annual_data_growth': 15,
            'use_direct_connect': False,
            'anthropic_api_key': '',
            'existing_licensing': 'no_licensing' # Added to ensure it's always set
        }

    params = st.session_state.migration_params

    st.subheader("Database Details")

    # Define the allowed source engines with consistent casing
    allowed_source_engines = ['PostgreSQL', 'Oracle-EE', 'SQLServer', 'MySQL']
    
    # Use st.selectbox for user input, ensuring the default value is handled
    # and providing a clear list of options
    selected_source_engine = st.selectbox(
        "Source Database Engine",
        options=[''] + allowed_source_engines, # Add an empty string for initial selection
        index=allowed_source_engines.index(params['source_engine']) + 1 if params['source_engine'] in allowed_source_engines else 0,
        help="Select the database engine currently used on-premises."
    )
    
    # Update params['source_engine'] after user selection
    params['source_engine'] = selected_source_engine

    # Add a check to prevent ValueError if an invalid engine is somehow set
    if params['source_engine'] not in allowed_source_engines and params['source_engine'] != '':
        st.warning(f"⚠️ Invalid source engine '{params['source_engine']}'. Please select from the list.")
        # Optionally, reset to a default or clear the invalid entry
        params['source_engine'] = None # Or set to a default valid engine

    # The rest of your migration configuration inputs would follow here,
    # ensuring all 'params' values are properly initialized and validated.

    # Example for target engine (similar validation can be applied)
    allowed_target_engines = ['PostgreSQL', 'Aurora-PostgreSQL', 'MySQL', 'Aurora-MySQL']
    params['target_engine'] = st.selectbox(
        "Target AWS Database Engine",
        options=allowed_target_engines,
        index=allowed_target_engines.index(params['target_engine']) if params['target_engine'] in allowed_target_engines else 0,
        help="Select the target database engine on AWS."
    )
    
    # Example input fields (add your existing fields here)
    params['data_size_gb'] = st.number_input("Total Database Size (GB)", value=params['data_size_gb'], min_value=1, help="Total size of your database(s) in Gigabytes.")
    params['migration_timeline_weeks'] = st.number_input("Estimated Migration Timeline (weeks)", value=params['migration_timeline_weeks'], min_value=1, max_value=52, help="Approximate time needed for the migration project.")
    
    # Save button (ensure it updates st.session_state)
    if st.button("Save Migration Configuration", type="primary"):
        # You might want to add more comprehensive validation here before saving
        st.session_state.migration_params = params
        st.success("Migration configuration saved successfully!")
    
    # Display current configuration
    if st.session_state.migration_params['source_engine']:
        st.markdown("---")
        st.subheader("Current Configuration Summary")
        st.write(f"**Source Engine:** {st.session_state.migration_params.get('source_engine', 'Not set')}")
        st.write(f"**Target Engine:** {st.session_state.migration_params.get('target_engine', 'Not set')}")
        st.write(f"**Data Size:** {st.session_state.migration_params.get('data_size_gb', 0)} GB")
        st.write(f"**Timeline:** {st.session_state.migration_params.get('migration_timeline_weeks', 0)} weeks")

def show_analysis_summary():
    """Shows a summary of the analysis results."""
    st.markdown("### 📊 Analysis Summary")

    if st.session_state.get('enhanced_analysis_results'):
        results = st.session_state.enhanced_analysis_results
        st.info("Using Enhanced Analysis Results for summary.")
    elif st.session_state.get('analysis_results'):
        results = st.session_state.analysis_results
        st.info("Using Standard Analysis Results for summary.")
    else:
        st.warning("No analysis results available. Please run an analysis.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Monthly AWS Cost", f"${results.get('monthly_aws_cost', 0):,.0f}")
    with col2:
        st.metric("Annual AWS Cost", f"${results.get('annual_aws_cost', 0):,.0f}")
    with col3:
        st.metric("Total Migration Cost", f"${results.get('migration_costs', {}).get('total', 0):,.0f}")

    if st.session_state.get('risk_assessment'):
        risk_level = st.session_state.risk_assessment.get('overall_risk_level', 'N/A')
        st.metric("Overall Risk Level", risk_level)
    
    if st.session_state.get('growth_analysis'):
        growth_pct = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
        st.metric("3-Year Cost Growth", f"{growth_pct:.1f}%")

    st.markdown("---")
    st.markdown("### 💡 Key Takeaways")
    # Dynamically generate key takeaways based on analysis
    takeaways = []
    if results.get('monthly_aws_cost', 0) > 10000:
        takeaways.append("Consider Reserved Instances or Savings Plans for significant cost reduction on ongoing AWS costs.")
    if st.session_state.migration_params.get('downtime_tolerance') == 'zero':
        takeaways.append("Zero downtime migration requires advanced strategies like AWS DMS CDC and careful planning for cutover.")
    if st.session_state.migration_params.get('source_engine') != st.session_state.migration_params.get('target_engine'):
        takeaways.append("Heterogeneous migration implies schema conversion and application refactoring efforts. Utilize AWS SCT.")
    if st.session_state.get('risk_assessment', {}).get('overall_risk_level', '') == 'High':
        takeaways.append("High overall migration risk. Prioritize mitigation strategies and conduct extensive testing.")
    if st.session_state.get('growth_analysis', {}).get('total_3_year_growth_percent', 0) > 20:
        takeaways.append("Significant projected growth. Ensure the target architecture can scale efficiently and consider long-term cost implications.")

    if takeaways:
        for i, item in enumerate(takeaways):
            st.markdown(f"- {item}")
    else:
        st.info("No specific key takeaways identified at this moment. Review all sections for details.")

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
        """)
    with col2:
        env_count = len(st.session_state.environment_specs)
        st.markdown(f"**Configured Environments:** {env_count}")
        st.markdown(f"**Downtime Tolerance:** {params['downtime_tolerance'].title()}")
        st.markdown(f"**Data Growth:** {params['annual_data_growth']}% per year")

    st.markdown("---")
    st.markdown("### ✨ Run Analysis")
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        run_streamlit_migration_analysis() # Calls the updated analysis function
    
    st.markdown("---")
    # Show summary if results exist
    if st.session_state.get('analysis_results') or st.session_state.get('enhanced_analysis_results'):
        show_analysis_summary()


def show_reports_section():
    """Show reports and export section - ROBUST VERSION"""
    st.markdown("## 📄 Reports & Export")
    # Check for both regular and enhanced analysis results
    has_regular_results = st.session_state.analysis_results is not None
    has_enhanced_results = hasattr(st.session_state, 'enhanced_analysis_results') and st.session_state.enhanced_analysis_results is not None

    if not has_regular_results and not has_enhanced_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        st.info("👆 Go to 'Analysis & Recommendations' section and click '🚀 Run Comprehensive Analysis'")
        return

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

    # Determine which results to use
    if has_enhanced_results:
        results = st.session_state.enhanced_analysis_results
        recommendations = getattr(st.session_state, 'enhanced_recommendations', {})
        st.info("📊 Using Enhanced Analysis Results for reports.")
    else:
        results = st.session_state.analysis_results
        recommendations = getattr(st.session_state, 'recommendations', {})
        st.info("📊 Using Standard Analysis Results for reports.")

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
                        results, st.session_state.migration_params
                    )
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Executive PDF",
                            data=pdf_buffer,
                            file_name="migration_executive_summary.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF. Buffer is empty.")
                except Exception as e:
                    st.error(f"Error generating executive PDF: {e}")
                    st.code(e)
    with col2:
        st.markdown("#### ⚙️ Technical Report")
        st.markdown("For engineers and project managers")
        st.markdown("**Includes:**")
        st.markdown("• Detailed environment specs")
        st.markdown("• AWS architecture recommendations")
        st.markdown("• Granular cost breakdown")
        st.markdown("• Deep dive into risks & performance")
        if st.button("📄 Generate Technical PDF", key="tech_pdf", use_container_width=True):
            with st.spinner("Generating technical report..."):
                try:
                    pdf_buffer = generate_technical_report_pdf_robust(
                        results, st.session_state.migration_params, st.session_state.environment_specs, recommendations
                    )
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Technical PDF",
                            data=pdf_buffer,
                            file_name="migration_technical_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF. Buffer is empty.")
                except Exception as e:
                    st.error(f"Error generating technical PDF: {e}")
                    st.code(e)
    with col3:
        st.markdown("#### 📈 Cost Projections (CSV)")
        st.markdown("Export detailed cost projections for financial analysis")
        st.markdown("**Includes:**")
        st.markdown("• Monthly & Annual Costs")
        st.markdown("• 3-Year Growth Projections")
        st.markdown("• Migration Investment Details")
        if st.button("📥 Export Cost Data (CSV)", key="cost_csv", use_container_width=True):
            export_refreshed_costs() # Re-using existing export function


def main():
    """Main Streamlit application logic."""
    st.set_page_config(layout="wide", page_title="AWS DB Migration Planner", page_icon="☁️")

    st.title("☁️ AWS Database Migration Planner")
    st.subheader("Plan your database migration to AWS with confidence using AI-powered insights.")

    initialize_session_state()

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "⚙️ Migration Configuration",
            "📊 Environment Setup",
            "🚀 Analysis & Recommendations",
            "🌐 Network Transfer Analysis", # Added Network Tab
            "📈 Data Visualizations",
            "⚠️ Risk Assessment",
            "🤖 AI-Powered Insights",
            "🗓️ Detailed Timeline",
            "📄 Reports & Export"
        ]
    )

    if page == "🏠 Home":
        st.markdown("""
        Welcome to the AWS Database Migration Planner! This interactive tool helps you:
        - Define your current database environment specifications.
        - Configure your migration parameters, including target AWS services.
        - Get instant cost analysis and AWS instance recommendations.
        - Assess migration risks and receive AI-powered mitigation strategies.
        - Visualize cost projections and migration timelines.
        - Generate comprehensive executive and technical PDF reports.

        Navigate through the sections using the sidebar to get started!
        """)
        st.image("https://d1.awsstatic.com/migration/dms-diagram-2.3080e722300b0df26532881a2886f3762740a3f6.png", caption="AWS Database Migration Service Overview")
        st.markdown("---")
        main_cost_refresh_section() # Show cost refresh on home page

    elif page == "⚙️ Migration Configuration":
        show_migration_configuration_tab()

    elif page == "📊 Environment Setup":
        show_enhanced_environment_setup_with_vrops()

    elif page == "🚀 Analysis & Recommendations":
        show_analysis_section_fixed()

    elif page == "🌐 Network Transfer Analysis": # Network Tab display
        show_network_transfer_analysis()

    elif page == "📈 Data Visualizations":
        show_visualizations_tab()

    elif page == "⚠️ Risk Assessment":
        show_risk_assessment_tab()

    elif page == "🤖 AI-Powered Insights":
        show_ai_insights_tab()

    elif page == "🗓️ Detailed Timeline":
        show_timeline_analysis_tab()

    elif page == "📄 Reports & Export":
        show_reports_section()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Developed with Streamlit and Anthropic AI")

if __name__ == "__main__":
    main()