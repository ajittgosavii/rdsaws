import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional

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
                'cpu_cores_allocated': {'required': True, 'description': 'Number of allocated vCPU cores'},
            },
            'memory_metrics': {
                'max_memory_usage_percent': {'required': True, 'description': 'Peak memory utilization'},
                'avg_memory_usage_percent': {'required': True, 'description': 'Average memory utilization'},
                'memory_allocated_gb': {'required': True, 'description': 'Total allocated memory to VM'},
                'memory_balloon_gb': {'required': False, 'description': 'Ballooned memory (overcommit indicator)'},
            },
            'storage_metrics': {
                'max_iops_total': {'required': True, 'description': 'Peak total IOPS (read + write)'},
                'avg_iops_total': {'required': True, 'description': 'Average total IOPS'},
                'max_disk_latency_ms': {'required': True, 'description': 'Peak disk latency in milliseconds'},
                'avg_disk_latency_ms': {'required': True, 'description': 'Average disk latency'},
                'storage_allocated_gb': {'required': True, 'description': 'Total allocated storage'},
                'storage_used_gb': {'required': True, 'description': 'Actually used storage'}
            },
            'network_metrics': {
                'max_network_throughput_mbps': {'required': True, 'description': 'Peak network throughput'},
                'avg_network_throughput_mbps': {'required': True, 'description': 'Average network throughput'},
            }
        }
    
    def _initialize_aws_instance_specs(self) -> Dict:
        """Initialize AWS instance specifications for accurate mapping"""
        return {
            # T3 instances (Burstable)
            'db.t3.micro': {'vcpu': 2, 'memory_gb': 1, 'network_gbps': 1.5},
            'db.t3.small': {'vcpu': 2, 'memory_gb': 2, 'network_gbps': 1.5},
            'db.t3.medium': {'vcpu': 2, 'memory_gb': 4, 'network_gbps': 1.5},
            'db.t3.large': {'vcpu': 2, 'memory_gb': 8, 'network_gbps': 1.5},
            'db.t3.xlarge': {'vcpu': 4, 'memory_gb': 16, 'network_gbps': 1.5},
            
            # R5 instances (Memory Optimized)
            'db.r5.large': {'vcpu': 2, 'memory_gb': 16, 'network_gbps': 10},
            'db.r5.xlarge': {'vcpu': 4, 'memory_gb': 32, 'network_gbps': 10},
            'db.r5.2xlarge': {'vcpu': 8, 'memory_gb': 64, 'network_gbps': 10},
            'db.r5.4xlarge': {'vcpu': 16, 'memory_gb': 128, 'network_gbps': 10},
            'db.r5.8xlarge': {'vcpu': 32, 'memory_gb': 256, 'network_gbps': 10},
        }
    
    def _initialize_performance_buffers(self) -> Dict:
        """Initialize performance buffers based on environment type"""
        return {
            'production': {
                'cpu_buffer': 0.3,      # 30% headroom
                'memory_buffer': 0.25,   # 25% headroom
                'iops_buffer': 0.4,      # 40% headroom
            },
            'staging': {
                'cpu_buffer': 0.2,
                'memory_buffer': 0.15,
                'iops_buffer': 0.3,
            },
            'testing': {
                'cpu_buffer': 0.15,
                'memory_buffer': 0.1,
                'iops_buffer': 0.2,
            },
            'development': {
                'cpu_buffer': 0.1,
                'memory_buffer': 0.05,
                'iops_buffer': 0.15,
            }
        }
    
    def analyze_metrics(self, environment_metrics: Dict) -> Dict:
        """Analyze vROps metrics for all environments"""
        
        analysis_results = {}
        
        for env_name, metrics in environment_metrics.items():
            analysis_results[env_name] = self._analyze_single_environment(env_name, metrics)
        
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
        
        # Generate AWS instance recommendations
        instance_recommendations = self._recommend_aws_instances(
            cpu_analysis, memory_analysis, storage_analysis, env_type
        )
        
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(metrics)
        
        return {
            'environment_type': env_type,
            'cpu_analysis': cpu_analysis,
            'memory_analysis': memory_analysis,
            'storage_analysis': storage_analysis,
            'instance_recommendations': instance_recommendations,
            'performance_scores': performance_scores,
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
        has_cpu_contention = cpu_ready > 5000  # 5+ seconds indicates contention
        
        # Recommend CPU scaling
        if required_cpu_percent > 80:
            cpu_scaling_recommendation = "Scale up CPU - current peak exceeds comfort zone"
        elif has_cpu_contention:
            cpu_scaling_recommendation = "Address CPU contention - consider optimization"
        else:
            cpu_scaling_recommendation = "CPU sizing appears adequate"
        
        return {
            'max_usage_percent': max_cpu,
            'avg_usage_percent': avg_cpu,
            'required_capacity_percent': required_cpu_percent,
            'current_cores': cpu_cores,
            'has_contention': has_cpu_contention,
            'scaling_recommendation': cpu_scaling_recommendation,
            'utilization_efficiency': avg_cpu / max_cpu if max_cpu > 0 else 0
        }
    
    def _analyze_memory_metrics(self, metrics: Dict, buffers: Dict) -> Dict:
        """Analyze memory metrics and requirements"""
        
        max_memory = metrics.get('max_memory_usage_percent', 0)
        avg_memory = metrics.get('avg_memory_usage_percent', 0)
        memory_allocated = metrics.get('memory_allocated_gb', 8)
        memory_balloon = metrics.get('memory_balloon_gb', 0)
        
        # Calculate required capacity with buffer
        required_memory_percent = max_memory * (1 + buffers['memory_buffer'])
        
        # Identify memory pressure indicators
        has_memory_pressure = memory_balloon > 0
        
        # Memory sizing recommendation
        if required_memory_percent > 85:
            memory_scaling_recommendation = "Increase memory allocation - peak usage too high"
        elif has_memory_pressure:
            memory_scaling_recommendation = "Address memory pressure - increase allocation"
        else:
            memory_scaling_recommendation = "Memory sizing appears adequate"
        
        return {
            'max_usage_percent': max_memory,
            'avg_usage_percent': avg_memory,
            'required_capacity_percent': required_memory_percent,
            'allocated_gb': memory_allocated,
            'has_pressure': has_memory_pressure,
            'scaling_recommendation': memory_scaling_recommendation,
            'utilization_efficiency': avg_memory / max_memory if max_memory > 0 else 0
        }
    
    def _analyze_storage_metrics(self, metrics: Dict, buffers: Dict) -> Dict:
        """Analyze storage performance metrics"""
        
        max_iops = metrics.get('max_iops_total', 0)
        avg_iops = metrics.get('avg_iops_total', 0)
        max_latency = metrics.get('max_disk_latency_ms', 0)
        avg_latency = metrics.get('avg_disk_latency_ms', 0)
        storage_used = metrics.get('storage_used_gb', 0)
        storage_allocated = metrics.get('storage_allocated_gb', 0)
        
        # Calculate required IOPS with buffer
        required_iops = max_iops * (1 + buffers['iops_buffer'])
        
        # Storage utilization
        storage_utilization = (storage_used / storage_allocated * 100) if storage_allocated > 0 else 0
        
        # Performance assessment
        if avg_latency > 20:
            latency_assessment = "High latency detected - consider faster storage"
        elif avg_latency > 10:
            latency_assessment = "Moderate latency - monitor closely"
        else:
            latency_assessment = "Latency within acceptable range"
        
        # Storage type recommendation
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
            'storage_utilization_percent': storage_utilization,
            'latency_assessment': latency_assessment,
            'recommended_storage_type': recommended_storage_type,
            'iops_efficiency': avg_iops / max_iops if max_iops > 0 else 0
        }
    
    def _recommend_aws_instances(self, cpu_analysis: Dict, memory_analysis: Dict, 
                               storage_analysis: Dict, env_type: str) -> List[Dict]:
        """Recommend AWS RDS instances based on analyzed metrics"""
        
        # Calculate requirements
        required_cpu_cores = max(2, int(cpu_analysis['required_capacity_percent'] / 100 * cpu_analysis['current_cores']))
        required_memory_gb = max(4, memory_analysis['allocated_gb'] * (memory_analysis['required_capacity_percent'] / 100))
        
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
                
                # Environment-specific adjustments
                if env_type == 'production' and 't3' in instance_type:
                    fit_score *= 0.9  # Slight penalty for burstable in production
                
                recommendations.append({
                    'instance_type': instance_type,
                    'vcpu': specs['vcpu'],
                    'memory_gb': specs['memory_gb'],
                    'fit_score': fit_score,
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
        
        if memory_usage < 30:
            memory_score = 70  # Under-utilized
        elif memory_usage < 80:
            memory_score = 100  # Optimal
        elif memory_usage < 90:
            memory_score = 75  # High but acceptable
        else:
            memory_score = 50  # Over-utilized
        
        if memory_balloon > 0:
            memory_score *= 0.8  # Penalty for ballooning
        
        scores['memory_health'] = min(100, memory_score)
        
        # Storage Health Score
        avg_latency = metrics.get('avg_disk_latency_ms', 0)
        
        if avg_latency < 5:
            storage_score = 100  # Excellent
        elif avg_latency < 10:
            storage_score = 90   # Good
        elif avg_latency < 20:
            storage_score = 70   # Acceptable
        else:
            storage_score = 40   # Poor
        
        scores['storage_health'] = min(100, storage_score)
        
        # Overall Health Score
        scores['overall_health'] = (scores['cpu_health'] + scores['memory_health'] + scores['storage_health']) / 3
        
        return scores


def show_enhanced_environment_setup_with_vrops():
    """Show enhanced environment setup with vROps integration"""
    
    st.markdown("## 📊 Enhanced Environment Setup with vROps Metrics")
    
    # Configuration method selection
    config_method = st.radio(
        "Choose configuration method:",
        [
            "📊 vROps Metrics Import", 
            "📝 Manual Configuration",
            "📁 Bulk Upload"
        ],
        horizontal=True
    )
    
    if config_method == "📊 vROps Metrics Import":
        show_vrops_import_interface()
    elif config_method == "📝 Manual Configuration":
        show_manual_configuration()
    else:
        show_bulk_upload_interface()

def show_vrops_import_interface():
    """Show vROps metrics import interface"""
    
    st.markdown("### 📊 vROps Metrics Import")
    
    # Initialize analyzer
    if 'vrops_analyzer' not in st.session_state:
        st.session_state.vrops_analyzer = VRopsMetricsAnalyzer()
    
    analyzer = st.session_state.vrops_analyzer
    
    # Sample metrics configuration
    st.markdown("#### 🔧 Configure vROps Metrics")
    
    with st.expander("Configure vROps Metrics for Analysis"):
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
    
    if st.button("🚀 Run vROps Analysis", type="primary"):
        analyzer = VRopsMetricsAnalyzer()
        st.session_state.vrops_analysis = analyzer.analyze_metrics(
            {"PROD": st.session_state.vrops_metrics}
        )
        st.success("✅ vROps analysis completed!")

def show_vrops_results_tab():
    """Show vROps analysis results in dashboard"""
    
    st.markdown("### 📊 vROps Metrics Analysis Results")
    
    if not hasattr(st.session_state, 'vrops_analysis') or not st.session_state.vrops_analysis:
        st.warning("⚠️ No vROps analysis results available. Please run the vROps analysis first.")
        st.info("👆 Go to 'Environment Setup' → 'vROps Metrics Import' to configure and analyze metrics")
        return
    
    results = st.session_state.vrops_analysis
    
    # Show results for each environment
    for env_name, analysis in results.items():
        st.markdown(f"#### 🏢 {env_name} Environment Analysis")
        
        # Performance scores overview
        scores = analysis['performance_scores']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_health = scores['cpu_health']
            color = 'normal' if cpu_health > 80 else 'inverse' if cpu_health > 60 else 'off'
            st.metric("CPU Health", f"{cpu_health:.0f}/100", delta="Good" if cpu_health > 80 else "Review")
        
        with col2:
            memory_health = scores['memory_health']
            st.metric("Memory Health", f"{memory_health:.0f}/100", delta="Good" if memory_health > 80 else "Review")
        
        with col3:
            storage_health = scores['storage_health']
            st.metric("Storage Health", f"{storage_health:.0f}/100", delta="Good" if storage_health > 80 else "Review")
        
        with col4:
            overall_health = scores['overall_health']
            st.metric("Overall Health", f"{overall_health:.0f}/100", delta="Good" if overall_health > 80 else "Review")
        
        # Detailed analysis in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💻 CPU Analysis", "🧠 Memory Analysis", "💾 Storage Analysis", "☁️ AWS Recommendations"])
        
        with tab1:
            cpu_analysis = analysis['cpu_analysis']
            st.markdown("**CPU Performance Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Usage", f"{cpu_analysis['max_usage_percent']:.1f}%")
                st.metric("Avg Usage", f"{cpu_analysis['avg_usage_percent']:.1f}%")
                st.metric("Current Cores", cpu_analysis['current_cores'])
            
            with col2:
                st.metric("Required Capacity", f"{cpu_analysis['required_capacity_percent']:.1f}%")
                st.metric("Utilization Efficiency", f"{cpu_analysis['utilization_efficiency']:.1%}")
                if cpu_analysis['has_contention']:
                    st.error("⚠️ CPU contention detected")
                else:
                    st.success("✅ No CPU contention")
            
            st.info(f"**Recommendation:** {cpu_analysis['scaling_recommendation']}")
        
        with tab2:
            memory_analysis = analysis['memory_analysis']
            st.markdown("**Memory Performance Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Usage", f"{memory_analysis['max_usage_percent']:.1f}%")
                st.metric("Avg Usage", f"{memory_analysis['avg_usage_percent']:.1f}%")
                st.metric("Allocated", f"{memory_analysis['allocated_gb']:.1f} GB")
            
            with col2:
                st.metric("Required Capacity", f"{memory_analysis['required_capacity_percent']:.1f}%")
                st.metric("Utilization Efficiency", f"{memory_analysis['utilization_efficiency']:.1%}")
                if memory_analysis['has_pressure']:
                    st.error("⚠️ Memory pressure detected")
                else:
                    st.success("✅ No memory pressure")
            
            st.info(f"**Recommendation:** {memory_analysis['scaling_recommendation']}")
        
        with tab3:
            storage_analysis = analysis['storage_analysis']
            st.markdown("**Storage Performance Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max IOPS", f"{storage_analysis['max_iops']:,}")
                st.metric("Avg IOPS", f"{storage_analysis['avg_iops']:,}")
                st.metric("Required IOPS", f"{storage_analysis['required_iops']:,}")
            
            with col2:
                st.metric("Max Latency", f"{storage_analysis['max_latency_ms']:.1f} ms")
                st.metric("Avg Latency", f"{storage_analysis['avg_latency_ms']:.1f} ms")
                st.metric("Storage Utilization", f"{storage_analysis['storage_utilization_percent']:.1f}%")
            
            st.info(f"**Latency Assessment:** {storage_analysis['latency_assessment']}")
            st.info(f"**Recommended Storage:** {storage_analysis['recommended_storage_type']}")
        
        with tab4:
            recommendations = analysis['instance_recommendations']
            st.markdown("**AWS RDS Instance Recommendations**")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Option {i}: {rec['instance_type']} (Fit Score: {rec['fit_score']:.1f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("vCPUs", rec['vcpu'])
                        st.metric("Memory", f"{rec['memory_gb']} GB")
                    
                    with col2:
                        st.metric("Fit Score", f"{rec['fit_score']:.1f}/100")
                        
                    st.markdown(f"**Reasoning:** {rec['recommendation_reason']}")

def show_manual_configuration():
    """Show manual configuration interface"""
    st.markdown("### 📝 Manual Environment Configuration")
    st.info("Configure environments manually with custom specifications")
    
    # Basic manual configuration form
    with st.form("manual_config"):
        env_name = st.text_input("Environment Name", value="Production")
        cpu_cores = st.number_input("CPU Cores", min_value=1, value=8)
        ram_gb = st.number_input("RAM (GB)", min_value=1, value=32)
        storage_gb = st.number_input("Storage (GB)", min_value=10, value=500)
        
        submitted = st.form_submit_button("Save Configuration")
        
        if submitted:
            st.session_state.environment_specs = {
                env_name: {
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'daily_usage_hours': 24,
                    'peak_connections': 100
                }
            }
            st.success("✅ Configuration saved!")

def show_bulk_upload_interface():
    """Show bulk upload interface"""
    st.markdown("### 📁 Bulk Configuration Upload")
    st.info("Upload CSV or Excel file with environment specifications")
    
    uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} rows")
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")