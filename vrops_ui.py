# vrops_ui.py
import streamlit as st
import pandas as pd
from vrops_processor import (
    create_vrops_sample_template,
    process_vrops_data,
    create_comprehensive_template,
    auto_detect_column_mappings,
    process_enhanced_data
)

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
                        st.write(f"Required Capacity: {cpu_analysis.get('required_capacity_percent', 0):.1f}%")
                    
                    with col2:
                        st.markdown("**Memory Analysis**")
                        memory_analysis = analysis.get('memory_analysis', {})
                        st.write(f"Max Usage: {memory_analysis.get('max_usage_percent', 0):.1f}%")
                        st.write(f"Avg Usage: {memory_analysis.get('avg_usage_percent', 0):.1f}%")
                        st.write(f"Allocated: {memory_analysis.get('allocated_gb', 0)} GB")
                    
                    with col3:
                        st.markdown("**Storage Analysis**")
                        storage_analysis = analysis.get('storage_analysis', {})
                        st.write(f"Max IOPS: {storage_analysis.get('max_iops', 0):,}")
                        st.write(f"Avg IOPS: {storage_analysis.get('avg_iops', 0):,}")
                        st.write(f"Storage Utilization: {storage_analysis.get('storage_utilization_percent', 0):.1f}%")
                    
                    # Instance recommendations
                    st.markdown("**🎯 AWS Instance Recommendations**")
                    recommendations = analysis.get('instance_recommendations', [])
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations[:3], 1):
                            st.markdown(f"{i}. **{rec['instance_type']}** - "
                                      f"CPU Efficiency: {rec['cpu_efficiency']:.1%}, "
                                      f"Memory Efficiency: {rec['memory_efficiency']:.1%}, "
                                      f"Fit Score: {rec['fit_score']:.1f}")
                    else:
                        st.write("No recommendations available")
    
    else:
        st.info("📊 vROps analysis not available. Use the enhanced environment setup with vROps metrics import to access detailed performance analysis.")

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
    
    # Use the same template and processing as before
    with st.expander("📋 Download Comprehensive Template", expanded=False):
        template_data = create_comprehensive_template()
        csv_data = template_data.to_csv(index=False)
        
        st.dataframe(template_data, use_container_width=True)
        
        st.download_button(
            label="📥 Download Performance Metrics Template",
            data=csv_data,
            file_name="performance_metrics_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload with enhanced processing
    uploaded_file = st.file_uploader(
        "Upload Performance Data",
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file with performance metrics"
    )
    
    if uploaded_file is not None:
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

def show_simple_configuration():
    """Show simple configuration for backward compatibility"""
    
    st.markdown("### 🔄 Simple Configuration (Legacy)")
    st.info("This is the simplified configuration mode. For better AWS sizing accuracy, consider using the vROps metrics import.")
    
    # Use the original simple interface
    show_manual_environment_setup()

def integrate_enhanced_environment_module():
    """Integration instructions for the enhanced environment module"""
    # Implementation would go here
    pass