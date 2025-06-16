import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import asyncio
import traceback
import json
import base64
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional
from io import BytesIO
import tempfile
import firebase_admin
from firebase_admin import credentials, auth, firestore
from data_transfer_calculator import DataTransferCalculator, TransferMethod, TransferMethodResult
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.io as pio
import kaleido  # Required for plotly image export


# Authentication imports
import bcrypt
import jwt

# Import our enhanced modules
try:
    from rds_sizing import EnhancedRDSSizingCalculator, MigrationType, WorkloadCharacteristics
    from aws_pricing import EnhancedAWSPricingAPI
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="Enterprise AWS RDS Migration & Sizing Tool",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ================================
# AUTHENTICATION FUNCTIONS
# ================================

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a stored password against one provided by user"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def get_users_from_secrets():
    """Get users from Streamlit secrets"""
    try:
        if "auth" in st.secrets and "users" in st.secrets["auth"]:
            return dict(st.secrets["auth"]["users"])
        return {}
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {}

def authenticate_user(email: str, password: str) -> dict:
    """Authenticate user credentials"""
    users = get_users_from_secrets()
    
    for username, user_data in users.items():
        user_dict = dict(user_data)
        if user_dict.get('email', '').lower() == email.lower():
            if verify_password(password, user_dict['password']):
                return {
                    'username': username,
                    'email': user_dict['email'],
                    'name': user_dict['name'],
                    'role': user_dict['role'],
                    'authenticated': True
                }
    return {'authenticated': False}

def create_session_token(user_data: dict) -> str:
    """Create a JWT session token"""
    try:
        config = dict(st.secrets["auth"]["config"])
        payload = {
            'username': user_data['username'],
            'email': user_data['email'],
            'name': user_data['name'],
            'role': user_data['role'],
            'exp': datetime.utcnow() + timedelta(days=int(config.get('cookie_expiry_days', 7)))
        }
        return jwt.encode(payload, config['cookie_key'], algorithm='HS256')
    except Exception as e:
        st.error(f"Error creating session token: {e}")
        return None

def verify_session_token(token: str) -> dict:
    """Verify and decode session token"""
    try:
        config = dict(st.secrets["auth"]["config"])
        payload = jwt.decode(token, config['cookie_key'], algorithms=['HS256'])
        return {
            'username': payload['username'],
            'email': payload['email'],
            'name': payload['name'],
            'role': payload['role'],
            'authenticated': True
        }
    except jwt.ExpiredSignatureError:
        return {'authenticated': False, 'error': 'Session expired'}
    except jwt.InvalidTokenError:
        return {'authenticated': False, 'error': 'Invalid session'}

def show_login_form():
    """Display the login form"""
    st.markdown("""
    <div style="max-width: 400px; margin: 50px auto; padding: 30px; 
                border: 1px solid #ddd; border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 Enterprise RDS Migration Tool - Login")
    st.markdown("Please enter your credentials to access the system.")
    
    with st.form("login_form"):
        email = st.text_input("📧 Email", placeholder="user@yourcompany.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("🚀 Login", use_container_width=True)
        with col2:
            if st.form_submit_button("👥 Show Test Users", use_container_width=True):
                st.session_state.show_test_users = True
    
    if login_button:
        if email and password:
            with st.spinner("Authenticating..."):
                user_data = authenticate_user(email, password)
                
                if user_data['authenticated']:
                    token = create_session_token(user_data)
                    if token:
                        st.session_state.user_authenticated = True
                        st.session_state.user_data = user_data
                        st.session_state.session_token = token
                        st.session_state.user_id = user_data['username']
                        st.session_state.is_logged_in = True
                        
                        st.success(f"✅ Welcome, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Error creating session. Please try again.")
                else:
                    st.error("❌ Invalid email or password. Please try again.")
        else:
            st.error("❌ Please enter both email and password.")
    
    if st.session_state.get('show_test_users', False):
        st.markdown("---")
        st.markdown("#### 👥 Test Users (Development Only)")
        users = get_users_from_secrets()
        for username, user_data in users.items():
            user_dict = dict(user_data)
            st.markdown(f"**{user_dict['name']}** ({user_dict['role']})")
            st.code(f"Email: {user_dict['email']}")
        st.markdown("*Passwords are set in secrets.toml*")
    
    st.markdown("</div>", unsafe_allow_html=True)

def logout_user():
    """Logout the current user"""
    auth_keys = [
        'user_authenticated', 'user_data', 'session_token', 
        'user_id', 'user_email', 'is_logged_in'
    ]
    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("👋 You have been logged out successfully!")
    st.rerun()

def check_authentication():
    """Check if user is authenticated and handle session"""
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    
    if not st.session_state.user_authenticated and 'session_token' in st.session_state:
        token_data = verify_session_token(st.session_state.session_token)
        if token_data['authenticated']:
            st.session_state.user_authenticated = True
            st.session_state.user_data = token_data
            st.session_state.user_id = token_data['username']
            st.session_state.is_logged_in = True
        else:
            if 'session_token' in st.session_state:
                del st.session_state['session_token']
    
    return st.session_state.user_authenticated

def show_user_info():
    """Display current user info in sidebar"""
    if st.session_state.get('user_authenticated', False) and 'user_data' in st.session_state:
        user_data = st.session_state.user_data
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Current User")
        st.sidebar.markdown(f"**Name:** {user_data['name']}")
        st.sidebar.markdown(f"**Email:** {user_data['email']}")
        st.sidebar.markdown(f"**Role:** {user_data['role'].title()}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()

# ================================
# FIREBASE FUNCTIONS
# ================================

@st.cache_resource(ttl=3600)
def initialize_firebase():
    """Initializes the Firebase app and authenticates the user."""
    try:
        if firebase_admin._apps:
            st.info("Firebase already initialized")
            app = firebase_admin.get_app()
            return app, auth, firestore.client()
        
        if "connections" not in st.secrets or "firebase" not in st.secrets["connections"]:
            st.error("Firebase configuration not found in Streamlit secrets.")
            return None, None, None

        firebase_config_dict = dict(st.secrets["connections"]["firebase"])
        
        required_fields = ['project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if not firebase_config_dict.get(field)]
        
        if missing_fields:
            st.error(f"Missing required Firebase fields: {missing_fields}")
            return None, None, None

        firebase_config_dict['type'] = 'service_account'
        
        private_key = firebase_config_dict.get('private_key', '').strip()
        
        if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
            st.error("Private key missing BEGIN header")
            return None, None, None
        
        if not private_key.endswith('-----END PRIVATE KEY-----'):
            st.error("Private key missing END footer")
            return None, None, None

        cred = credentials.Certificate(firebase_config_dict)
        firebase_app = firebase_admin.initialize_app(
            cred, 
            options={'projectId': firebase_config_dict['project_id']}
        )
        
        db_client = firestore.client(firebase_app)
        
        st.success("🎉 Firebase Admin SDK initialized successfully!")
        
        return firebase_app, auth, db_client
        
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# ================================
# UTILITY FUNCTIONS
# ================================

def safe_get(dictionary, key, default=0):
    """Safely get a value from a dictionary with a default fallback"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default

def safe_get_str(dictionary, key, default="N/A"):
    """Safely get a string value from a dictionary with a default fallback"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default

# ================================
# ENHANCED REPORT GENERATOR
# ================================

# Enhanced PDF Report Generator with Charts and Detailed Analysis
# This code replaces the existing EnhancedReportGenerator class in your streamlit_app.py

import io
import base64
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.colors import HexColor

# COMPLETE FIX: Replace the entire setup_custom_styles method
# Find this method in your ComprehensiveReportGenerator class (around line 332) and replace it:
# ================================
# FIXED COMPREHENSIVE REPORT GENERATOR CLASS
# ================================

# ================================
# IMPROVED COMPREHENSIVE REPORT GENERATOR CLASS
# ================================

import io
import base64
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

class ImprovedReportGenerator:
    """Enhanced PDF Report Generator with Better Formatting and Layout"""
    
    def __init__(self):
        """Initialize the improved report generator"""
        self.styles = getSampleStyleSheet()
        self.chart_width = 5.5*inch
        self.chart_height = 3.5*inch
        self.setup_custom_styles()
        
        # Debug: Verify styles are created
        self._verify_styles()
    
    def _verify_styles(self):
        """Verify that all required styles are available"""
        required_styles = [
            'ReportTitle', 'SectionHeader', 'SubsectionHeader', 
            'BodyText', 'KeyMetric', 'TableHeader', 'BulletPoint'
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
                    name='ReportTitle',
                    parent=self.styles['Title'],
                    fontSize=24,
                    fontName='Helvetica-Bold',
                    alignment=TA_CENTER
                ))
            elif style_name == 'SectionHeader':
                self.styles.add(ParagraphStyle(
                    name='SectionHeader',
                    parent=self.styles['Heading1'],
                    fontSize=16,
                    fontName='Helvetica-Bold',
                    spaceBefore=20,
                    spaceAfter=12
                ))
            else:
                # Generic fallback
                self.styles.add(ParagraphStyle(
                    name=style_name,
                    parent=base_style,
                    fontSize=12,
                    fontName='Helvetica'
                ))
        except Exception as e:
            print(f"Error creating fallback style {style_name}: {e}")
    
    def setup_custom_styles(self):
        """Setup improved custom styles for the report"""
        try:
            # Report Title Style
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceBefore=0,
                spaceAfter=20,
                textColor=colors.darkblue,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
            
            # Section Header Style
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5
            ))
            
            # Subsection Header Style
            self.styles.add(ParagraphStyle(
                name='SubsectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=15,
                spaceAfter=8,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            ))
            
            # Body Text Style
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=6,
                spaceAfter=6,
                textColor=colors.black,
                fontName='Helvetica',
                leading=14
            ))
            
            # Key Metric Style
            self.styles.add(ParagraphStyle(
                name='KeyMetric',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.darkred,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            ))
            
            # Table Header Style
            self.styles.add(ParagraphStyle(
                name='TableHeader',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.white,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            ))
            
            # Bullet Point Style
            self.styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=4,
                spaceAfter=4,
                leftIndent=20,
                bulletIndent=10,
                fontName='Helvetica'
            ))
            
        except Exception as e:
            print(f"Error setting up custom styles: {e}")
            self._create_basic_styles()
    
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
                        name=style_name,
                        parent=self.styles[parent],
                        fontSize=size,
                        fontName='Helvetica'
                    ))
            except Exception as e:
                print(f"Error creating basic style {style_name}: {e}")
    
    def safe_get_style(self, style_name, fallback='Normal'):
        """Safely get a style, falling back to Normal if not found"""
        try:
            return self.styles[style_name]
        except KeyError:
            print(f"Style '{style_name}' not found, using '{fallback}'")
            return self.styles[fallback]
    
    def create_improved_cost_chart(self, analysis_results, analysis_mode):
        """Create an improved cost breakdown chart with better formatting"""
        try:
            if analysis_mode == 'single':
                valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
                if not valid_results:
                    return None
                
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                
                # Extract cost components with better logic
                cost_data = {}
                
                if 'writer' in prod_result:
                    # Aurora/RDS Cluster setup
                    writer_cost = prod_result.get('cost_breakdown', {}).get('writer_monthly', 0)
                    readers_cost = prod_result.get('cost_breakdown', {}).get('readers_monthly', 0)
                    storage_cost = prod_result.get('cost_breakdown', {}).get('storage_monthly', 0)
                    backup_cost = prod_result.get('cost_breakdown', {}).get('backup_monthly', 0)
                    transfer_cost = prod_result.get('cost_breakdown', {}).get('transfer_monthly', 50)
                    
                    if writer_cost > 0:
                        cost_data['Writer Instance'] = writer_cost
                    if readers_cost > 0:
                        cost_data['Reader Instances'] = readers_cost
                    if storage_cost > 0:
                        cost_data['Storage'] = storage_cost
                    if backup_cost > 0:
                        cost_data['Backup'] = backup_cost
                    if transfer_cost > 0:
                        cost_data['Data Transfer'] = transfer_cost
                else:
                    # Standard RDS setup
                    cost_breakdown = prod_result.get('cost_breakdown', {})
                    instance_cost = cost_breakdown.get('instance_monthly', prod_result.get('instance_cost', 0))
                    storage_cost = cost_breakdown.get('storage_monthly', prod_result.get('storage_cost', 0))
                    backup_cost = cost_breakdown.get('backup_monthly', storage_cost * 0.1)
                    transfer_cost = cost_breakdown.get('transfer_monthly', 50)
                    
                    if instance_cost > 0:
                        cost_data['Compute Instance'] = instance_cost
                    if storage_cost > 0:
                        cost_data['Storage'] = storage_cost
                    if backup_cost > 0:
                        cost_data['Backup'] = backup_cost
                    if transfer_cost > 0:
                        cost_data['Data Transfer'] = transfer_cost
                
                # Filter out zero values and ensure we have data
                cost_data = {k: v for k, v in cost_data.items() if v > 0}
                
                if not cost_data:
                    # Fallback data if no breakdown available
                    total_cost = prod_result.get('total_cost', 0)
                    cost_data = {
                        'Database Instance': total_cost * 0.7,
                        'Storage & Backup': total_cost * 0.25,
                        'Data Transfer': total_cost * 0.05
                    }
                
                # Create improved pie chart
                colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(cost_data.keys()),
                    values=list(cost_data.values()),
                    hole=.4,
                    textinfo='label+percent',
                    textposition='outside',
                    texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
                    textfont={"size": 10},
                    marker=dict(colors=colors_list[:len(cost_data)], line=dict(color='#FFFFFF', width=2))
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Monthly Cost Breakdown - ${sum(cost_data.values()):,.0f}',
                        'x': 0.5,
                        'font': {'size': 14, 'color': 'darkblue'},
                        'y': 0.95
                    },
                    font=dict(size=10),
                    showlegend=True,
                    legend=dict(
                        orientation="v", 
                        yanchor="middle", 
                        y=0.5, 
                        xanchor="left", 
                        x=1.05,
                        font=dict(size=9)
                    ),
                    margin=dict(t=50, b=50, l=50, r=100),
                    width=600,
                    height=400
                )
                
                return self.create_plotly_chart_image(fig, width=600, height=400)
            
            else:  # Bulk analysis
                server_costs = []
                server_names = []
                
                for server_name, server_results in analysis_results.items():
                    if 'error' not in server_results:
                        result = server_results.get('PROD', list(server_results.values())[0])
                        if 'error' not in result:
                            cost = result.get('total_cost', 0)
                            if cost > 0:
                                server_costs.append(cost)
                                # Truncate long server names for better display
                                display_name = server_name[:15] + '...' if len(server_name) > 15 else server_name
                                server_names.append(display_name)
                
                if not server_costs:
                    return None
                
                # Create improved bar chart for bulk analysis
                fig = go.Figure(data=[go.Bar(
                    x=server_names,
                    y=server_costs,
                    marker_color='lightblue',
                    text=[f'${cost:,.0f}' for cost in server_costs],
                    textposition='outside',
                    textfont=dict(size=9)
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Monthly Cost by Server - Total: ${sum(server_costs):,.0f}',
                        'x': 0.5,
                        'font': {'size': 14, 'color': 'darkblue'}
                    },
                    xaxis_title='Server Name',
                    yaxis_title='Monthly Cost ($)',
                    font=dict(size=10),
                    xaxis={'tickangle': 45, 'tickfont': {'size': 8}},
                    margin=dict(t=50, b=80, l=70, r=50),
                    width=700,
                    height=400
                )
                
                return self.create_plotly_chart_image(fig, width=700, height=400)
                
        except Exception as e:
            print(f"Error creating cost chart: {e}")
            return None
    
    def create_plotly_chart_image(self, fig, width=600, height=400):
        """Convert Plotly figure to image for PDF inclusion with improved quality"""
        try:
            # Convert plotly figure to image bytes with high quality
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file.flush()
                
                # Create ReportLab Image object with appropriate sizing
                img = Image(tmp_file.name, width=self.chart_width, height=self.chart_height)
                return img
        except Exception as e:
            print(f"Error creating chart image: {e}")
            return None
    
    def create_improved_technical_table(self, analysis_results, server_specs, env_name):
        """Create an improved technical specifications table"""
        try:
            result = analysis_results.get(env_name, {})
            if 'error' in result:
                return None
            
            # Table data with better formatting
            table_data = [
                ['Component', 'Current (On-Premises)', 'Recommended (AWS)', 'Improvement']
            ]
            
            current_cpu = server_specs.get('cores', 0) if server_specs else 0
            current_ram = server_specs.get('ram', 0) if server_specs else 0
            current_storage = server_specs.get('storage', 0) if server_specs else 0
            
            if 'writer' in result:
                # Aurora/RDS Cluster configuration
                writer = result['writer']
                recommended_cpu = writer.get('actual_vCPUs', 0)
                recommended_ram = writer.get('actual_RAM_GB', 0)
                recommended_storage = result.get('storage_GB', 0)
                
                table_data.extend([
                    ['Writer Instance', 
                     f"{current_cpu} cores", 
                     f"{writer.get('instance_type', 'N/A')}", 
                     f"{(recommended_cpu/max(current_cpu,1)):.1f}x CPU"],
                    ['Writer Memory', 
                     f"{current_ram} GB", 
                     f"{recommended_ram} GB", 
                     f"{(recommended_ram/max(current_ram,1)):.1f}x RAM"],
                    ['Storage Capacity', 
                     f"{current_storage} GB", 
                     f"{recommended_storage} GB", 
                     f"{(recommended_storage/max(current_storage,1)):.1f}x"],
                ])
                
                if result.get('readers'):
                    table_data.append([
                        'Read Replicas', 
                        'None', 
                        f"{len(result['readers'])} instances", 
                        'New capability'
                    ])
            
            else:
                # Standard RDS configuration
                recommended_cpu = result.get('actual_vCPUs', 0)
                recommended_ram = result.get('actual_RAM_GB', 0)
                recommended_storage = result.get('storage_GB', 0)
                
                table_data.extend([
                    ['Instance Type', 
                     'Physical Server', 
                     result.get('instance_type', 'N/A'), 
                     'Cloud Native'],
                    ['CPU Cores', 
                     f"{current_cpu} cores", 
                     f"{recommended_cpu} vCPUs", 
                     f"{(recommended_cpu/max(current_cpu,1)):.1f}x"],
                    ['Memory', 
                     f"{current_ram} GB", 
                     f"{recommended_ram} GB", 
                     f"{(recommended_ram/max(current_ram,1)):.1f}x"],
                    ['Storage', 
                     f"{current_storage} GB", 
                     f"{recommended_storage} GB", 
                     f"{(recommended_storage/max(current_storage,1)):.1f}x"]
                ])
            
            # Create table with improved styling
            col_widths = [1.8*inch, 1.6*inch, 1.8*inch, 1.2*inch]
            tech_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            
            # Enhanced table styling
            tech_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                
                # Data rows styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
                
                # Left align the first column
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                
                # Padding for better readability
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ]))
            
            return tech_table
            
        except Exception as e:
            print(f"Error creating technical table for {env_name}: {e}")
            return None
    
    def format_ai_insights_text(self, ai_text, max_line_length=80):
        """Format AI insights text for better PDF display"""
        if not ai_text:
            return "No AI insights available."
        
        # Clean up the text
        cleaned_text = ai_text.replace('...', '')
        
        # Split into sentences and paragraphs
        sentences = cleaned_text.split('. ')
        formatted_paragraphs = []
        current_paragraph = ""
        
        for sentence in sentences:
            if len(current_paragraph) + len(sentence) < max_line_length:
                current_paragraph += sentence + ". "
            else:
                if current_paragraph:
                    formatted_paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence + ". "
        
        if current_paragraph:
            formatted_paragraphs.append(current_paragraph.strip())
        
        return formatted_paragraphs[:5]  # Limit to 5 paragraphs for PDF
    
    def generate_improved_pdf_report(self, analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
        """Generate improved PDF report with better formatting and layout"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch,
            topMargin=1*inch, 
            bottomMargin=1*inch
        )
        
        story = []
        
        try:
            # Title Page
            story.extend(self._create_title_page(analysis_mode, analysis_results))
            story.append(PageBreak())
            
            # Executive Summary with Chart
            story.extend(self._create_executive_summary(analysis_results, analysis_mode, ai_insights))
            story.append(PageBreak())
            
            # Technical Analysis
            story.extend(self._create_technical_analysis(analysis_results, analysis_mode, server_specs))
            story.append(PageBreak())
            
            # Financial Analysis
            story.extend(self._create_financial_analysis(analysis_results, analysis_mode))
            story.append(PageBreak())
            
            # Migration Strategy
            story.extend(self._create_migration_strategy())
            story.append(PageBreak())
            
            # AI Insights (if available)
            if ai_insights:
                story.extend(self._create_ai_insights_section(ai_insights))
                story.append(PageBreak())
            
            # Implementation Roadmap
            story.extend(self._create_implementation_roadmap())
            
            # Build the PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error building improved PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_title_page(self, analysis_mode, analysis_results):
        """Create an improved title page"""
        story = []
        
        # Main title
        title_style = self.safe_get_style('ReportTitle', 'Title')
        story.append(Paragraph("AWS RDS Migration & Sizing", title_style))
        story.append(Paragraph("Comprehensive Analysis Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis type
        analysis_type = "Single Server Analysis" if analysis_mode == 'single' else "Bulk Server Analysis"
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph(analysis_type, section_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                summary_items = [
                    f"Monthly Cloud Cost: ${monthly_cost:,.2f}",
                    f"Annual Investment: ${monthly_cost * 12:,.2f}",
                    "Migration Type: Heterogeneous Database Migration",
                    "Estimated Timeline: 12-16 weeks",
                    "Risk Level: Medium (Manageable with proper planning)"
                ]
        else:
            total_servers = len(analysis_results)
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            summary_items = [
                f"Total Servers Analyzed: {total_servers}",
                f"Successful Analyses: {successful_servers}",
                f"Total Monthly Cost: ${total_monthly_cost:,.2f}",
                f"Total Annual Investment: ${total_monthly_cost * 12:,.2f}",
                f"Average Cost per Server: ${total_monthly_cost/max(successful_servers,1):,.2f}/month"
            ]
        
        # Summary box
        summary_style = self.safe_get_style('KeyMetric', 'Normal')
        story.append(Paragraph("<b>Executive Summary</b>", summary_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        for item in summary_items:
            story.append(Paragraph(f"• {item}", bullet_style))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        body_style = self.safe_get_style('BodyText', 'Normal')
        generation_time = datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(f"<b>Generated:</b> {generation_time}", body_style))
        story.append(Paragraph("<b>Report Type:</b> Comprehensive Technical & Financial Analysis", body_style))
        story.append(Paragraph("<b>Prepared for:</b> Enterprise Cloud Migration Team", body_style))
        
        return story
    
    def _create_executive_summary(self, analysis_results, analysis_mode, ai_insights):
        """Create improved executive summary with chart"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Executive Summary & Key Findings", section_style))
        story.append(Spacer(1, 12))
        
        # Cost breakdown chart
        cost_chart = self.create_improved_cost_chart(analysis_results, analysis_mode)
        if cost_chart:
            story.append(cost_chart)
            story.append(Spacer(1, 20))
        
        # Key findings
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Key Findings & Recommendations:", subsection_style))
        
        body_style = self.safe_get_style('BodyText', 'Normal')
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                findings = [
                    f"<b>Cost Efficiency:</b> Projected monthly operational cost of ${monthly_cost:,.2f} provides significant operational benefits over traditional on-premises infrastructure",
                    "<b>Performance Scaling:</b> AWS RDS configuration will provide improved performance and automatic scaling capabilities",
                    "<b>Risk Mitigation:</b> Multi-AZ deployment ensures 99.95% uptime SLA with automatic failover",
                    "<b>Operational Benefits:</b> Reduced maintenance overhead with managed database services"
                ]
        else:
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            findings = [
                f"<b>Scale Efficiency:</b> {successful_servers} servers successfully analyzed with total monthly cost of ${total_monthly_cost:,.2f}",
                "<b>Migration Approach:</b> Phased migration recommended with 3-5 servers per wave",
                "<b>Cost Optimization:</b> Bulk Reserved Instance purchases can reduce costs by 30-40%",
                "<b>Timeline:</b> Estimated 6-9 months for complete bulk migration with parallel streams"
            ]
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_technical_analysis(self, analysis_results, analysis_mode, server_specs):
        """Create improved technical analysis section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Technical Analysis & Performance Assessment", section_style))
        story.append(Spacer(1, 15))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            
            for env, result in valid_results.items():
                subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
                story.append(Paragraph(f"{env} Environment Configuration", subsection_style))
                story.append(Spacer(1, 10))
                
                tech_table = self.create_improved_technical_table(analysis_results, server_specs, env)
                if tech_table:
                    story.append(tech_table)
                    story.append(Spacer(1, 20))
        
        else:
            # Bulk analysis summary table
            story.extend(self._create_bulk_technical_summary(analysis_results, server_specs))
        
        return story
    
    def _create_bulk_technical_summary(self, analysis_results, server_specs):
        """Create bulk technical analysis summary"""
        story = []
        
        # Aggregate statistics
        total_current_cpu = sum(server.get('cpu_cores', 0) for server in server_specs) if server_specs else 0
        total_current_ram = sum(server.get('ram_gb', 0) for server in server_specs) if server_specs else 0
        total_current_storage = sum(server.get('storage_gb', 0) for server in server_specs) if server_specs else 0
        
        total_recommended_cpu = 0
        total_recommended_ram = 0
        total_recommended_storage = 0
        successful_count = 0
        
        for server_results in analysis_results.values():
            if 'error' not in server_results:
                result = server_results.get('PROD', list(server_results.values())[0])
                if 'error' not in result:
                    successful_count += 1
                    if 'writer' in result:
                        writer = result['writer']
                        total_recommended_cpu += writer.get('actual_vCPUs', 0)
                        total_recommended_ram += writer.get('actual_RAM_GB', 0)
                    else:
                        total_recommended_cpu += result.get('actual_vCPUs', 0)
                        total_recommended_ram += result.get('actual_RAM_GB', 0)
                    total_recommended_storage += result.get('storage_GB', 0)
        
        # Create aggregate comparison table
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Aggregate Resource Comparison", subsection_style))
        story.append(Spacer(1, 10))
        
        aggregate_data = [
            ['Resource Type', 'Current Total', 'Recommended Total', 'Efficiency Gain'],
            ['Total vCPUs', f"{total_current_cpu}", f"{total_recommended_cpu}", f"{((total_recommended_cpu/max(total_current_cpu,1))-1)*100:+.1f}%"],
            ['Total RAM (GB)', f"{total_current_ram:,}", f"{total_recommended_ram:,}", f"{((total_recommended_ram/max(total_current_ram,1))-1)*100:+.1f}%"],
            ['Total Storage (GB)', f"{total_current_storage:,}", f"{total_recommended_storage:,}", f"{((total_recommended_storage/max(total_current_storage,1))-1)*100:+.1f}%"],
            ['Server Count', f"{len(server_specs) if server_specs else 0}", f"{successful_count}", f"{(successful_count/max(len(server_specs) if server_specs else 1,1))*100:.0f}% success"]
        ]
        
        aggregate_table = Table(aggregate_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        aggregate_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(aggregate_table)
        
        return story
    
    def _create_financial_analysis(self, analysis_results, analysis_mode):
        """Create improved financial analysis section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Financial Analysis & ROI Projection", section_style))
        story.append(Spacer(1, 15))
        
        # Calculate costs
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                financial_data = [
                    ['Cost Component', 'Monthly', 'Annual', '3-Year Total'],
                    ['AWS Infrastructure', f'${monthly_cost:,.2f}', f'${monthly_cost*12:,.2f}', f'${monthly_cost*36:,.2f}'],
                    ['Migration Costs (One-time)', '-', f'${monthly_cost*2:,.2f}', f'${monthly_cost*2:,.2f}'],
                    ['Training & Support', '-', f'${monthly_cost*0.5:,.2f}', f'${monthly_cost*1.5:,.2f}'],
                    ['Total Investment', f'${monthly_cost:,.2f}', f'${monthly_cost*14.5:,.2f}', f'${monthly_cost*39.5:,.2f}']
                ]
        else:
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            financial_data = [
                ['Cost Component', 'Monthly', 'Annual', '3-Year Total'],
                ['AWS Infrastructure', f'${total_monthly_cost:,.2f}', f'${total_monthly_cost*12:,.2f}', f'${total_monthly_cost*36:,.2f}'],
                ['Migration Costs (One-time)', '-', f'${total_monthly_cost*3:,.2f}', f'${total_monthly_cost*3:,.2f}'],
                ['Training & Support', '-', f'${total_monthly_cost*1:,.2f}', f'${total_monthly_cost*3:,.2f}'],
                ['Total Investment', f'${total_monthly_cost:,.2f}', f'${total_monthly_cost*16:,.2f}', f'${total_monthly_cost*42:,.2f}']
            ]
        
        # Create financial table
        financial_table = Table(financial_data, colWidths=[2.2*inch, 1.4*inch, 1.4*inch, 1.4*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.mistyrose),
            ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(financial_table)
        
        return story
    
    def _create_migration_strategy(self):
        """Create migration strategy section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Migration Strategy & Implementation Timeline", section_style))
        story.append(Spacer(1, 15))
        
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Migration Strategy Overview:", subsection_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        migration_phases = [
            "<b>Assessment Phase:</b> Complete discovery and compatibility analysis",
            "<b>Planning Phase:</b> Detailed migration planning and resource allocation",
            "<b>Execution Phase:</b> Phased migration with minimal downtime",
            "<b>Validation Phase:</b> Comprehensive testing and performance validation",
            "<b>Optimization Phase:</b> Post-migration tuning and optimization"
        ]
        
        for phase in migration_phases:
            story.append(Paragraph(f"• {phase}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_ai_insights_section(self, ai_insights):
        """Create improved AI insights section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("AI-Powered Insights & Recommendations", section_style))
        story.append(Spacer(1, 15))
        
        # AI analysis
        ai_text = ai_insights.get("ai_analysis", "")
        if ai_text:
            formatted_paragraphs = self.format_ai_insights_text(ai_text)
            
            body_style = self.safe_get_style('BodyText', 'Normal')
            for paragraph in formatted_paragraphs:
                story.append(Paragraph(paragraph, body_style))
                story.append(Spacer(1, 10))
        else:
            # Default AI insights
            bullet_style = self.safe_get_style('BulletPoint', 'Normal')
            default_insights = [
                "Migration analysis indicates favorable conditions for AWS RDS migration",
                "Recommended architecture provides optimal balance of performance and cost",
                "Risk assessment suggests manageable migration with proper planning",
                "Cost optimization opportunities identified for long-term efficiency"
            ]
            
            for insight in default_insights:
                story.append(Paragraph(f"• {insight}", bullet_style))
                story.append(Spacer(1, 6))
        
        return story
    
    def _create_implementation_roadmap(self):
        """Create implementation roadmap section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Implementation Roadmap & Next Steps", section_style))
        story.append(Spacer(1, 15))
        
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Implementation Roadmap:", subsection_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        roadmap_phases = [
            "<b>Phase 1 - Planning:</b> Finalize migration strategy and resource allocation",
            "<b>Phase 2 - Preparation:</b> Set up AWS environment and migration tools",
            "<b>Phase 3 - Migration:</b> Execute data and application migration",
            "<b>Phase 4 - Testing:</b> Comprehensive validation and performance testing",
            "<b>Phase 5 - Go-Live:</b> Production cutover and monitoring",
            "<b>Phase 6 - Optimization:</b> Post-migration tuning and optimization"
        ]
        
        for phase in roadmap_phases:
            story.append(Paragraph(f"• {phase}", bullet_style))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 20))
        
        # Success criteria
        story.append(Paragraph("Success Criteria:", subsection_style))
        
        success_criteria = [
            "Zero data loss during migration",
            "Minimal downtime during cutover",
            "Performance meets or exceeds baseline",
            "Cost targets achieved within budget",
            "Team readiness and knowledge transfer complete"
        ]
        
        for criteria in success_criteria:
            story.append(Paragraph(f"• {criteria}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story


# Helper function to use the improved generator
def generate_improved_pdf_report(analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
    """Helper function with improved formatting and error handling"""
    try:
        print("Creating ImprovedReportGenerator...")
        improved_generator = ImprovedReportGenerator()
        
        print("Validating inputs...")
        if not analysis_results:
            print("Error: No analysis results provided")
            return None
        
        if analysis_mode not in ['single', 'bulk']:
            print(f"Error: Invalid analysis mode '{analysis_mode}'")
            return None
        
        print("Generating improved PDF...")
        pdf_bytes = improved_generator.generate_improved_pdf_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results
        )
        
        if pdf_bytes:
            print(f"Improved PDF generated successfully. Size: {len(pdf_bytes)} bytes")
            return pdf_bytes
        else:
            print("Error: PDF generation returned None")
            return None
        
    except Exception as e:
        print(f"Error in improved PDF generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ALTERNATIVE: Cached version for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_comprehensive_pdf_report_cached(analysis_results_str, analysis_mode, server_specs_str=None, ai_insights_str=None, transfer_results_str=None):
    """
    Cached version of PDF generator (converts strings back to objects)
    Use this if you're generating the same report multiple times
    """
    import json
    
    try:
        # Convert string parameters back to objects
        analysis_results = json.loads(analysis_results_str) if analysis_results_str else None
        server_specs = json.loads(server_specs_str) if server_specs_str else None
        ai_insights = json.loads(ai_insights_str) if ai_insights_str else None
        transfer_results = json.loads(transfer_results_str) if transfer_results_str else None
        
        # Call the main function
        return generate_improved_pdf_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results
        )
    
    except json.JSONDecodeError as e:
        print(f"Error parsing cached parameters: {e}")
        return None

# USAGE EXAMPLE in your TAB 6:
def example_usage_in_tab6():
    """Example of how to use the improved helper function"""
    
    # Your existing code in TAB 6, replace the PDF generation section with:
    if st.button("📄 Generate Enhanced PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating Enhanced PDF Report..."):
            try:
                # Prepare inputs with validation
                if not current_analysis_results:
                    st.error("❌ No analysis results available")
                    return
                
                if analysis_mode_for_pdf not in ['single', 'bulk']:
                    st.error(f"❌ Invalid analysis mode: {analysis_mode_for_pdf}")
                    return
                
                # Prepare AI insights (ensure it's not None)
                ai_insights_for_pdf = st.session_state.ai_insights if st.session_state.ai_insights else {
                    "risk_level": "Unknown",
                    "cost_optimization_potential": 0,
                    "ai_analysis": "AI insights were not available during analysis generation."
                }
                
                # Prepare transfer results
                transfer_results_for_pdf = None
                if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
                    transfer_results_for_pdf = st.session_state.transfer_results
                
                # Generate PDF using the improved helper function
                pdf_bytes = generate_improved_pdf_report(
                    analysis_results=current_analysis_results,
                    analysis_mode=analysis_mode_for_pdf,
                    server_specs=current_server_specs_for_pdf,
                    ai_insights=ai_insights_for_pdf,
                    transfer_results=transfer_results_for_pdf
                )
                
                if pdf_bytes:
                    st.success("✅ Enhanced PDF Report generated successfully!")
                    
                    # Calculate file size
                    file_size_mb = len(pdf_bytes) / (1024 * 1024)
                    st.info(f"📄 PDF Size: {file_size_mb:.2f} MB")
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"aws_rds_migration_enhanced_{analysis_mode_for_pdf}_{timestamp}.pdf"
                    
                    st.download_button(
                        label="📥 Download Enhanced PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Failed to generate PDF. Check the error messages above.")
                    
                    # Offer alternative export
                    st.subheader("🔄 Alternative Export Options")
                    if st.button("📊 Export Analysis as JSON (Fallback)", use_container_width=True):
                        export_data = {
                            'analysis_mode': analysis_mode_for_pdf,
                            'analysis_results': current_analysis_results,
                            'server_specs': current_server_specs_for_pdf,
                            'ai_insights': ai_insights_for_pdf,
                            'transfer_results': transfer_results_for_pdf,
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        json_data = json.dumps(export_data, indent=2, default=str)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=json_data,
                            file_name=f"analysis_report_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Error in PDF generation process: {str(e)}")
                st.code(traceback.format_exc())

# DEBUG HELPER: Add this temporarily to see what's happening
def debug_pdf_generation(analysis_results, analysis_mode):
    """Debug helper to test PDF generation step by step"""
    print(f"Debug: Analysis mode = {analysis_mode}")
    print(f"Debug: Analysis results type = {type(analysis_results)}")
    print(f"Debug: Analysis results keys = {list(analysis_results.keys()) if isinstance(analysis_results, dict) else 'Not a dict'}")
    
    try:
        generator = ImprovedReportGenerator()
        print("Debug: ComprehensiveReportGenerator created successfully")
        
        # Test if styles are set up correctly
        required_styles = ['ComprehensiveTitle', 'ExecutiveHeader', 'DetailedSectionHeader', 'SubsectionHeader', 'TechnicalSpec']
        missing_styles = []
        for style_name in required_styles:
            try:
                _ = generator.styles[style_name]
            except KeyError:
                missing_styles.append(style_name)
        
        if missing_styles:
            print(f"Debug: Missing styles: {missing_styles}")
        else:
            print("Debug: All required styles are available")
            
        return True
        
    except Exception as e:
        print(f"Debug: Error creating generator: {e}")
        return False

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_cost_heatmap(results):
    """Create cost heatmap for environment comparison"""
    if not results:
        return None
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        return None
    
    environments = list(valid_results.keys())
    cost_categories = ['Instance Cost', 'Storage Cost', 'Backup Cost', 'Total Cost']
    
    cost_matrix = []
    for env in environments:
        result = valid_results[env]
        if 'writer' in result and 'readers' in result:
            instance_cost_sum = safe_get(result['cost_breakdown'], 'writer_monthly', 0) + \
                                safe_get(result['cost_breakdown'], 'readers_monthly', 0)
            storage_cost = safe_get(result['cost_breakdown'], 'storage_monthly', 0)
            backup_cost = safe_get(result['cost_breakdown'], 'backup_monthly', 0)
            total_cost = safe_get(result, 'total_cost', 0)
        else:
            cost_breakdown = safe_get(result, 'cost_breakdown', {})
            instance_cost_sum = safe_get(cost_breakdown, 'instance_monthly', safe_get(result, 'instance_cost', 0))
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', safe_get(result, 'storage_cost', 0))
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', safe_get(result, 'storage_cost', 0) * 0.25)
            total_cost = safe_get(result, 'total_cost', 0)

        row = [instance_cost_sum, storage_cost, backup_cost, total_cost]
        cost_matrix.append(row)
    
    cost_matrix = np.array(cost_matrix).T
    
    fig = go.Figure(data=go.Heatmap(
        z=cost_matrix,
        x=environments,
        y=cost_categories,
        colorscale='RdYlBu_r',
        text=[[f'${cost:,.0f}' for cost in row] for row in cost_matrix],
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Monthly Cost ($)")
    ))
    
    fig.update_layout(
        title={
            'text': "🔥 Cost Heatmap - All Categories vs Environments",
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title="Environment",
        yaxis_title="Cost Category",
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_workload_distribution_pie(workload_chars):
    """Create workload characteristics pie chart"""
    if not workload_chars:
        return None
    
    io_mapping = {'read_heavy': 70, 'write_heavy': 30, 'mixed': 50}
    read_pct = io_mapping.get(workload_chars.io_pattern, 50)
    write_pct = 100 - read_pct
    
    fig = go.Figure(data=[go.Pie(
        labels=['Read Operations', 'Write Operations'],
        values=[read_pct, write_pct],
        hole=.3,
        marker_colors=['#36A2EB', '#FF6384']
    )])
    
    fig.update_layout(
        title='📊 Workload I/O Distribution',
        height=350
    )
    
    return fig

def create_bulk_analysis_summary_chart(bulk_results):
    """Create summary chart for bulk analysis results"""
    if not bulk_results:
        return None
    
    server_names = []
    total_costs = []
    instance_types = []
    vcpus = []
    ram_gb = []
    
    for server_name, results in bulk_results.items():
        if 'error' not in results:
            result = results.get('PROD', list(results.values())[0])
            if 'error' not in result:
                server_names.append(server_name)
                total_costs.append(safe_get(result, 'total_cost', 0))
                
                if 'writer' in result:
                    instance_types.append(safe_get_str(result['writer'], 'instance_type', 'Unknown'))
                    vcpus.append(safe_get(result['writer'], 'actual_vCPUs', 0))
                    ram_gb.append(safe_get(result['writer'], 'actual_RAM_GB', 0))
                else:
                    instance_types.append(safe_get_str(result, 'instance_type', 'Unknown'))
                    vcpus.append(safe_get(result, 'actual_vCPUs', 0))
                    ram_gb.append(safe_get(result, 'actual_RAM_GB', 0))
    
    if not server_names:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Cost by Server', 'vCPUs by Server', 'RAM by Server', 'Instance Type Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    fig.add_trace(
        go.Bar(x=server_names, y=total_costs, name='Monthly Cost', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=server_names, y=vcpus, name='vCPUs', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=server_names, y=ram_gb, name='RAM (GB)', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    instance_counts = pd.Series(instance_types).value_counts()
    fig.add_trace(
        go.Pie(labels=instance_counts.index, values=instance_counts.values, name="Instance Types"),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="📊 Bulk Analysis Summary Dashboard",
        showlegend=False,
        height=600
    )
    
    return fig

def parse_bulk_upload_file(uploaded_file):
    """Parse bulk upload file and extract server specifications"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        df.columns = df.columns.str.lower().str.strip()
        
        column_mapping = {
            'server_name': ['server_name', 'servername', 'name', 'hostname', 'server'],
            'cpu_cores': ['cpu_cores', 'cpucores', 'cores', 'cpu', 'processors'],
            'ram_gb': ['ram_gb', 'ramgb', 'ram', 'memory', 'memory_gb'],
            'storage_gb': ['storage_gb', 'storagegb', 'storage', 'disk', 'disk_gb'],
            'peak_cpu_percent': ['peak_cpu_percent', 'peak_cpu', 'cpu_util', 'cpu_utilization', 'max_cpu'],
            'peak_ram_percent': ['peak_ram_percent', 'peak_ram', 'ram_util', 'ram_utilization', 'max_memory'],
            'max_iops': ['max_iops', 'maxiops', 'iops', 'peak_iops'],
            'max_throughput_mbps': ['max_throughput_mbps', 'max_throughput', 'throughput', 'bandwidth'],
            'database_engine': ['database_engine', 'db_engine', 'engine', 'database']
        }
        
        mapped_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col in possible_names:
                    mapped_columns[standard_name] = col
                    break
        
        required_columns = ['server_name', 'cpu_cores', 'ram_gb', 'storage_gb']
        missing_columns = [col for col in required_columns if col not in mapped_columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: Server Name, CPU Cores, RAM (GB), Storage (GB)")
            return None
        
        servers = []
        for idx, row in df.iterrows():
            try:
                server = {
                    'server_name': str(row[mapped_columns['server_name']]).strip(),
                    'cpu_cores': int(float(row[mapped_columns['cpu_cores']])),
                    'ram_gb': int(float(row[mapped_columns['ram_gb']])),
                    'storage_gb': int(float(row[mapped_columns['storage_gb']])),
                    'peak_cpu_percent': int(float(row.get(mapped_columns.get('peak_cpu_percent', ''), 75))),
                    'peak_ram_percent': int(float(row.get(mapped_columns.get('peak_ram_percent', ''), 80))),
                    'max_iops': int(float(row.get(mapped_columns.get('max_iops', ''), 1000))),
                    'max_throughput_mbps': int(float(row.get(mapped_columns.get('max_throughput_mbps', ''), 125))),
                    'database_engine': str(row.get(mapped_columns.get('database_engine', ''), 'oracle-ee')).strip().lower()
                }
                
                if server['cpu_cores'] <= 0 or server['ram_gb'] <= 0 or server['storage_gb'] <= 0:
                    st.warning(f"Invalid data for server {server['server_name']} at row {idx + 1}. Skipping.")
                    continue
                
                servers.append(server)
                
            except (ValueError, TypeError) as e:
                st.warning(f"Error parsing row {idx + 1}: {e}. Skipping.")
                continue
        
        if not servers:
            st.error("No valid server data found in the uploaded file.")
            return None
        
        st.success(f"Successfully parsed {len(servers)} servers from the uploaded file.")
        return servers
        
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

# ================================
# INITIALIZATION
# ================================

# Check authentication before showing the app
if not check_authentication():
    show_login_form()
    st.stop()

# Enhanced Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .status-success {
        background: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .status-error {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .status-info {
        background: #d1ecf1;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    .migration-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }    
    .advisory-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }    
    .phase-timeline {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }    
    .chart-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }    
    .writer-box {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }    
    .reader-box {
        background: #f3e5f5;
        border: 2px solid #9c27b0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }    
    .spec-section {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }    
    .bulk-upload-zone {
        border: 2px dashed #007bff;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }    
    .server-summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'calculator' not in st.session_state:
    st.session_state.calculator = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'migration_configured' not in st.session_state:
    st.session_state.migration_configured = False
if 'user_claude_api_key_input' not in st.session_state:
    st.session_state.user_claude_api_key_input = ""
if 'source_engine' not in st.session_state:
    st.session_state.source_engine = None
if 'target_engine' not in st.session_state:
    st.session_state.target_engine = None
if 'deployment_option' not in st.session_state:
    st.session_state.deployment_option = "Multi-AZ"
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = {}
if 'on_prem_servers' not in st.session_state:
    st.session_state.on_prem_servers = []
if 'bulk_upload_data' not in st.session_state:
    st.session_state.bulk_upload_data = None
if 'current_analysis_mode' not in st.session_state:
    st.session_state.current_analysis_mode = 'single'
if 'firebase_app' not in st.session_state:
    st.session_state.firebase_app = None
if 'firebase_auth' not in st.session_state:
    st.session_state.firebase_auth = None
if 'firebase_db' not in st.session_state:
    st.session_state.firebase_db = None
if 'transfer_results' not in st.session_state:
    st.session_state.transfer_results = None
if 'transfer_data_size' not in st.session_state:
    st.session_state.transfer_data_size = 0
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
# Add to existing session state initialization
if 'selected_environments' not in st.session_state:
    st.session_state.selected_environments = ['PROD']
if 'env_configs' not in st.session_state:
    st.session_state.env_configs = {
        'DEV': {'cpu_ratio': 0.25, 'ram_ratio': 0.25, 'storage_ratio': 0.3, 'deployment': 'Single-AZ'},
        'QA': {'cpu_ratio': 0.4, 'ram_ratio': 0.4, 'storage_ratio': 0.5, 'deployment': 'Single-AZ'},
        'UAT': {'cpu_ratio': 0.6, 'ram_ratio': 0.6, 'storage_ratio': 0.7, 'deployment': 'Multi-AZ'},
        'PREPROD': {'cpu_ratio': 0.8, 'ram_ratio': 0.8, 'storage_ratio': 0.9, 'deployment': 'Multi-AZ'},
        'PROD': {'cpu_ratio': 1.0, 'ram_ratio': 1.0, 'storage_ratio': 1.0, 'deployment': 'Multi-AZ'}
    }
if 'enhanced_bulk_upload_data' not in st.session_state:
    st.session_state.enhanced_bulk_upload_data = None
    

# Initialize Firebase
if st.session_state.firebase_app is None:
    st.session_state.firebase_app, st.session_state.firebase_auth, st.session_state.firebase_db = initialize_firebase()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize the pricing API"""
    try:
        pricing_api = EnhancedAWSPricingAPI()
        return pricing_api
    except Exception as e:
        st.error(f"Error initializing static components: {e}")
        return None

pricing_api = initialize_components()
if not pricing_api:
    st.error("Failed to initialize required components")
    st.stop()
    
@st.cache_resource
def initialize_transfer_calculator():
    """Initialize the data transfer calculator"""
    try:
        return DataTransferCalculator()
    except Exception as e:
        st.error(f"Error initializing transfer calculator: {e}")
        return None

transfer_calculator = initialize_transfer_calculator()

# Initialize calculator
if st.session_state.calculator is None:
    claude_api_key = None
    if st.session_state.user_claude_api_key_input:
        claude_api_key = st.session_state.user_claude_api_key_input
    elif "anthropic" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["anthropic"]:
        claude_api_key = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]
    else:
        claude_api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    st.session_state.calculator = EnhancedRDSSizingCalculator(
        anthropic_api_key=claude_api_key,
        use_real_time_pricing=True
    )

# ================================
# MAIN APPLICATION
# ================================

# Header
st.title("🚀 Enterprise AWS RDS Migration & Sizing Tool")
st.markdown("**AI-Powered Analysis • Homogeneous & Heterogeneous Migrations • Real-time AWS Pricing • Advanced Analytics**")

# API Key input
st.subheader("🤖 AI Integration (Anthropic Claude API Key)")
st.session_state.user_claude_api_key_input = st.text_input(
    "Enter your Anthropic API Key (optional)",
    type="password",
    value=st.session_state.user_claude_api_key_input,
    help="Provide your Anthropic API key here to enable AI-powered insights."
)
st.markdown("---")

# Show user info in sidebar
show_user_info()

# Display System Status
st.sidebar.subheader("System Status")
if st.session_state.firebase_app:
    st.sidebar.success("🔥 Firebase Connected")
    st.sidebar.write(f"**Project:** {st.session_state.firebase_app.project_id}")
else:
    st.sidebar.warning("Firebase not connected")

st.markdown("---")

# Inject mock data for testing PDF generation (remove in production)
# This mock data is ONLY injected if bulk_results is empty.
if 'bulk_results' not in st.session_state or not st.session_state.bulk_results:
    if 'server1' not in st.session_state.bulk_results:
        st.session_state.bulk_results = {
            'server1': {
                'PROD': {
                    'total_cost': 1500,
                    'instance_type': 'db.m5.large',
                    'actual_vCPUs': 2,
                    'actual_RAM_GB': 8,
                    'storage_GB': 100,
                    'cost_breakdown': {
                        'instance_monthly': 1200,
                        'storage_monthly': 200,
                        'backup_monthly': 100
                    },
                    'writer': {
                        'instance_type': 'db.m5.large',
                        'actual_vCPUs': 2,
                        'actual_RAM_GB': 8
                    }
                }
            }
        }
        st.session_state.on_prem_servers = [{
            'server_name': 'server1',
            'cpu_cores': 2,
            'ram_gb': 8,
            'storage_gb': 100,
            'peak_cpu_percent': 75,
            'peak_ram_percent': 80,
            'max_iops': 1000,
            'max_throughput_mbps': 125,
            'database_engine': 'oracle-ee'
        }]
    
    # Also inject a dummy AI insight for mock data if none exists
    if not st.session_state.ai_insights:
        st.session_state.ai_insights = {
            "risk_level": "Medium",
            "cost_optimization_potential": 0.15,
            "recommended_writers": 1,
            "recommended_readers": 1,
            "ai_analysis": "This is a mock AI analysis for the single server. It suggests optimizing storage and considering a multi-AZ deployment for high availability. The database workload seems balanced with a slight read bias. Further analysis with historical performance data would refine recommendations."
        }
    
    # Also inject a dummy transfer result for mock data if none exists
    if not st.session_state.transfer_results:
        st.session_state.transfer_results = {
            'datasync_dx': TransferMethodResult(
                recommended_method='AWS DataSync (Direct Connect)',
                transfer_time_hours=10.5,
                transfer_time_days=0.4,
                total_cost=50.0,
                bandwidth_utilization=90.0,
                estimated_downtime_hours=0.1,
                cost_breakdown={'data_transfer': 25.0, 'datasync_task': 25.0}
            ),
            'datasync_internet': TransferMethodResult(
                recommended_method='AWS DataSync (Internet)',
                transfer_time_hours=24.0,
                transfer_time_days=1.0,
                total_cost=30.0,
                bandwidth_utilization=70.0,
                estimated_downtime_hours=0.5,
                cost_breakdown={'data_transfer': 10.0, 'datasync_task': 20.0}
            )
        }
        st.session_state.transfer_data_size = 500

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Migration Planning", "🖥️ Server Specifications", "📊 Sizing Analysis", "💰 Financial Analysis", "🤖 AI Insights", "📋 Reports"])

# ================================
# TAB 1: MIGRATION PLANNING
# ================================

with tab1:
    st.header("Migration Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Migration Type")
        
        source_engine_selection = st.selectbox(
            "Source Database Engine",
            ["oracle-ee", "oracle-se", "oracle-se1", "oracle-se2", "sqlserver-ee", "sqlserver-se", 
             "mysql", "postgres", "mariadb"],
            index=0 if st.session_state.source_engine is None else ["oracle-ee", "oracle-se", "oracle-se1", "oracle-se2", "sqlserver-ee", "sqlserver-se", 
             "mysql", "postgres", "mariadb"].index(st.session_state.source_engine),
            key="source_engine_select"
        )
        
        target_engine_selection = st.selectbox(
            "Target AWS Database Engine",
            ["postgres", "aurora-postgresql", "aurora-mysql", "mysql", "oracle-ee", "oracle-se2", 
             "sqlserver-ee", "sqlserver-se"],
            index=0 if st.session_state.target_engine is None else ["postgres", "aurora-postgresql", "aurora-mysql", "mysql", "oracle-ee", "oracle-se2", 
             "sqlserver-ee", "sqlserver-se"].index(st.session_state.target_engine),
            key="target_engine_select"
        )
        
        if source_engine_selection.split('-')[0] == target_engine_selection.split('-')[0]:
            migration_type_display = "Homogeneous"
            migration_color = "success"
        else:
            migration_type_display = "Heterogeneous"
            migration_color = "warning"
        
        st.markdown(f"""
        <div class="status-card status-{migration_color}">
            <strong>Migration Type: {migration_type_display}</strong><br>
            {source_engine_selection} → {target_engine_selection}
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("⚙️ Workload Characteristics")
        
        cpu_pattern = st.selectbox("CPU Utilization Pattern", ["steady", "bursty", "peak_hours"], key="cpu_pattern_select")
        memory_pattern = st.selectbox("Memory Usage Pattern", ["steady", "high_variance", "growing"], key="memory_pattern_select")
        io_pattern = st.selectbox("I/O Pattern", ["read_heavy", "write_heavy", "mixed"], key="io_pattern_select")
        connection_count = st.number_input("Typical Connection Count", min_value=10, max_value=10000, value=100, step=10, key="connection_count_input")
        transaction_volume = st.selectbox("Transaction Volume", ["low", "medium", "high", "very_high"], index=1, key="transaction_volume_select")
        analytical_workload = st.checkbox("Analytical/Reporting Workload", key="analytical_workload_checkbox")
    
    st.subheader("☁️ AWS Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"], key="region_select")
    
    with col2:
        st.session_state.deployment_option = st.selectbox(
            "Deployment Option", 
            ["Single-AZ", "Multi-AZ", "Multi-AZ Cluster", "Aurora Global", "Serverless"], 
            index=["Single-AZ", "Multi-AZ", "Multi-AZ Cluster", "Aurora Global", "Serverless"].index(st.session_state.deployment_option),
            key="deployment_option_select"
        )
    
    with col3:
        storage_type = st.selectbox("Storage Type", ["gp3", "gp2", "io1", "io2", "aurora"], key="storage_type_select")
    
    st.session_state.source_engine = source_engine_selection
    st.session_state.target_engine = target_engine_selection
    st.session_state.region = region
    st.session_state.storage_type = storage_type

    # Add this section to TAB 1 (Migration Planning) after the existing configuration

# Enhanced Environment Configuration Section
st.subheader("🌐 Multi-Environment Configuration")

# Environment selection
st.markdown("**Select environments to analyze:**")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    analyze_dev = st.checkbox("🔧 DEV", value=True, key="env_dev")
with col2:
    analyze_qa = st.checkbox("🧪 QA", value=True, key="env_qa")
with col3:
    analyze_uat = st.checkbox("🚀 UAT", value=True, key="env_uat")
with col4:
    analyze_preprod = st.checkbox("⚡ PreProd", value=True, key="env_preprod")
with col5:
    analyze_prod = st.checkbox("🏭 PROD", value=True, key="env_prod")

# Store selected environments in session state
st.session_state.selected_environments = []
env_mapping = {
    'analyze_dev': 'DEV',
    'analyze_qa': 'QA', 
    'analyze_uat': 'UAT',
    'analyze_preprod': 'PREPROD',
    'analyze_prod': 'PROD'
}

for checkbox_key, env_name in env_mapping.items():
    if st.session_state.get(checkbox_key, False):
        st.session_state.selected_environments.append(env_name)

if st.session_state.selected_environments:
    st.success(f"✅ Selected environments: {', '.join(st.session_state.selected_environments)}")
else:
    st.warning("⚠️ Please select at least one environment to analyze")

# Environment-specific sizing ratios
st.subheader("📊 Environment Sizing Configuration")
st.markdown("Configure relative sizing compared to PROD environment:")

# Initialize environment configs in session state
if 'env_configs' not in st.session_state:
    st.session_state.env_configs = {
        'DEV': {'cpu_ratio': 0.25, 'ram_ratio': 0.25, 'storage_ratio': 0.3, 'deployment': 'Single-AZ'},
        'QA': {'cpu_ratio': 0.4, 'ram_ratio': 0.4, 'storage_ratio': 0.5, 'deployment': 'Single-AZ'},
        'UAT': {'cpu_ratio': 0.6, 'ram_ratio': 0.6, 'storage_ratio': 0.7, 'deployment': 'Multi-AZ'},
        'PREPROD': {'cpu_ratio': 0.8, 'ram_ratio': 0.8, 'storage_ratio': 0.9, 'deployment': 'Multi-AZ'},
        'PROD': {'cpu_ratio': 1.0, 'ram_ratio': 1.0, 'storage_ratio': 1.0, 'deployment': 'Multi-AZ'}
    }

# Display environment configuration table
env_config_data = []
for env in ['DEV', 'QA', 'UAT', 'PREPROD', 'PROD']:
    if env in st.session_state.selected_environments:
        config = st.session_state.env_configs[env]
        env_config_data.append({
            'Environment': env,
            'CPU Ratio': f"{config['cpu_ratio']:.2f}x",
            'RAM Ratio': f"{config['ram_ratio']:.2f}x", 
            'Storage Ratio': f"{config['storage_ratio']:.2f}x",
            'Deployment': config['deployment']
        })

if env_config_data:
    config_df = pd.DataFrame(env_config_data)
    st.dataframe(config_df, use_container_width=True)

# Advanced environment configuration
with st.expander("🔧 Advanced Environment Configuration"):
    st.markdown("**Customize environment-specific parameters:**")
    
    for env in st.session_state.selected_environments:
        st.markdown(f"**{env} Environment Configuration:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_ratio = st.slider(
                f"CPU Ratio ({env})",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.env_configs[env]['cpu_ratio'],
                step=0.1,
                key=f"cpu_ratio_{env}"
            )
        
        with col2:
            ram_ratio = st.slider(
                f"RAM Ratio ({env})",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.env_configs[env]['ram_ratio'],
                step=0.1,
                key=f"ram_ratio_{env}"
            )
        
        with col3:
            storage_ratio = st.slider(
                f"Storage Ratio ({env})",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.env_configs[env]['storage_ratio'],
                step=0.1,
                key=f"storage_ratio_{env}"
            )
        
        with col4:
            deployment = st.selectbox(
                f"Deployment ({env})",
                ["Single-AZ", "Multi-AZ", "Multi-AZ Cluster"],
                index=["Single-AZ", "Multi-AZ", "Multi-AZ Cluster"].index(st.session_state.env_configs[env]['deployment']),
                key=f"deployment_{env}"
            )
        
        # Update session state
        st.session_state.env_configs[env] = {
            'cpu_ratio': cpu_ratio,
            'ram_ratio': ram_ratio,
            'storage_ratio': storage_ratio,
            'deployment': deployment
        }
    
    
    if st.button("🎯 Configure Migration", type="primary", use_container_width=True):
        with st.spinner("Configuring migration parameters..."):
            try:
                # Re-initialize calculator with updated API key if provided
                claude_api_key_current = None
                if st.session_state.user_claude_api_key_input:
                    claude_api_key_current = st.session_state.user_claude_api_key_input
                elif "anthropic" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["anthropic"]:
                    claude_api_key_current = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]
                else:
                    claude_api_key_current = os.environ.get('ANTHROPIC_API_KEY')

                # Only re-initialize if the key has changed or calculator is None
                if st.session_state.calculator is None or \
                   (hasattr(st.session_state.calculator, 'anthropic_api_key') and \
                    st.session_state.calculator.anthropic_api_key != claude_api_key_current) or \
                   (not hasattr(st.session_state.calculator, 'anthropic_api_key') and claude_api_key_current):
                    st.session_state.calculator = EnhancedRDSSizingCalculator(
                        anthropic_api_key=claude_api_key_current,
                        use_real_time_pricing=True
                    )

                workload_chars = WorkloadCharacteristics(
                    cpu_utilization_pattern=cpu_pattern,
                    memory_usage_pattern=memory_pattern,
                    io_pattern=io_pattern,
                    connection_count=connection_count,
                    transaction_volume=transaction_volume,
                    analytical_workload=analytical_workload
                )
                
                st.session_state.calculator.set_migration_parameters(
                    source_engine_selection, target_engine_selection, workload_chars
                )
                
                st.session_state.migration_configured = True
                
                migration_profile = st.session_state.calculator.migration_profile
                
                st.markdown(f"""
                <div class="migration-card">
                    <h3>🎯 Migration Configuration Complete</h3>
                    <strong>Migration Type:</strong> {migration_profile.migration_type.value.title()}<br>
                    <strong>Complexity Factor:</strong> {migration_profile.complexity_factor:.1f}x<br>
                    <strong>Feature Compatibility:</strong> {migration_profile.feature_compatibility*100:.1f}%<br>
                    <strong>Recommended Sizing Buffer:</strong> {migration_profile.recommended_sizing_buffer*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Migration configured successfully! Proceed to Server Specifications.")
                
            except Exception as e:
                st.error(f"❌ Error configuring migration: {str(e)}")


# ================================
# TAB 2: SERVER SPECIFICATIONS (CORRECTED)
# ================================
# Enhanced Bulk Upload Functions
# Add these to your TAB 2 (Server Specifications)

def generate_enhanced_bulk_template():
    """
    Generate enhanced CSV template with environment-specific columns
    """
    selected_environments = st.session_state.get('selected_environments', ['DEV', 'QA', 'UAT', 'PREPROD', 'PROD'])
    
    # Base columns
    base_columns = [
        'server_name', 'cpu_cores', 'ram_gb', 'storage_gb',
        'peak_cpu_percent', 'peak_ram_percent', 'max_iops', 
        'max_throughput_mbps', 'database_engine'
    ]
    
    # Environment-specific columns
    env_columns = []
    for env in selected_environments:
        env_columns.extend([
            f'{env.lower()}_cpu_ratio',
            f'{env.lower()}_ram_ratio', 
            f'{env.lower()}_storage_ratio',
            f'{env.lower()}_deployment_type'
        ])
    
    all_columns = base_columns + env_columns
    
    # Sample data
    sample_data = []
    
    # Server 1 - Oracle Production Server
    server1 = {
        'server_name': 'PROD-DB-01',
        'cpu_cores': 16,
        'ram_gb': 64,
        'storage_gb': 1000,
        'peak_cpu_percent': 75,
        'peak_ram_percent': 80,
        'max_iops': 5000,
        'max_throughput_mbps': 250,
        'database_engine': 'oracle-ee'
    }
    
    # Add environment-specific ratios
    env_ratios = {
        'DEV': {'cpu': 0.25, 'ram': 0.25, 'storage': 0.3, 'deployment': 'Single-AZ'},
        'QA': {'cpu': 0.4, 'ram': 0.4, 'storage': 0.5, 'deployment': 'Single-AZ'},
        'UAT': {'cpu': 0.6, 'ram': 0.6, 'storage': 0.7, 'deployment': 'Multi-AZ'},
        'PREPROD': {'cpu': 0.8, 'ram': 0.8, 'storage': 0.9, 'deployment': 'Multi-AZ'},
        'PROD': {'cpu': 1.0, 'ram': 1.0, 'storage': 1.0, 'deployment': 'Multi-AZ'}
    }
    
    for env in selected_environments:
        ratios = env_ratios.get(env, {'cpu': 1.0, 'ram': 1.0, 'storage': 1.0, 'deployment': 'Multi-AZ'})
        server1[f'{env.lower()}_cpu_ratio'] = ratios['cpu']
        server1[f'{env.lower()}_ram_ratio'] = ratios['ram']
        server1[f'{env.lower()}_storage_ratio'] = ratios['storage']
        server1[f'{env.lower()}_deployment_type'] = ratios['deployment']
    
    sample_data.append(server1)
    
    # Server 2 - MySQL Application Server
    server2 = {
        'server_name': 'APP-DB-01',
        'cpu_cores': 8,
        'ram_gb': 32,
        'storage_gb': 500,
        'peak_cpu_percent': 60,
        'peak_ram_percent': 70,
        'max_iops': 3000,
        'max_throughput_mbps': 150,
        'database_engine': 'mysql'
    }
    
    for env in selected_environments:
        ratios = env_ratios.get(env, {'cpu': 1.0, 'ram': 1.0, 'storage': 1.0, 'deployment': 'Multi-AZ'})
        server2[f'{env.lower()}_cpu_ratio'] = ratios['cpu']
        server2[f'{env.lower()}_ram_ratio'] = ratios['ram']
        server2[f'{env.lower()}_storage_ratio'] = ratios['storage']
        server2[f'{env.lower()}_deployment_type'] = ratios['deployment']
    
    sample_data.append(server2)
    
    # Server 3 - PostgreSQL Analytics Server
    server3 = {
        'server_name': 'ANALYTICS-DB-01',
        'cpu_cores': 12,
        'ram_gb': 48,
        'storage_gb': 2000,
        'peak_cpu_percent': 85,
        'peak_ram_percent': 75,
        'max_iops': 4000,
        'max_throughput_mbps': 200,
        'database_engine': 'postgres'
    }
    
    for env in selected_environments:
        ratios = env_ratios.get(env, {'cpu': 1.0, 'ram': 1.0, 'storage': 1.0, 'deployment': 'Multi-AZ'})
        server3[f'{env.lower()}_cpu_ratio'] = ratios['cpu']
        server3[f'{env.lower()}_ram_ratio'] = ratios['ram']
        server3[f'{env.lower()}_storage_ratio'] = ratios['storage']
        server3[f'{env.lower()}_deployment_type'] = ratios['deployment']
    
    sample_data.append(server3)
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Ensure all columns are present
    for col in all_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[all_columns]
    
    return df

def parse_enhanced_bulk_upload_file(uploaded_file):
    """
    Parse enhanced bulk upload file with environment support
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        # Clean column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Basic column mapping
        base_column_mapping = {
            'server_name': ['server_name', 'servername', 'name', 'hostname', 'server'],
            'cpu_cores': ['cpu_cores', 'cpucores', 'cores', 'cpu', 'processors'],
            'ram_gb': ['ram_gb', 'ramgb', 'ram', 'memory', 'memory_gb'],
            'storage_gb': ['storage_gb', 'storagegb', 'storage', 'disk', 'disk_gb'],
            'peak_cpu_percent': ['peak_cpu_percent', 'peak_cpu', 'cpu_util', 'cpu_utilization', 'max_cpu'],
            'peak_ram_percent': ['peak_ram_percent', 'peak_ram', 'ram_util', 'ram_utilization', 'max_memory'],
            'max_iops': ['max_iops', 'maxiops', 'iops', 'peak_iops'],
            'max_throughput_mbps': ['max_throughput_mbps', 'max_throughput', 'throughput', 'bandwidth'],
            'database_engine': ['database_engine', 'db_engine', 'engine', 'database']
        }
        
        # Map base columns
        mapped_columns = {}
        for standard_name, possible_names in base_column_mapping.items():
            for col in df.columns:
                if col in possible_names:
                    mapped_columns[standard_name] = col
                    break
        
        # Check for required base columns
        required_columns = ['server_name', 'cpu_cores', 'ram_gb', 'storage_gb']
        missing_columns = [col for col in required_columns if col not in mapped_columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Detect environment-specific columns
        selected_environments = st.session_state.get('selected_environments', ['PROD'])
        env_column_patterns = {}
        
        for env in selected_environments:
            env_lower = env.lower()
            env_column_patterns[env] = {}
            
            # Look for environment-specific columns
            for col in df.columns:
                if col.startswith(f'{env_lower}_'):
                    if 'cpu_ratio' in col:
                        env_column_patterns[env]['cpu_ratio'] = col
                    elif 'ram_ratio' in col:
                        env_column_patterns[env]['ram_ratio'] = col
                    elif 'storage_ratio' in col:
                        env_column_patterns[env]['storage_ratio'] = col
                    elif 'deployment' in col:
                        env_column_patterns[env]['deployment'] = col
        
        # Parse server data
        servers = []
        for idx, row in df.iterrows():
            try:
                # Base server configuration
                server = {
                    'server_name': str(row[mapped_columns['server_name']]).strip(),
                    'cpu_cores': int(float(row[mapped_columns['cpu_cores']])),
                    'ram_gb': int(float(row[mapped_columns['ram_gb']])),
                    'storage_gb': int(float(row[mapped_columns['storage_gb']])),
                    'peak_cpu_percent': int(float(row.get(mapped_columns.get('peak_cpu_percent', ''), 75))),
                    'peak_ram_percent': int(float(row.get(mapped_columns.get('peak_ram_percent', ''), 80))),
                    'max_iops': int(float(row.get(mapped_columns.get('max_iops', ''), 1000))),
                    'max_throughput_mbps': int(float(row.get(mapped_columns.get('max_throughput_mbps', ''), 125))),
                    'database_engine': str(row.get(mapped_columns.get('database_engine', ''), 'oracle-ee')).strip().lower()
                }
                
                # Environment-specific configurations
                server['environment_configs'] = {}
                for env in selected_environments:
                    env_config = env_column_patterns.get(env, {})
                    
                    # Default environment configuration
                    default_ratios = {
                        'DEV': {'cpu': 0.25, 'ram': 0.25, 'storage': 0.3, 'deployment': 'Single-AZ'},
                        'QA': {'cpu': 0.4, 'ram': 0.4, 'storage': 0.5, 'deployment': 'Single-AZ'},
                        'UAT': {'cpu': 0.6, 'ram': 0.6, 'storage': 0.7, 'deployment': 'Multi-AZ'},
                        'PREPROD': {'cpu': 0.8, 'ram': 0.8, 'storage': 0.9, 'deployment': 'Multi-AZ'},
                        'PROD': {'cpu': 1.0, 'ram': 1.0, 'storage': 1.0, 'deployment': 'Multi-AZ'}
                    }
                    
                    defaults = default_ratios.get(env, {'cpu': 1.0, 'ram': 1.0, 'storage': 1.0, 'deployment': 'Multi-AZ'})
                    
                    server['environment_configs'][env] = {
                        'cpu_ratio': float(row.get(env_config.get('cpu_ratio', ''), defaults['cpu'])),
                        'ram_ratio': float(row.get(env_config.get('ram_ratio', ''), defaults['ram'])),
                        'storage_ratio': float(row.get(env_config.get('storage_ratio', ''), defaults['storage'])),
                        'deployment': str(row.get(env_config.get('deployment', ''), defaults['deployment'])).strip()
                    }
                
                # Validate data
                if server['cpu_cores'] <= 0 or server['ram_gb'] <= 0 or server['storage_gb'] <= 0:
                    st.warning(f"Invalid data for server {server['server_name']} at row {idx + 1}. Skipping.")
                    continue
                
                servers.append(server)
                
            except (ValueError, TypeError) as e:
                st.warning(f"Error parsing row {idx + 1}: {e}. Skipping.")
                continue
        
        if not servers:
            st.error("No valid server data found in the uploaded file.")
            return None
        
        st.success(f"Successfully parsed {len(servers)} servers with environment configurations from {uploaded_file.name}")
        return servers
        
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

# Enhanced Financial Analysis Display
def display_enhanced_financial_analysis(results, analysis_mode):
    """
    Display enhanced financial analysis with environment breakdown
    """
    st.subheader("🏢 Multi-Environment Financial Analysis")
    
    # Environment cost comparison chart
    cost_comparison_fig = create_environment_cost_comparison_chart(results, analysis_mode)
    if cost_comparison_fig:
        st.plotly_chart(cost_comparison_fig, use_container_width=True)
    
    # Cost waterfall chart (single mode only)
    if analysis_mode == 'single':
        waterfall_fig = create_environment_cost_waterfall(results, analysis_mode)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Comprehensive cost summary table
    st.subheader("📊 Environment Cost Summary")
    summary_df = generate_environment_cost_summary_table(results, analysis_mode)
    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export Environment Cost Summary", use_container_width=True):
            csv_data = summary_df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            st.download_button(
                label="📥 Download Environment Cost Summary CSV",
                data=csv_data,
                file_name=f"environment_cost_summary_{analysis_mode}_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Cost optimization recommendations
    st.subheader("💡 Environment-Specific Cost Optimization")
    
    if analysis_mode == 'single':
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        optimization_recommendations = []
        
        for env_name, result in valid_results.items():
            total_cost = safe_get(result, 'total_cost', 0)
            
            if env_name in ['DEV', 'QA']:
                optimization_recommendations.append({
                    'Environment': env_name,
                    'Recommendation': 'Consider using Spot instances or smaller instance types',
                    'Potential Savings': f"${total_cost * 0.3:,.0f}/month (30%)",
                    'Risk Level': 'Low'
                })
            elif env_name == 'UAT':
                optimization_recommendations.append({
                    'Environment': env_name,
                    'Recommendation': 'Use scheduled scaling for non-business hours',
                    'Potential Savings': f"${total_cost * 0.2:,.0f}/month (20%)",
                    'Risk Level': 'Low'
                })
            elif env_name == 'PREPROD':
                optimization_recommendations.append({
                    'Environment': env_name,
                    'Recommendation': 'Optimize storage IOPS and enable compression',
                    'Potential Savings': f"${total_cost * 0.15:,.0f}/month (15%)",
                    'Risk Level': 'Medium'
                })
            elif env_name == 'PROD':
                optimization_recommendations.append({
                    'Environment': env_name,
                    'Recommendation': 'Consider Reserved Instances for 1-3 year commitment',
                    'Potential Savings': f"${total_cost * 0.4:,.0f}/month (40%)",
                    'Risk Level': 'Low'
                })
        
        if optimization_recommendations:
            opt_df = pd.DataFrame(optimization_recommendations)
            st.dataframe(opt_df, use_container_width=True)

# Add this to your TAB 4 financial analysis section
# Enhanced bulk upload section for TAB 2
def show_enhanced_bulk_upload_section():
    """
    Display enhanced bulk upload section with environment support
    """
    st.subheader("📊 Enhanced Bulk Server Analysis")
    
    # Check if environments are selected
    selected_environments = st.session_state.get('selected_environments', [])
    if not selected_environments:
        st.warning("⚠️ Please select environments in the Migration Planning tab first.")
        return
    
    st.info(f"📋 Configured for environments: {', '.join(selected_environments)}")
    
    # Enhanced template download
    with st.expander("📋 Enhanced File Format with Environment Support", expanded=False):
        st.markdown(f"""
        **Enhanced CSV/Excel format supports environment-specific configurations:**
        
        **Base Required Columns:**
        - `server_name` - Server hostname or identifier
        - `cpu_cores` - Number of CPU cores (PROD baseline)
        - `ram_gb` - RAM in GB (PROD baseline)
        - `storage_gb` - Storage in GB (PROD baseline)
        
        **Environment-Specific Columns (Optional):**
        """)
        
        for env in selected_environments:
            env_lower = env.lower()
            st.markdown(f"""
        **{env} Environment:**
        - `{env_lower}_cpu_ratio` - CPU ratio vs PROD (e.g., 0.5 for 50%)
        - `{env_lower}_ram_ratio` - RAM ratio vs PROD  
        - `{env_lower}_storage_ratio` - Storage ratio vs PROD
        - `{env_lower}_deployment_type` - Single-AZ, Multi-AZ, etc.
        """)
        
        # Generate and offer enhanced template
        enhanced_template_df = generate_enhanced_bulk_template()
        enhanced_template_csv = enhanced_template_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Enhanced Template with Environment Support",
            data=enhanced_template_csv,
            file_name=f"enhanced_server_template_{len(selected_environments)}envs.csv",
            mime="text/csv",
            help="Download template configured for your selected environments"
        )
    
    # File upload with enhanced parsing
    uploaded_file = st.file_uploader(
        "Choose Enhanced CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file with environment-specific configurations",
        key="enhanced_bulk_upload_file"
    )
    
    if uploaded_file is not None:
        with st.spinner("📖 Parsing enhanced file with environment configurations..."):
            parsed_servers = parse_enhanced_bulk_upload_file(uploaded_file)
            
            if parsed_servers:
                st.session_state.enhanced_bulk_upload_data = parsed_servers
                st.session_state.on_prem_servers = parsed_servers  # Backward compatibility
                
                st.success(f"✅ Successfully parsed {len(parsed_servers)} servers with environment configurations")
                
                # Enhanced preview
                st.subheader("📊 Enhanced Server Data Preview")
                
                # Base configuration preview
                base_preview_data = []
                for server in parsed_servers:
                    base_preview_data.append({
                        'Server Name': server['server_name'],
                        'CPU Cores': server['cpu_cores'],
                        'RAM (GB)': server['ram_gb'],
                        'Storage (GB)': server['storage_gb'],
                        'Database Engine': server['database_engine'],
                        'Peak CPU %': server['peak_cpu_percent'],
                        'Peak RAM %': server['peak_ram_percent']
                    })
                
                base_preview_df = pd.DataFrame(base_preview_data)
                st.dataframe(base_preview_df, use_container_width=True)
                
                # Environment configuration preview
                st.subheader("🌐 Environment Configuration Preview")
                
                for env in selected_environments:
                    with st.expander(f"{env} Environment Configuration"):
                        env_config_data = []
                        for server in parsed_servers:
                            env_config = server['environment_configs'].get(env, {})
                            env_config_data.append({
                                'Server Name': server['server_name'],
                                'CPU Ratio': f"{env_config.get('cpu_ratio', 1.0):.2f}x",
                                'RAM Ratio': f"{env_config.get('ram_ratio', 1.0):.2f}x",
                                'Storage Ratio': f"{env_config.get('storage_ratio', 1.0):.2f}x",
                                'Deployment': env_config.get('deployment', 'Multi-AZ'),
                                'Est. CPU Cores': int(server['cpu_cores'] * env_config.get('cpu_ratio', 1.0)),
                                'Est. RAM (GB)': int(server['ram_gb'] * env_config.get('ram_ratio', 1.0)),
                                'Est. Storage (GB)': int(server['storage_gb'] * env_config.get('storage_ratio', 1.0))
                            })
                        
                        env_config_df = pd.DataFrame(env_config_data)
                        st.dataframe(env_config_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Environment Summary Statistics")
                
                summary_cols = st.columns(len(selected_environments))
                
                for idx, env in enumerate(selected_environments):
                    with summary_cols[idx]:
                        total_cpu = sum(server['cpu_cores'] * server['environment_configs'][env]['cpu_ratio'] 
                                      for server in parsed_servers)
                        total_ram = sum(server['ram_gb'] * server['environment_configs'][env]['ram_ratio'] 
                                      for server in parsed_servers)
                        total_storage = sum(server['storage_gb'] * server['environment_configs'][env]['storage_ratio'] 
                                          for server in parsed_servers)
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; padding: 1rem; border-radius: 8px;">
                            <h4 style="margin: 0; text-align: center;">{env}</h4>
                            <p style="margin: 0.25rem 0;">CPU Cores: {total_cpu:.0f}</p>
                            <p style="margin: 0.25rem 0;">RAM: {total_ram:.0f} GB</p>
                            <p style="margin: 0.25rem 0;">Storage: {total_storage:.0f} GB</p>
                        </div>
                        """, unsafe_allow_html=True)

# Enhanced TAB2 - Server Specifications
# Add these enhancements to your existing TAB2 implementation

# 1. Add server discovery integration
def show_server_discovery_section():
    """Add automated server discovery options"""
    st.subheader("🔍 Automated Server Discovery")
    
    discovery_method = st.selectbox(
        "Choose Discovery Method",
        ["Manual Entry", "CSV Upload", "Database Query", "Monitoring Tool Export"],
        help="Select how you want to discover and import server specifications"
    )
    
    if discovery_method == "Database Query":
        st.markdown("**Connect to existing monitoring database:**")
        col1, col2 = st.columns(2)
        
        with col1:
            db_host = st.text_input("Database Host")
            db_name = st.text_input("Database Name")
        
        with col2:
            db_user = st.text_input("Username")
            db_password = st.text_input("Password", type="password")
        
        if st.button("🔌 Test Connection & Discover Servers"):
            st.info("This would connect to your monitoring database and auto-discover servers")
    
    elif discovery_method == "Monitoring Tool Export":
        st.markdown("**Import from monitoring tools:**")
        monitoring_tool = st.selectbox(
            "Monitoring Tool",
            ["SolarWinds", "PRTG", "Nagios", "Zabbix", "Custom"],
            help="Select your monitoring tool format"
        )
        
        uploaded_monitoring_file = st.file_uploader(
            f"Upload {monitoring_tool} Export",
            type=['csv', 'xlsx', 'json', 'xml']
        )

# 2. Enhanced server validation
def enhanced_server_validation(servers):
    """Enhanced validation with detailed recommendations"""
    validation_results = {
        'valid': [],
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    for i, server in enumerate(servers):
        server_name = server.get('server_name', f'Server_{i+1}')
        
        # CPU validation
        cpu_cores = server.get('cpu_cores', 0)
        if cpu_cores <= 0:
            validation_results['errors'].append(f"{server_name}: Invalid CPU cores ({cpu_cores})")
        elif cpu_cores > 128:
            validation_results['warnings'].append(f"{server_name}: Very high CPU count ({cpu_cores}) - verify accuracy")
        
        # RAM validation
        ram_gb = server.get('ram_gb', 0)
        if ram_gb <= 0:
            validation_results['errors'].append(f"{server_name}: Invalid RAM ({ram_gb} GB)")
        elif ram_gb > 1024:
            validation_results['warnings'].append(f"{server_name}: Very high RAM ({ram_gb} GB) - verify accuracy")
        
        # Storage validation
        storage_gb = server.get('storage_gb', 0)
        if storage_gb <= 0:
            validation_results['errors'].append(f"{server_name}: Invalid storage ({storage_gb} GB)")
        
        # Performance metrics validation
        cpu_util = server.get('peak_cpu_percent', 0)
        if cpu_util > 95:
            validation_results['warnings'].append(f"{server_name}: Very high CPU utilization ({cpu_util}%)")
        elif cpu_util < 10:
            validation_results['recommendations'].append(f"{server_name}: Low CPU utilization - consider smaller instance")
        
        # Architecture recommendations
        if cpu_cores >= 16 and ram_gb >= 64:
            validation_results['recommendations'].append(f"{server_name}: Large server - consider Aurora for better scalability")
        
        validation_results['valid'].append(server_name)
    
    return validation_results

# 3. Server comparison and analysis
def show_server_comparison():
    """Show server comparison and optimization suggestions"""
    if not st.session_state.on_prem_servers:
        return
    
    st.subheader("⚖️ Server Comparison & Optimization")
    
    servers = st.session_state.on_prem_servers
    
    # Create comparison DataFrame
    comparison_data = []
    for server in servers:
        comparison_data.append({
            'Server': server['server_name'],
            'CPU Cores': server['cpu_cores'],
            'RAM (GB)': server['ram_gb'],
            'Storage (GB)': server['storage_gb'],
            'CPU Util %': server['peak_cpu_percent'],
            'RAM Util %': server['peak_ram_percent'],
            'CPU/RAM Ratio': round(server['cpu_cores'] / max(server['ram_gb'], 1), 2),
            'Storage/RAM Ratio': round(server['storage_gb'] / max(server['ram_gb'], 1), 2)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Show comparison table with highlighting
    st.dataframe(
        comparison_df.style.background_gradient(subset=['CPU Util %', 'RAM Util %']),
        use_container_width=True
    )
    
    # Optimization suggestions
    st.markdown("**🎯 Optimization Suggestions:**")
    
    for i, server in enumerate(servers):
        suggestions = []
        
        if server['peak_cpu_percent'] < 30:
            suggestions.append("Consider smaller CPU allocation")
        if server['peak_ram_percent'] < 40:
            suggestions.append("Consider smaller RAM allocation")
        if server['cpu_cores'] / server['ram_gb'] > 1:
            suggestions.append("CPU-heavy workload - optimize for compute")
        if server['storage_gb'] / server['ram_gb'] > 100:
            suggestions.append("Storage-heavy workload - optimize for I/O")
        
        if suggestions:
            st.write(f"**{server['server_name']}:** {', '.join(suggestions)}")

# 4. Template generator for different scenarios
def generate_scenario_templates():
    """Generate templates for common migration scenarios"""
    st.subheader("📋 Scenario-Based Templates")
    
    scenario = st.selectbox(
        "Choose Migration Scenario",
        [
            "Oracle to Aurora PostgreSQL",
            "SQL Server to Aurora MySQL", 
            "MySQL to RDS MySQL",
            "PostgreSQL to Aurora PostgreSQL",
            "Multi-tier Application Stack",
            "Data Warehouse Migration",
            "OLTP High-Performance",
            "Mixed Workload Environment"
        ]
    )
    
    scenario_templates = {
        "Oracle to Aurora PostgreSQL": [
            {"server_name": "PROD-ORACLE-01", "cpu_cores": 16, "ram_gb": 64, "storage_gb": 1000, "database_engine": "oracle-ee"},
            {"server_name": "UAT-ORACLE-01", "cpu_cores": 8, "ram_gb": 32, "storage_gb": 500, "database_engine": "oracle-ee"},
            {"server_name": "DEV-ORACLE-01", "cpu_cores": 4, "ram_gb": 16, "storage_gb": 200, "database_engine": "oracle-ee"}
        ],
        "SQL Server to Aurora MySQL": [
            {"server_name": "PROD-SQLSRV-01", "cpu_cores": 12, "ram_gb": 48, "storage_gb": 800, "database_engine": "sqlserver-ee"},
            {"server_name": "UAT-SQLSRV-01", "cpu_cores": 6, "ram_gb": 24, "storage_gb": 400, "database_engine": "sqlserver-ee"}
        ],
        "Data Warehouse Migration": [
            {"server_name": "DW-PROD-01", "cpu_cores": 32, "ram_gb": 128, "storage_gb": 5000, "database_engine": "oracle-ee"},
            {"server_name": "DW-STAGE-01", "cpu_cores": 16, "ram_gb": 64, "storage_gb": 2000, "database_engine": "oracle-ee"}
        ]
    }
    
    if scenario in scenario_templates:
        template_servers = scenario_templates[scenario]
        
        st.markdown(f"**Template for {scenario}:**")
        template_df = pd.DataFrame(template_servers)
        st.dataframe(template_df, use_container_width=True)
        
        if st.button(f"📥 Load {scenario} Template"):
            st.session_state.on_prem_servers = template_servers
            st.success(f"✅ Loaded {len(template_servers)} servers from {scenario} template")
            st.rerun()

# 5. Advanced bulk operations
def show_advanced_bulk_operations():
    """Advanced bulk operations for server management"""
    if not st.session_state.on_prem_servers:
        return
    
    st.subheader("🔧 Advanced Bulk Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎛️ Batch Modifications**")
        
        # Bulk CPU adjustment
        cpu_multiplier = st.number_input(
            "CPU Multiplier", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0, 
            step=0.1,
            help="Multiply all CPU values by this factor"
        )
        
        if st.button("Apply CPU Multiplier"):
            for server in st.session_state.on_prem_servers:
                server['cpu_cores'] = int(server['cpu_cores'] * cpu_multiplier)
            st.success(f"✅ Applied {cpu_multiplier}x CPU multiplier to all servers")
            st.rerun()
    
    with col2:
        st.markdown("**🏷️ Batch Tagging**")
        
        # Add environment tags
        tag_environment = st.selectbox(
            "Tag Environment",
            ["PROD", "UAT", "DEV", "QA", "STAGE"]
        )
        
        tag_prefix = st.text_input("Server Name Prefix")
        
        if st.button("Apply Tags"):
            for server in st.session_state.on_prem_servers:
                if tag_prefix and not server['server_name'].startswith(tag_prefix):
                    server['server_name'] = f"{tag_prefix}-{server['server_name']}"
                server['environment'] = tag_environment
            st.success("✅ Applied tags to all servers")
            st.rerun()
    
    with col3:
        st.markdown("**🔍 Filtering & Selection**")
        
        # Filter by criteria
        min_cpu = st.number_input("Min CPU Cores", min_value=0, value=0)
        min_ram = st.number_input("Min RAM (GB)", min_value=0, value=0)
        
        if st.button("Filter Servers"):
            filtered_servers = [
                server for server in st.session_state.on_prem_servers
                if server['cpu_cores'] >= min_cpu and server['ram_gb'] >= min_ram
            ]
            
            st.info(f"Found {len(filtered_servers)} servers matching criteria")
            
            if filtered_servers and st.button("Keep Filtered Only"):
                st.session_state.on_prem_servers = filtered_servers
                st.success("✅ Kept only filtered servers")
                st.rerun()

# Add these functions to your existing TAB2 implementation
# Call them in appropriate sections of your TAB2 code

# Example integration:
"""
with tab2:
    st.header("🖥️ On-Premises Server Specifications")
    
    # ... existing code ...
    
    # Add enhanced features
    show_server_discovery_section()
    
    # ... existing bulk upload code ...
    
    if st.session_state.on_prem_servers:
        # Enhanced validation
        validation_results = enhanced_server_validation(st.session_state.on_prem_servers)
        
        if validation_results['errors']:
            st.error("❌ Validation Errors:")
            for error in validation_results['errors']:
                st.write(f"• {error}")
        
        if validation_results['warnings']:
            st.warning("⚠️ Warnings:")
            for warning in validation_results['warnings']:
                st.write(f"• {warning}")
        
        if validation_results['recommendations']:
            st.info("💡 Recommendations:")
            for rec in validation_results['recommendations']:
                st.write(f"• {rec}")
        
        # Show comparison and advanced operations
        show_server_comparison()
        show_advanced_bulk_operations()
    
    # Add scenario templates
    generate_scenario_templates()
"""

# ================================
# TAB 3: SIZING ANALYSIS (REWRITTEN)
# ================================

with tab3:
    st.header("📊 AWS RDS Sizing Analysis")
    
    # Check migration configuration
    if not st.session_state.migration_configured:
        st.warning("⚠️ Please configure migration settings in the Migration Planning tab first.")
        st.stop()
    
    # Get calculator instance
    calculator = st.session_state.calculator
    if not calculator:
        st.error("❌ Calculator not initialized. Please check your configuration.")
        st.stop()
    
    # ================================
    # SINGLE SERVER ANALYSIS
    # ================================
    
    if st.session_state.current_analysis_mode == 'single':
        st.subheader("🖥️ Single Server Analysis")
        
        # Check for server specification
        if 'current_server_spec' not in st.session_state:
            st.warning("⚠️ Please configure server specifications in the Server Specifications tab first.")
            
            # Quick server input for analysis
            with st.expander("⚡ Quick Server Configuration", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quick_server_name = st.text_input("Server Name", value="QUICK-SERVER-01", key="quick_server_name")
                    quick_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=8, key="quick_cores")
                    quick_ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=32, key="quick_ram")
                
                with col2:
                    quick_storage = st.number_input("Storage (GB)", min_value=10, value=500, key="quick_storage")
                    quick_cpu_util = st.number_input("Peak CPU %", min_value=1, max_value=100, value=75, key="quick_cpu_util")
                    quick_ram_util = st.number_input("Peak RAM %", min_value=1, max_value=100, value=80, key="quick_ram_util")
                
                with col3:
                    quick_iops = st.number_input("Max IOPS", min_value=100, value=2500, key="quick_iops")
                    quick_throughput = st.number_input("Max Throughput (MB/s)", min_value=10, value=125, key="quick_throughput")
                    quick_engine = st.selectbox("Database Engine", ["oracle-ee", "mysql", "postgres"], key="quick_engine")
                
                if st.button("🎯 Use Quick Configuration", type="primary", use_container_width=True):
                    st.session_state.current_server_spec = {
                        'server_name': quick_server_name,
                        'cores': quick_cores,
                        'cpu_ghz': 2.4,
                        'ram': quick_ram,
                        'ram_type': 'DDR4',
                        'storage': quick_storage,
                        'storage_type': 'SSD',
                        'cpu_util': quick_cpu_util,
                        'ram_util': quick_ram_util,
                        'max_iops': quick_iops,
                        'max_throughput_mbps': quick_throughput,
                        'max_connections': 500,
                        'growth_rate': 20,
                        'years': 3,
                        'enable_encryption': True,
                        'enable_perf_insights': True,
                        'enable_enhanced_monitoring': False,
                        'monthly_transfer_gb': 100
                    }
                    st.success(f"✅ Quick configuration set for {quick_server_name}")
                    st.rerun()
        else:
            # Display current server specification
            server_spec = st.session_state.current_server_spec
            
            st.subheader("🖥️ Current Server Configuration")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Server Name", server_spec.get('server_name', 'Unknown'))
                st.metric("CPU Cores", server_spec.get('cores', 0))
            
            with col2:
                st.metric("RAM (GB)", server_spec.get('ram', 0))
                st.metric("Storage (GB)", server_spec.get('storage', 0))
            
            with col3:
                st.metric("Peak CPU %", f"{server_spec.get('cpu_util', 0)}%")
                st.metric("Peak RAM %", f"{server_spec.get('ram_util', 0)}%")
            
            with col4:
                st.metric("Max IOPS", server_spec.get('max_iops', 0))
                st.metric("Throughput", f"{server_spec.get('max_throughput_mbps', 0)} MB/s")
            
            # Analysis Options
            st.subheader("⚙️ Analysis Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                growth_rate = st.slider("Annual Growth Rate (%)", 0, 50, server_spec.get('growth_rate', 20))
                analysis_years = st.selectbox("Analysis Period (Years)", [1, 3, 5], index=1)
            
            with col2:
                enable_encryption = st.checkbox("Enable Encryption", value=True)
                enable_perf_insights = st.checkbox("Enable Performance Insights", value=True)
            
            # Generate Analysis Button
            if st.button("🚀 Generate Sizing Recommendations", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing workload and generating recommendations..."):
                    start_time = time.time()
                    
                    try:
                        # Prepare inputs for analysis
                        inputs = {
                            "region": st.session_state.region,
                            "target_engine": st.session_state.target_engine,
                            "source_engine": st.session_state.source_engine,
                            "deployment": st.session_state.deployment_option,
                            "storage_type": st.session_state.storage_type,
                            "on_prem_cores": server_spec['cores'],
                            "peak_cpu_percent": server_spec['cpu_util'],
                            "on_prem_ram_gb": server_spec['ram'],
                            "peak_ram_percent": server_spec['ram_util'],
                            "storage_current_gb": server_spec['storage'],
                            "storage_growth_rate": growth_rate / 100,
                            "years": analysis_years,
                            "enable_encryption": enable_encryption,
                            "enable_perf_insights": enable_perf_insights,
                            "enable_enhanced_monitoring": False,
                            "monthly_data_transfer_gb": server_spec.get('monthly_transfer_gb', 100),
                            "max_iops": server_spec.get('max_iops', 1000),
                            "max_throughput_mbps": server_spec.get('max_throughput_mbps', 125),
                            "max_connections": server_spec.get('max_connections', 500)
                        }
                        
                        # Get environment configuration
                        selected_environments = st.session_state.get('selected_environments', ['PROD'])
                        env_configs = st.session_state.get('env_configs', {})
                        
                        # Generate recommendations
                        results = calculator.generate_comprehensive_recommendations(
                            inputs, 
                            selected_environments=selected_environments,
                            env_configs=env_configs
                        )
                        
                        # Store results
                        st.session_state.results = results
                        st.session_state.generation_time = time.time() - start_time
                        
                        # Generate AI insights if available
                        if calculator.ai_client:
                            with st.spinner("🤖 Generating AI insights..."):
                                try:
                                    ai_insights = asyncio.run(calculator.generate_ai_insights(results, inputs))
                                    st.session_state.ai_insights = ai_insights
                                except Exception as e:
                                    st.warning(f"AI insights generation failed: {e}")
                                    st.session_state.ai_insights = None
                        
                        st.success(f"✅ Analysis complete in {st.session_state.generation_time:.1f} seconds!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
                        st.code(traceback.format_exc())
    
    # ================================
    # BULK SERVER ANALYSIS
    # ================================
    
    elif st.session_state.current_analysis_mode == 'bulk':
        st.subheader("📊 Bulk Server Analysis")
        
        # Check for server data
        if not st.session_state.on_prem_servers:
            st.warning("⚠️ No servers configured for bulk analysis. Please add servers in the Server Specifications tab.")
            st.stop()
        
        servers = st.session_state.on_prem_servers
        st.info(f"📋 Ready to analyze {len(servers)} servers")
        
        # Display server summary
        st.subheader("📊 Server Summary")
        
        summary_data = []
        total_cores = 0
        total_ram = 0
        total_storage = 0
        
        for server in servers:
            total_cores += server['cpu_cores']
            total_ram += server['ram_gb']
            total_storage += server['storage_gb']
            
            summary_data.append({
                'Server Name': server['server_name'],
                'CPU Cores': server['cpu_cores'],
                'RAM (GB)': server['ram_gb'],
                'Storage (GB)': server['storage_gb'],
                'Database Engine': server['database_engine'],
                'Peak CPU %': server['peak_cpu_percent'],
                'Peak RAM %': server['peak_ram_percent']
            })
        
        # Show summary table
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Aggregate metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Servers", len(servers))
        
        with col2:
            st.metric("Total CPU Cores", total_cores)
        
        with col3:
            st.metric("Total RAM (GB)", f"{total_ram:,}")
        
        with col4:
            st.metric("Total Storage (GB)", f"{total_storage:,}")
        
        # Bulk Analysis Settings
        with st.expander("⚙️ Bulk Analysis Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=10, value=3, 
                                           help="Number of servers to process simultaneously")
                bulk_growth_rate = st.slider("Default Growth Rate (%)", 0, 50, 20)
            
            with col2:
                bulk_analysis_years = st.selectbox("Analysis Period (Years)", [1, 3, 5], index=1)
                enable_parallel_processing = st.checkbox("Enable Parallel Processing", value=True)
        
        # Start Bulk Analysis
        if st.button("🚀 Start Bulk Analysis", type="primary", use_container_width=True):
            # Validate calculator
            if not calculator:
                st.error("❌ Calculator not initialized. Please configure migration settings first.")
                st.stop()
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            results_placeholder = st.empty()
            
            bulk_results = {}
            total_monthly_cost_for_ai = 0
            
            try:
                total_servers = len(servers)
                
                for i, server in enumerate(servers):
                    # Update status
                    status_placeholder.text(f"🔄 Analyzing {server['server_name']} ({i+1}/{total_servers})")
                    
                    try:
                        # Prepare inputs for each server
                        inputs = {
                            "region": st.session_state.region,
                            "target_engine": st.session_state.target_engine,
                            "source_engine": server.get('database_engine', st.session_state.source_engine),
                            "deployment": st.session_state.deployment_option,
                            "storage_type": st.session_state.storage_type,
                            "on_prem_cores": server['cpu_cores'],
                            "peak_cpu_percent": server['peak_cpu_percent'],
                            "on_prem_ram_gb": server['ram_gb'],
                            "peak_ram_percent": server['peak_ram_percent'],
                            "storage_current_gb": server['storage_gb'],
                            "storage_growth_rate": bulk_growth_rate / 100,
                            "years": bulk_analysis_years,
                            "enable_encryption": True,
                            "enable_perf_insights": True,
                            "enable_enhanced_monitoring": False,
                            "monthly_data_transfer_gb": 100,
                            "max_iops": server['max_iops'],
                            "max_throughput_mbps": server['max_throughput_mbps']
                        }
                        
                        # Get environment configuration
                        selected_environments = st.session_state.get('selected_environments', ['PROD'])
                        env_configs = st.session_state.get('env_configs', {})
                        
                        # Handle server-specific environment configs if available
                        if 'environment_configs' in server:
                            server_env_configs = {}
                            for env in selected_environments:
                                if env in server['environment_configs']:
                                    server_env_configs[env] = server['environment_configs'][env]
                                else:
                                    server_env_configs[env] = env_configs.get(env, {
                                        'cpu_ratio': 1.0, 'ram_ratio': 1.0, 'storage_ratio': 1.0, 'deployment': 'Multi-AZ'
                                    })
                            env_configs_to_use = server_env_configs
                        else:
                            env_configs_to_use = env_configs
                        
                        # Generate recommendations for this server
                        server_results = calculator.generate_comprehensive_recommendations(
                            inputs,
                            selected_environments=selected_environments,
                            env_configs=env_configs_to_use
                        )
                        
                        bulk_results[server['server_name']] = server_results
                        
                        # Accumulate cost for AI insights
                        if 'error' not in server_results:
                            prod_result = server_results.get('PROD', list(server_results.values())[0])
                            if 'error' not in prod_result:
                                total_monthly_cost_for_ai += safe_get(prod_result, 'total_cost', 0)
                        
                        status_placeholder.success(f"✅ Completed {server['server_name']} ({i+1}/{total_servers})")
                        
                    except Exception as e:
                        bulk_results[server['server_name']] = {'error': str(e)}
                        st.warning(f"⚠️ Error analyzing {server['server_name']}: {e}")
                        status_placeholder.error(f"❌ Failed {server['server_name']} ({i+1}/{total_servers})")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_servers)
                    
                    # Show intermediate results
                    with results_placeholder.container():
                        completed_count = i + 1
                        successful_count = len([r for r in bulk_results.values() if 'error' not in r])
                        failed_count = completed_count - successful_count
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Completed", f"{completed_count}/{total_servers}")
                        with col2:
                            st.metric("Successful", successful_count)
                        with col3:
                            st.metric("Failed", failed_count)
                    
                    # Small delay for visibility
                    time.sleep(0.2)
                
                # Store results
                st.session_state.bulk_results = bulk_results
                
                successful_analyses = len([r for r in bulk_results.values() if 'error' not in r])
                failed_analyses = total_servers - successful_analyses
                
                progress_bar.progress(1.0)
                status_placeholder.success(f"🎉 Bulk analysis complete! {successful_analyses} successful, {failed_analyses} failed")
                results_placeholder.empty()
                
                # Generate AI insights for bulk analysis
                if successful_analyses > 0 and calculator.ai_client:
                    with st.spinner("🤖 Generating AI insights for bulk analysis..."):
                        try:
                            # Aggregate results for AI analysis
                            aggregated_results = {}
                            for server_name, server_data in bulk_results.items():
                                if 'error' not in server_data:
                                    if 'PROD' in server_data:
                                        aggregated_results[server_name] = server_data['PROD']
                                    else:
                                        for env_key, env_result in server_data.items():
                                            if 'error' not in env_result:
                                                aggregated_results[server_name] = env_result
                                                break
                            
                            # Create bulk inputs for AI
                            bulk_inputs = {
                                "region": st.session_state.region,
                                "target_engine": st.session_state.target_engine,
                                "source_engine": st.session_state.source_engine,
                                "deployment": st.session_state.deployment_option,
                                "storage_type": st.session_state.storage_type,
                                "num_servers_analyzed": successful_analyses,
                                "total_monthly_cost": total_monthly_cost_for_ai,
                                "analysis_mode": "bulk"
                            }
                            
                            # Generate AI insights
                            bulk_ai_insights = asyncio.run(calculator.generate_ai_insights(aggregated_results, bulk_inputs))
                            st.session_state.ai_insights = bulk_ai_insights
                            st.success("✅ AI insights generated!")
                            
                        except Exception as e:
                            st.warning(f"AI insights generation failed: {e}")
                            st.session_state.ai_insights = None
                
                st.success("🎉 Bulk analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Bulk analysis failed: {str(e)}")
                st.code(traceback.format_exc())
    
    # ================================
    # RESULTS DISPLAY
    # ================================
    
    # Display results if available
    if st.session_state.results or st.session_state.bulk_results:
        st.markdown("---")
        
        current_results = st.session_state.results if st.session_state.current_analysis_mode == 'single' else st.session_state.bulk_results
        
        # Call the enhanced results display function
        display_multi_environment_results(current_results, st.session_state.current_analysis_mode)
        
        # ================================
        # VISUALIZATIONS
        # ================================
        
        st.subheader("📈 Analysis Visualizations")
        
        # Cost heatmap for single analysis
        if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
            cost_heatmap = create_cost_heatmap(st.session_state.results)
            if cost_heatmap:
                st.plotly_chart(cost_heatmap, use_container_width=True)
            
            # Workload distribution if available
            if hasattr(calculator, 'migration_profile') and calculator.migration_profile:
                workload_chars = calculator.migration_profile.workload_characteristics
                if workload_chars:
                    workload_dist = create_workload_distribution_pie(workload_chars)
                    if workload_dist:
                        st.plotly_chart(workload_dist, use_container_width=True)
        
        # Bulk analysis summary chart
        elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
            bulk_summary_chart = create_bulk_analysis_summary_chart(st.session_state.bulk_results)
            if bulk_summary_chart:
                st.plotly_chart(bulk_summary_chart, use_container_width=True)
        
        # ================================
        # EXPORT OPTIONS
        # ================================
        
        st.subheader("📊 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export to CSV", use_container_width=True):
                if st.session_state.current_analysis_mode == 'single':
                    # Export single server results
                    export_data = []
                    valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
                    
                    for env_name, result in valid_results.items():
                        export_data.append({
                            'Environment': env_name,
                            'Instance Type': safe_get_str(result.get('writer', result), 'instance_type', 'N/A'),
                            'vCPUs': safe_get(result.get('writer', result), 'actual_vCPUs', 0),
                            'RAM (GB)': safe_get(result.get('writer', result), 'actual_RAM_GB', 0),
                            'Storage (GB)': safe_get(result, 'storage_GB', 0),
                            'Monthly Cost': safe_get(result, 'total_cost', 0),
                            'Annual Cost': safe_get(result, 'total_cost', 0) * 12
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv_data = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"single_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                else:
                    # Export bulk results
                    bulk_export_data = []
                    
                    for server_name, server_results in st.session_state.bulk_results.items():
                        if 'error' not in server_results:
                            result = server_results.get('PROD', list(server_results.values())[0])
                            if 'error' not in result:
                                bulk_export_data.append({
                                    'Server Name': server_name,
                                    'Instance Type': safe_get_str(result.get('writer', result), 'instance_type', 'N/A'),
                                    'vCPUs': safe_get(result.get('writer', result), 'actual_vCPUs', 0),
                                    'RAM (GB)': safe_get(result.get('writer', result), 'actual_RAM_GB', 0),
                                    'Storage (GB)': safe_get(result, 'storage_GB', 0),
                                    'Monthly Cost': safe_get(result, 'total_cost', 0),
                                    'Annual Cost': safe_get(result, 'total_cost', 0) * 12
                                })
                    
                    bulk_export_df = pd.DataFrame(bulk_export_data)
                    bulk_csv_data = bulk_export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Bulk CSV",
                        data=bulk_csv_data,
                        file_name=f"bulk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col2:
            if st.button("📊 Export to JSON", use_container_width=True):
                export_json = {
                    'analysis_mode': st.session_state.current_analysis_mode,
                    'migration_config': {
                        'source_engine': st.session_state.source_engine,
                        'target_engine': st.session_state.target_engine,
                        'deployment': st.session_state.deployment_option,
                        'region': st.session_state.region
                    },
                    'results': current_results,
                    'generated_at': datetime.now().isoformat()
                }
                
                json_data = json.dumps(export_json, indent=2, default=str)
                
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📋 Copy Summary", use_container_width=True):
                if st.session_state.current_analysis_mode == 'single':
                    valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
                    if valid_results:
                        prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                        summary_text = f"""AWS RDS Migration Analysis Summary
Server: {st.session_state.current_server_spec.get('server_name', 'Unknown')}
Migration: {st.session_state.source_engine} → {st.session_state.target_engine}
Recommended Instance: {safe_get_str(prod_result.get('writer', prod_result), 'instance_type', 'N/A')}
Monthly Cost: ${safe_get(prod_result, 'total_cost', 0):,.2f}
Annual Cost: ${safe_get(prod_result, 'total_cost', 0) * 12:,.2f}"""
                        
                        # Show summary in a text area for copying
                        st.text_area("Copy this summary:", summary_text, height=150)
                
                else:
                    successful_count = len([r for r in st.session_state.bulk_results.values() if 'error' not in r])
                    total_cost = sum(safe_get(result.get('PROD', {}), 'total_cost', 0) 
                                   for result in st.session_state.bulk_results.values() if 'error' not in result)
                    
                    bulk_summary_text = f"""AWS RDS Bulk Migration Analysis Summary
Total Servers: {len(st.session_state.bulk_results)}
Successful Analyses: {successful_count}
Migration: {st.session_state.source_engine} → {st.session_state.target_engine}
Total Monthly Cost: ${total_cost:,.2f}
Total Annual Cost: ${total_cost * 12:,.2f}
Average Cost per Server: ${total_cost / max(successful_count, 1):,.2f}"""
                    
                    st.text_area("Copy this summary:", bulk_summary_text, height=150)
    
    else:
        # No results available
        st.info("💡 Configure your migration settings and server specifications, then click 'Generate Sizing Recommendations' to see analysis results.")
# ================================
# HELPER FUNCTIONS FOR TAB 4
# ================================

def create_environment_cost_comparison_chart(results, analysis_mode):
    """
    Create comprehensive environment cost comparison visualizations
    """
    if analysis_mode == 'single':
        # Single server environment comparison
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if not valid_results:
            return None
        
        # Environment costs data
        env_names = list(valid_results.keys())
        env_costs = [safe_get(result, 'total_cost', 0) for result in valid_results.values()]
        env_colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0', '#F44336'][:len(env_names)]
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Cost by Environment',
                'Cost Breakdown by Component', 
                'Environment Sizing Comparison',
                'Annual Cost Projection'
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Monthly cost by environment
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=env_costs,
                name='Monthly Cost',
                marker_color=env_colors,
                text=[f'${cost:,.0f}' for cost in env_costs],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Cost breakdown pie (using PROD as example)
        prod_result = valid_results.get('PROD', list(valid_results.values())[0])
        cost_breakdown = safe_get(prod_result, 'cost_breakdown', {})
        
        if cost_breakdown:
            breakdown_labels = []
            breakdown_values = []
            
            if 'writer_monthly' in cost_breakdown:
                breakdown_labels.extend(['Writer Instance', 'Reader Instances', 'Storage', 'Backup'])
                breakdown_values.extend([
                    safe_get(cost_breakdown, 'writer_monthly', 0),
                    safe_get(cost_breakdown, 'readers_monthly', 0),
                    safe_get(cost_breakdown, 'storage_monthly', 0),
                    safe_get(cost_breakdown, 'backup_monthly', 0)
                ])
            else:
                breakdown_labels.extend(['Database Instance', 'Storage', 'Backup', 'Transfer'])
                breakdown_values.extend([
                    safe_get(cost_breakdown, 'instance_monthly', 0),
                    safe_get(cost_breakdown, 'storage_monthly', 0),
                    safe_get(cost_breakdown, 'backup_monthly', 0),
                    safe_get(cost_breakdown, 'transfer_monthly', 50)
                ])
            
            # Filter out zero values
            filtered_labels = []
            filtered_values = []
            for label, value in zip(breakdown_labels, breakdown_values):
                if value > 0:
                    filtered_labels.append(label)
                    filtered_values.append(value)
            
            if filtered_labels:
                fig.add_trace(
                    go.Pie(
                        labels=filtered_labels,
                        values=filtered_values,
                        name="Cost Breakdown"
                    ),
                    row=1, col=2
                )
        
        # 3. Environment sizing comparison
        vcpus_by_env = []
        ram_by_env = []
        
        for result in valid_results.values():
            if 'writer' in result:
                vcpus_by_env.append(safe_get(result['writer'], 'actual_vCPUs', 0))
                ram_by_env.append(safe_get(result['writer'], 'actual_RAM_GB', 0))
            else:
                vcpus_by_env.append(safe_get(result, 'actual_vCPUs', 0))
                ram_by_env.append(safe_get(result, 'actual_RAM_GB', 0))
        
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=vcpus_by_env,
                name='vCPUs',
                marker_color='lightblue',
                text=vcpus_by_env,
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # 4. Annual cost projection
        annual_costs = [cost * 12 for cost in env_costs]
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=annual_costs,
                name='Annual Cost',
                marker_color='lightcoral',
                text=[f'${cost:,.0f}' for cost in annual_costs],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🏢 Multi-Environment Cost Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    else:  # Bulk analysis
        # Aggregate by environment across all servers
        env_aggregates = {}
        
        for server_name, server_results in results.items():
            if 'error' not in server_results:
                for env_name, env_result in server_results.items():
                    if 'error' not in env_result:
                        if env_name not in env_aggregates:
                            env_aggregates[env_name] = {
                                'total_cost': 0,
                                'server_count': 0,
                                'total_vcpus': 0,
                                'total_ram': 0
                            }
                        
                        env_aggregates[env_name]['total_cost'] += safe_get(env_result, 'total_cost', 0)
                        env_aggregates[env_name]['server_count'] += 1
                        
                        if 'writer' in env_result:
                            env_aggregates[env_name]['total_vcpus'] += safe_get(env_result['writer'], 'actual_vCPUs', 0)
                            env_aggregates[env_name]['total_ram'] += safe_get(env_result['writer'], 'actual_RAM_GB', 0)
                        else:
                            env_aggregates[env_name]['total_vcpus'] += safe_get(env_result, 'actual_vCPUs', 0)
                            env_aggregates[env_name]['total_ram'] += safe_get(env_result, 'actual_RAM_GB', 0)
        
        if not env_aggregates:
            return None
        
        # Create bulk environment comparison
        env_names = sorted(env_aggregates.keys())
        env_costs = [env_aggregates[env]['total_cost'] for env in env_names]
        server_counts = [env_aggregates[env]['server_count'] for env in env_names]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Total Cost by Environment (All Servers)',
                'Server Count by Environment',
                'Average Cost per Server by Environment', 
                'Total Resource Allocation'
            )
        )
        
        # Environment colors
        env_colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0', '#F44336'][:len(env_names)]
        
        # 1. Total cost by environment
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=env_costs,
                name='Total Cost',
                marker_color=env_colors,
                text=[f'${cost:,.0f}' for cost in env_costs],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Server count by environment
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=server_counts,
                name='Server Count',
                marker_color='lightgreen',
                text=server_counts,
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Average cost per server
        avg_costs = [cost/count if count > 0 else 0 for cost, count in zip(env_costs, server_counts)]
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=avg_costs,
                name='Avg Cost/Server',
                marker_color='lightblue',
                text=[f'${cost:,.0f}' for cost in avg_costs],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # 4. Total vCPUs by environment
        total_vcpus = [env_aggregates[env]['total_vcpus'] for env in env_names]
        fig.add_trace(
            go.Bar(
                x=env_names,
                y=total_vcpus,
                name='Total vCPUs',
                marker_color='lightyellow',
                text=total_vcpus,
                textposition='outside'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🏢 Bulk Multi-Environment Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig

def create_environment_cost_waterfall(results, analysis_mode):
    """
    Create waterfall chart showing cost progression across environments
    """
    if analysis_mode == 'single':
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if not valid_results:
            return None
        
        # Sort environments by typical progression
        env_order = ['DEV', 'QA', 'UAT', 'PREPROD', 'PROD']
        ordered_envs = [env for env in env_order if env in valid_results]
        
        # Calculate cumulative costs
        env_costs = [safe_get(valid_results[env], 'total_cost', 0) for env in ordered_envs]
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Environment Costs",
            orientation="v",
            measure=["relative"] * len(ordered_envs),
            x=ordered_envs,
            textposition="outside",
            text=[f"${cost:,.0f}" for cost in env_costs],
            y=env_costs,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title="💧 Environment Cost Waterfall - Monthly Progression",
            showlegend=False,
            height=400,
            yaxis_title="Monthly Cost ($)"
        )
        
        return fig
    
    return None

def display_multi_environment_results(results, analysis_mode):
    """
    Display comprehensive multi-environment results
    """
    if analysis_mode == 'single':
        st.subheader("🖥️ Single Server Analysis Results")
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        
        if failed_results:
            st.warning(f"⚠️ {len(failed_results)} environment(s) failed analysis: {', '.join(failed_results.keys())}")
        
        if not valid_results:
            st.error("❌ No successful analyses to display")
            return
        
        # Display results for each environment
        for env_name, result in valid_results.items():
            with st.expander(f"🌐 {env_name} Environment Details", expanded=(env_name == 'PROD')):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_cost = safe_get(result, 'total_cost', 0)
                    st.metric("Monthly Cost", f"${total_cost:,.2f}")
                    st.metric("Annual Cost", f"${total_cost * 12:,.2f}")
                
                with col2:
                    if 'writer' in result:
                        writer = result['writer']
                        st.metric("Writer Instance", safe_get_str(writer, 'instance_type', 'N/A'))
                        st.metric("Writer Resources", f"{safe_get(writer, 'actual_vCPUs', 0)} vCPUs, {safe_get(writer, 'actual_RAM_GB', 0)} GB")
                    else:
                        st.metric("Instance Type", safe_get_str(result, 'instance_type', 'N/A'))
                        st.metric("Resources", f"{safe_get(result, 'actual_vCPUs', 0)} vCPUs, {safe_get(result, 'actual_RAM_GB', 0)} GB")
                
                with col3:
                    storage_gb = safe_get(result, 'storage_GB', 0)
                    st.metric("Storage", f"{storage_gb:,} GB")
                    
                    if result.get('readers'):
                        st.metric("Read Replicas", f"{len(result['readers'])} instances")
                    
                # Cost breakdown
                cost_breakdown = safe_get(result, 'cost_breakdown', {})
                if cost_breakdown:
                    st.markdown("**💰 Cost Breakdown:**")
                    breakdown_col1, breakdown_col2 = st.columns(2)
                    
                    with breakdown_col1:
                        if 'writer_monthly' in cost_breakdown:
                            st.write(f"• Writer: ${safe_get(cost_breakdown, 'writer_monthly', 0):,.2f}")
                            st.write(f"• Readers: ${safe_get(cost_breakdown, 'readers_monthly', 0):,.2f}")
                        else:
                            st.write(f"• Instance: ${safe_get(cost_breakdown, 'instance_monthly', 0):,.2f}")
                    
                    with breakdown_col2:
                        st.write(f"• Storage: ${safe_get(cost_breakdown, 'storage_monthly', 0):,.2f}")
                        st.write(f"• Backup: ${safe_get(cost_breakdown, 'backup_monthly', 0):,.2f}")
    
    else:  # Bulk analysis
        st.subheader("📊 Bulk Analysis Results")
        
        total_servers = len(results)
        successful_servers = sum(1 for result in results.values() if 'error' not in result)
        failed_servers = total_servers - successful_servers
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Servers", total_servers)
        with col2:
            st.metric("Successful", successful_servers)
        with col3:
            st.metric("Failed", failed_servers)
        
        if failed_servers > 0:
            failed_list = [name for name, result in results.items() if 'error' in result]
            st.warning(f"⚠️ Failed servers: {', '.join(failed_list[:5])}{'...' if len(failed_list) > 5 else ''}")
        
        # Summary table
        if successful_servers > 0:
            summary_data = []
            
            for server_name, server_results in results.items():
                if 'error' not in server_results:
                    # Get PROD result or first available result
                    prod_result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in prod_result:
                        summary_data.append({
                            'Server Name': server_name,
                            'Instance Type': safe_get_str(prod_result.get('writer', prod_result), 'instance_type', 'N/A'),
                            'Monthly Cost': f"${safe_get(prod_result, 'total_cost', 0):,.2f}",
                            'vCPUs': safe_get(prod_result.get('writer', prod_result), 'actual_vCPUs', 0),
                            'RAM (GB)': safe_get(prod_result.get('writer', prod_result), 'actual_RAM_GB', 0),
                            'Storage (GB)': safe_get(prod_result, 'storage_GB', 0)
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Calculate totals
                total_monthly = sum(float(row['Monthly Cost'].replace('

with tab4:
    st.header("💰 Comprehensive Financial Analysis")
    
    # Check if we have analysis results
    current_results = None
    analysis_mode = None
    
    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_results = st.session_state.results
        analysis_mode = 'single'
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        current_results = st.session_state.bulk_results
        analysis_mode = 'bulk'
    
    if not current_results:
        st.warning("⚠️ Please run an analysis first to view financial insights.")
        st.info("💡 Go to the **Sizing Analysis** tab to generate recommendations.")
        st.stop()
    
    # ================================
    # FINANCIAL OVERVIEW METRICS
    # ================================
    
    st.subheader("📊 Financial Overview")
    
    # Calculate key financial metrics
    if analysis_mode == 'single':
        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
        
        # Environment-based calculations
        total_monthly_all_envs = sum(safe_get(result, 'total_cost', 0) for result in valid_results.values())
        prod_monthly = safe_get(valid_results.get('PROD', {}), 'total_cost', 0)
        
        # Cost breakdown for PROD environment
        prod_result = valid_results.get('PROD', list(valid_results.values())[0] if valid_results else {})
        cost_breakdown = safe_get(prod_result, 'cost_breakdown', {})
        
        # Extract compute and storage costs
        if 'writer_monthly' in cost_breakdown:
            # Aurora/Cluster setup
            compute_cost = safe_get(cost_breakdown, 'writer_monthly', 0) + safe_get(cost_breakdown, 'readers_monthly', 0)
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', 0)
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', 0)
        else:
            # Standard RDS
            compute_cost = safe_get(cost_breakdown, 'instance_monthly', 0)
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', 0)
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', 0)
        
        transfer_cost = safe_get(cost_breakdown, 'transfer_monthly', 50)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${prod_monthly:,.0f}</div>
                <div class="metric-label">PROD Monthly Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${total_monthly_all_envs:,.0f}</div>
                <div class="metric-label">All Environments</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            annual_cost = prod_monthly * 12
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${annual_cost:,.0f}</div>
                <div class="metric-label">PROD Annual Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            three_year_cost = prod_monthly * 36
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${three_year_cost:,.0f}</div>
                <div class="metric-label">3-Year TCO (PROD)</div>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Bulk analysis
        # Aggregate bulk metrics
        total_servers = len(current_results)
        successful_servers = sum(1 for result in current_results.values() if 'error' not in result)
        
        # Calculate total costs across all servers and environments
        total_monthly_bulk = 0
        environment_totals = {}
        
        for server_name, server_results in current_results.items():
            if 'error' not in server_results:
                for env_name, env_result in server_results.items():
                    if 'error' not in env_result:
                        cost = safe_get(env_result, 'total_cost', 0)
                        total_monthly_bulk += cost
                        
                        if env_name not in environment_totals:
                            environment_totals[env_name] = 0
                        environment_totals[env_name] += cost
        
        # Display bulk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{successful_servers}</div>
                <div class="metric-label">Successful Analyses</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${total_monthly_bulk:,.0f}</div>
                <div class="metric-label">Total Monthly Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_cost_per_server = total_monthly_bulk / max(successful_servers, 1)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${avg_cost_per_server:,.0f}</div>
                <div class="metric-label">Avg Cost/Server</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            annual_bulk_cost = total_monthly_bulk * 12
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${annual_bulk_cost:,.0f}</div>
                <div class="metric-label">Total Annual Cost</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ================================
    # COST VISUALIZATION CHARTS
    # ================================
    
    st.subheader("📈 Cost Analysis Visualizations")
    
    # Create cost visualization based on analysis mode
    if analysis_mode == 'single':
        # Environment cost comparison chart
        cost_comparison_fig = create_environment_cost_comparison_chart(current_results, analysis_mode)
        if cost_comparison_fig:
            st.plotly_chart(cost_comparison_fig, use_container_width=True)
        
        # Cost heatmap
        cost_heatmap = create_cost_heatmap(current_results)
        if cost_heatmap:
            st.plotly_chart(cost_heatmap, use_container_width=True)
        
        # Cost waterfall chart
        waterfall_fig = create_environment_cost_waterfall(current_results, analysis_mode)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
    
    else:  # Bulk analysis
        # Bulk summary chart
        bulk_summary_chart = create_bulk_analysis_summary_chart(current_results)
        if bulk_summary_chart:
            st.plotly_chart(bulk_summary_chart, use_container_width=True)
        
        # Environment comparison for bulk
        bulk_env_fig = create_environment_cost_comparison_chart(current_results, analysis_mode)
        if bulk_env_fig:
            st.plotly_chart(bulk_env_fig, use_container_width=True)
    
    # ================================
    # DETAILED COST BREAKDOWN
    # ================================
    
    st.subheader("🔍 Detailed Cost Breakdown")
    
    if analysis_mode == 'single':
        # Environment-by-environment breakdown
        for env_name, result in valid_results.items():
            with st.expander(f"💰 {env_name} Environment Cost Details", expanded=(env_name == 'PROD')):
                cost_breakdown = safe_get(result, 'cost_breakdown', {})
                total_cost = safe_get(result, 'total_cost', 0)
                
                # Create two columns for cost details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**💻 Compute Costs**")
                    
                    if 'writer_monthly' in cost_breakdown:
                        # Aurora/Cluster breakdown
                        writer_cost = safe_get(cost_breakdown, 'writer_monthly', 0)
                        readers_cost = safe_get(cost_breakdown, 'readers_monthly', 0)
                        
                        st.write(f"• Writer Instance: ${writer_cost:,.2f}/month")
                        st.write(f"• Reader Instances: ${readers_cost:,.2f}/month")
                        st.write(f"• **Total Compute: ${writer_cost + readers_cost:,.2f}/month**")
                        
                        # Instance details
                        if 'writer' in result:
                            writer_info = result['writer']
                            st.write(f"• Writer Type: {safe_get_str(writer_info, 'instance_type', 'N/A')}")
                            st.write(f"• Writer Resources: {safe_get(writer_info, 'actual_vCPUs', 0)} vCPUs, {safe_get(writer_info, 'actual_RAM_GB', 0)} GB RAM")
                        
                        if result.get('readers'):
                            reader_count = len(result['readers'])
                            st.write(f"• Readers: {reader_count} x {safe_get_str(result['readers'][0], 'instance_type', 'N/A')}")
                    
                    else:
                        # Standard RDS breakdown
                        instance_cost = safe_get(cost_breakdown, 'instance_monthly', 0)
                        st.write(f"• Database Instance: ${instance_cost:,.2f}/month")
                        st.write(f"• Instance Type: {safe_get_str(result, 'instance_type', 'N/A')}")
                        st.write(f"• Resources: {safe_get(result, 'actual_vCPUs', 0)} vCPUs, {safe_get(result, 'actual_RAM_GB', 0)} GB RAM")
                
                with col2:
                    st.markdown("**💾 Storage & Data Costs**")
                    
                    storage_cost = safe_get(cost_breakdown, 'storage_monthly', 0)
                    backup_cost = safe_get(cost_breakdown, 'backup_monthly', 0)
                    transfer_cost = safe_get(cost_breakdown, 'transfer_monthly', 50)
                    
                    st.write(f"• Primary Storage: ${storage_cost:,.2f}/month")
                    st.write(f"• Backup Storage: ${backup_cost:,.2f}/month")
                    st.write(f"• Data Transfer: ${transfer_cost:,.2f}/month")
                    st.write(f"• **Total Storage: ${storage_cost + backup_cost + transfer_cost:,.2f}/month**")
                    
                    # Storage details
                    storage_gb = safe_get(result, 'storage_GB', 0)
                    st.write(f"• Storage Capacity: {storage_gb:,} GB")
                    if storage_gb > 0:
                        cost_per_gb = storage_cost / storage_gb
                        st.write(f"• Cost per GB: ${cost_per_gb:.4f}/month")
                
                # Summary for this environment
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Monthly Total", f"${total_cost:,.2f}")
                
                with col2:
                    st.metric("Annual Total", f"${total_cost * 12:,.2f}")
                
                with col3:
                    if total_monthly_all_envs > 0:
                        percentage = (total_cost / total_monthly_all_envs) * 100
                        st.metric("% of Total", f"{percentage:.1f}%")
    
    else:  # Bulk analysis breakdown
        # Server-by-server cost summary
        st.markdown("**📊 Server Cost Summary**")
        
        # Create summary table
        server_cost_data = []
        
        for server_name, server_results in current_results.items():
            if 'error' not in server_results:
                # Calculate total cost across all environments for this server
                server_total = 0
                env_costs = {}
                
                for env_name, env_result in server_results.items():
                    if 'error' not in env_result:
                        cost = safe_get(env_result, 'total_cost', 0)
                        server_total += cost
                        env_costs[env_name] = cost
                
                # Add to summary data
                row_data = {
                    'Server Name': server_name,
                    'Total Monthly': f"${server_total:,.2f}",
                    'Total Annual': f"${server_total * 12:,.2f}"
                }
                
                # Add environment-specific costs
                for env in ['DEV', 'QA', 'UAT', 'PREPROD', 'PROD']:
                    if env in env_costs:
                        row_data[f'{env} Monthly'] = f"${env_costs[env]:,.2f}"
                
                server_cost_data.append(row_data)
        
        if server_cost_data:
            server_cost_df = pd.DataFrame(server_cost_data)
            st.dataframe(server_cost_df, use_container_width=True)
        
        # Environment aggregation
        st.markdown("**🏢 Environment Cost Aggregation**")
        
        if environment_totals:
            env_agg_data = []
            
            for env_name in sorted(environment_totals.keys()):
                env_total = environment_totals[env_name]
                server_count_in_env = sum(1 for server_results in current_results.values() 
                                        if 'error' not in server_results and env_name in server_results and 'error' not in server_results[env_name])
                
                env_agg_data.append({
                    'Environment': env_name,
                    'Server Count': server_count_in_env,
                    'Total Monthly': f"${env_total:,.2f}",
                    'Total Annual': f"${env_total * 12:,.2f}",
                    'Avg per Server': f"${env_total / max(server_count_in_env, 1):,.2f}",
                    '% of Total': f"{(env_total / max(total_monthly_bulk, 1)) * 100:.1f}%"
                })
            
            env_agg_df = pd.DataFrame(env_agg_data)
            st.dataframe(env_agg_df, use_container_width=True)
    
    # ================================
    # COST OPTIMIZATION RECOMMENDATIONS
    # ================================
    
    st.subheader("💡 Cost Optimization Recommendations")
    
    # Reserved Instance Savings Calculator
    with st.expander("💰 Reserved Instance Savings Calculator", expanded=True):
        st.markdown("**Calculate potential savings with Reserved Instances:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ri_term = st.selectbox("Reserved Instance Term", ["1 Year", "3 Years"], index=1)
            ri_payment = st.selectbox("Payment Option", ["No Upfront", "Partial Upfront", "All Upfront"], index=1)
        
        with col2:
            # Savings percentages based on typical AWS RI discounts
            ri_savings_map = {
                ("1 Year", "No Upfront"): 0.25,
                ("1 Year", "Partial Upfront"): 0.35,
                ("1 Year", "All Upfront"): 0.40,
                ("3 Years", "No Upfront"): 0.35,
                ("3 Years", "Partial Upfront"): 0.45,
                ("3 Years", "All Upfront"): 0.50
            }
            
            savings_percentage = ri_savings_map.get((ri_term, ri_payment), 0.40)
            
            if analysis_mode == 'single':
                base_annual_cost = prod_monthly * 12
                ri_savings_annual = base_annual_cost * savings_percentage
                ri_cost_after = base_annual_cost - ri_savings_annual
                
                st.metric("Potential Annual Savings", f"${ri_savings_annual:,.0f}")
                st.metric("Cost After RI", f"${ri_cost_after:,.0f}")
            else:
                prod_total = environment_totals.get('PROD', 0)
                base_annual_cost_bulk = prod_total * 12
                ri_savings_annual_bulk = base_annual_cost_bulk * savings_percentage
                ri_cost_after_bulk = base_annual_cost_bulk - ri_savings_annual_bulk
                
                st.metric("PROD Annual Savings", f"${ri_savings_annual_bulk:,.0f}")
                st.metric("PROD Cost After RI", f"${ri_cost_after_bulk:,.0f}")
        
        with col3:
            st.info(f"**Estimated Savings: {savings_percentage*100:.0f}%**")
            st.write(f"• Term: {ri_term}")
            st.write(f"• Payment: {ri_payment}")
            st.write("• Based on typical AWS RDS RI discounts")
    
    # Environment-specific optimization suggestions
    optimization_suggestions = []
    
    if analysis_mode == 'single':
        for env_name, result in valid_results.items():
            total_cost = safe_get(result, 'total_cost', 0)
            
            if env_name in ['DEV', 'QA']:
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Use smaller instances or stop during off-hours',
                    'Potential Savings': f"${total_cost * 0.4:,.0f}/month (40%)",
                    'Implementation': 'Schedule start/stop, use burstable instances',
                    'Risk Level': '🟢 Low'
                })
            elif env_name == 'UAT':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Implement scheduled scaling for business hours only',
                    'Potential Savings': f"${total_cost * 0.25:,.0f}/month (25%)",
                    'Implementation': 'AWS Instance Scheduler, Lambda automation',
                    'Risk Level': '🟢 Low'
                })
            elif env_name == 'PREPROD':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Optimize storage IOPS and enable compression',
                    'Potential Savings': f"${total_cost * 0.15:,.0f}/month (15%)",
                    'Implementation': 'Right-size storage, enable compression features',
                    'Risk Level': '🟡 Medium'
                })
            elif env_name == 'PROD':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Purchase Reserved Instances for predictable workloads',
                    'Potential Savings': f"${total_cost * 0.4:,.0f}/month (40%)",
                    'Implementation': '1-3 year RI commitment with upfront payment',
                    'Risk Level': '🟢 Low'
                })
    
    else:  # Bulk recommendations
        for env_name, env_total in environment_totals.items():
            if env_name in ['DEV', 'QA']:
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Implement bulk stop/start scheduling',
                    'Potential Savings': f"${env_total * 0.5:,.0f}/month (50%)",
                    'Implementation': 'Automated scheduling across all servers',
                    'Risk Level': '🟢 Low'
                })
            elif env_name == 'PROD':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Bulk Reserved Instance purchasing',
                    'Potential Savings': f"${env_total * 0.4:,.0f}/month (40%)",
                    'Implementation': 'Coordinated RI strategy across all PROD servers',
                    'Risk Level': '🟢 Low'
                })
    
    if optimization_suggestions:
        st.markdown("**🎯 Environment-Specific Optimization Opportunities:**")
        opt_df = pd.DataFrame(optimization_suggestions)
        st.dataframe(opt_df, use_container_width=True)
        
        # Calculate total optimization potential
        total_potential_savings = 0
        for suggestion in optimization_suggestions:
            savings_str = suggestion['Potential Savings']
            # Extract numeric value (rough estimation)
            import re
            savings_match = re.search(r'\$([0-9,]+)', savings_str)
            if savings_match:
                savings_value = float(savings_match.group(1).replace(',', ''))
                total_potential_savings += savings_value
        
        if total_potential_savings > 0:
            st.success(f"💰 **Total Optimization Potential: ${total_potential_savings:,.0f}/month (${total_potential_savings * 12:,.0f}/year)**")
    
    # ================================
    # TCO COMPARISON & ROI ANALYSIS
    # ================================
    
    st.subheader("📊 TCO Analysis & ROI Projection")
    
    with st.expander("🏢 Total Cost of Ownership (TCO) Analysis", expanded=False):
        st.markdown("**Compare AWS costs against current on-premises infrastructure:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏢 Current On-Premises Costs (Annual)**")
            
            # Input fields for on-premises costs
            onprem_hardware = st.number_input("Hardware/Licensing", min_value=0, value=50000, step=1000, help="Annual hardware refresh and software licensing")
            onprem_personnel = st.number_input("Personnel Costs", min_value=0, value=120000, step=1000, help="DBA and infrastructure team costs")
            onprem_facilities = st.number_input("Facilities/Power", min_value=0, value=15000, step=1000, help="Data center, power, cooling costs")
            onprem_backup = st.number_input("Backup/DR", min_value=0, value=25000, step=1000, help="Backup systems and disaster recovery")
            onprem_maintenance = st.number_input("Maintenance/Support", min_value=0, value=30000, step=1000, help="Vendor support and maintenance contracts")
            
            total_onprem_annual = onprem_hardware + onprem_personnel + onprem_facilities + onprem_backup + onprem_maintenance
            
            st.metric("**Total On-Premises Annual**", f"${total_onprem_annual:,.0f}")
        
        with col2:
            st.markdown("**☁️ AWS Cloud Costs (Annual)**")
            
            if analysis_mode == 'single':
                aws_infrastructure_annual = prod_monthly * 12
            else:
                aws_infrastructure_annual = environment_totals.get('PROD', 0) * 12
            
            # Estimated additional AWS costs
            aws_migration_onetime = st.number_input("Migration Costs (One-time)", min_value=0, value=int(aws_infrastructure_annual * 0.2), step=1000)
            aws_training_annual = st.number_input("Training/Upskilling", min_value=0, value=15000, step=1000)
            aws_management_annual = st.number_input("Cloud Management Tools", min_value=0, value=int(aws_infrastructure_annual * 0.05), step=1000)
            
            total_aws_annual = aws_infrastructure_annual + aws_training_annual + aws_management_annual
            
            st.metric("Infrastructure (Annual)", f"${aws_infrastructure_annual:,.0f}")
            st.metric("**Total AWS Annual**", f"${total_aws_annual:,.0f}")
            st.metric("Migration Investment", f"${aws_migration_onetime:,.0f}")
        
        # TCO Comparison
        st.markdown("---")
        st.markdown("**📊 3-Year TCO Comparison**")
        
        # Calculate 3-year costs
        three_year_onprem = total_onprem_annual * 3
        three_year_aws = (total_aws_annual * 3) + aws_migration_onetime
        tco_savings = three_year_onprem - three_year_aws
        roi_percentage = (tco_savings / three_year_aws) * 100 if three_year_aws > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("On-Premises 3-Year", f"${three_year_onprem:,.0f}")
        
        with col2:
            st.metric("AWS 3-Year", f"${three_year_aws:,.0f}")
        
        with col3:
            if tco_savings > 0:
                st.metric("💰 Total Savings", f"${tco_savings:,.0f}", delta=f"{(tco_savings/three_year_onprem)*100:.1f}%")
            else:
                st.metric("💰 Additional Cost", f"${abs(tco_savings):,.0f}")
        
        with col4:
            if roi_percentage > 0:
                st.metric("📈 ROI", f"{roi_percentage:.1f}%")
            else:
                st.metric("📈 ROI", "Negative")
        
        # Payback period calculation
        if tco_savings > 0:
            monthly_savings = (total_onprem_annual - total_aws_annual) / 12
            if monthly_savings > 0:
                payback_months = aws_migration_onetime / monthly_savings
                st.info(f"💡 **Payback Period: {payback_months:.1f} months** (excluding migration time)")
    
    # ================================
    # EXPORT FINANCIAL ANALYSIS
    # ================================
    
    st.subheader("📤 Export Financial Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Cost Summary", use_container_width=True):
            # Generate comprehensive cost summary
            cost_summary = {
                'analysis_mode': analysis_mode,
                'generated_at': datetime.now().isoformat(),
                'summary_metrics': {},
                'detailed_breakdown': {},
                'optimization_recommendations': optimization_suggestions
            }
            
            if analysis_mode == 'single':
                cost_summary['summary_metrics'] = {
                    'prod_monthly_cost': prod_monthly,
                    'total_monthly_all_envs': total_monthly_all_envs,
                    'prod_annual_cost': prod_monthly * 12,
                    'three_year_tco': prod_monthly * 36
                }
                
                cost_summary['detailed_breakdown'] = {
                    env_name: {
                        'monthly_cost': safe_get(result, 'total_cost', 0),
                        'annual_cost': safe_get(result, 'total_cost', 0) * 12,
                        'cost_breakdown': safe_get(result, 'cost_breakdown', {}),
                        'instance_details': {
                            'instance_type': safe_get_str(result.get('writer', result), 'instance_type', 'N/A'),
                            'vcpus': safe_get(result.get('writer', result), 'actual_vCPUs', 0),
                            'ram_gb': safe_get(result.get('writer', result), 'actual_RAM_GB', 0)
                        }
                    }
                    for env_name, result in valid_results.items()
                }
            
            else:  # Bulk analysis
                cost_summary['summary_metrics'] = {
                    'total_servers': total_servers,
                    'successful_servers': successful_servers,
                    'total_monthly_cost': total_monthly_bulk,
                    'total_annual_cost': total_monthly_bulk * 12,
                    'average_cost_per_server': total_monthly_bulk / max(successful_servers, 1)
                }
                
                cost_summary['environment_totals'] = environment_totals
                cost_summary['server_breakdown'] = {
                    server_name: {
                        env_name: {
                            'monthly_cost': safe_get(env_result, 'total_cost', 0),
                            'annual_cost': safe_get(env_result, 'total_cost', 0) * 12
                        }
                        for env_name, env_result in server_results.items() if 'error' not in env_result
                    }
                    for server_name, server_results in current_results.items() if 'error' not in server_results
                }
            
            cost_summary_json = json.dumps(cost_summary, indent=2, default=str)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            st.download_button(
                label="📥 Download Cost Summary",
                data=cost_summary_json,
                file_name=f"cost_summary_{analysis_mode}_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("📈 Export TCO Analysis", use_container_width=True):
            # Generate TCO analysis export
            tco_analysis = {
                'analysis_date': datetime.now().isoformat(),
                'analysis_mode': analysis_mode,
                'tco_comparison': {
                    'onpremises_annual': total_onprem_annual if 'total_onprem_annual' in locals() else 0,
                    'aws_annual, '').replace(',', '')) for row in summary_data)
                st.success(f"💰 **Total Monthly Cost: ${total_monthly:,.2f}** | **Annual: ${total_monthly * 12:,.2f}**")

# ================================
# TAB 4: ENHANCED FINANCIAL ANALYSIS
# ================================

with tab4:
    st.header("💰 Comprehensive Financial Analysis")
    
    # Check if we have analysis results
    current_results = None
    analysis_mode = None
    
    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_results = st.session_state.results
        analysis_mode = 'single'
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        current_results = st.session_state.bulk_results
        analysis_mode = 'bulk'
    
    if not current_results:
        st.warning("⚠️ Please run an analysis first to view financial insights.")
        st.info("💡 Go to the **Sizing Analysis** tab to generate recommendations.")
        st.stop()
    
    # ================================
    # FINANCIAL OVERVIEW METRICS
    # ================================
    
    st.subheader("📊 Financial Overview")
    
    # Calculate key financial metrics
    if analysis_mode == 'single':
        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
        
        # Environment-based calculations
        total_monthly_all_envs = sum(safe_get(result, 'total_cost', 0) for result in valid_results.values())
        prod_monthly = safe_get(valid_results.get('PROD', {}), 'total_cost', 0)
        
        # Cost breakdown for PROD environment
        prod_result = valid_results.get('PROD', list(valid_results.values())[0] if valid_results else {})
        cost_breakdown = safe_get(prod_result, 'cost_breakdown', {})
        
        # Extract compute and storage costs
        if 'writer_monthly' in cost_breakdown:
            # Aurora/Cluster setup
            compute_cost = safe_get(cost_breakdown, 'writer_monthly', 0) + safe_get(cost_breakdown, 'readers_monthly', 0)
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', 0)
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', 0)
        else:
            # Standard RDS
            compute_cost = safe_get(cost_breakdown, 'instance_monthly', 0)
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', 0)
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', 0)
        
        transfer_cost = safe_get(cost_breakdown, 'transfer_monthly', 50)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${prod_monthly:,.0f}</div>
                <div class="metric-label">PROD Monthly Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${total_monthly_all_envs:,.0f}</div>
                <div class="metric-label">All Environments</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            annual_cost = prod_monthly * 12
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${annual_cost:,.0f}</div>
                <div class="metric-label">PROD Annual Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            three_year_cost = prod_monthly * 36
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${three_year_cost:,.0f}</div>
                <div class="metric-label">3-Year TCO (PROD)</div>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Bulk analysis
        # Aggregate bulk metrics
        total_servers = len(current_results)
        successful_servers = sum(1 for result in current_results.values() if 'error' not in result)
        
        # Calculate total costs across all servers and environments
        total_monthly_bulk = 0
        environment_totals = {}
        
        for server_name, server_results in current_results.items():
            if 'error' not in server_results:
                for env_name, env_result in server_results.items():
                    if 'error' not in env_result:
                        cost = safe_get(env_result, 'total_cost', 0)
                        total_monthly_bulk += cost
                        
                        if env_name not in environment_totals:
                            environment_totals[env_name] = 0
                        environment_totals[env_name] += cost
        
        # Display bulk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{successful_servers}</div>
                <div class="metric-label">Successful Analyses</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${total_monthly_bulk:,.0f}</div>
                <div class="metric-label">Total Monthly Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_cost_per_server = total_monthly_bulk / max(successful_servers, 1)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${avg_cost_per_server:,.0f}</div>
                <div class="metric-label">Avg Cost/Server</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            annual_bulk_cost = total_monthly_bulk * 12
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${annual_bulk_cost:,.0f}</div>
                <div class="metric-label">Total Annual Cost</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ================================
    # COST VISUALIZATION CHARTS
    # ================================
    
    st.subheader("📈 Cost Analysis Visualizations")
    
    # Create cost visualization based on analysis mode
    if analysis_mode == 'single':
        # Environment cost comparison chart
        cost_comparison_fig = create_environment_cost_comparison_chart(current_results, analysis_mode)
        if cost_comparison_fig:
            st.plotly_chart(cost_comparison_fig, use_container_width=True)
        
        # Cost heatmap
        cost_heatmap = create_cost_heatmap(current_results)
        if cost_heatmap:
            st.plotly_chart(cost_heatmap, use_container_width=True)
        
        # Cost waterfall chart
        waterfall_fig = create_environment_cost_waterfall(current_results, analysis_mode)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
    
    else:  # Bulk analysis
        # Bulk summary chart
        bulk_summary_chart = create_bulk_analysis_summary_chart(current_results)
        if bulk_summary_chart:
            st.plotly_chart(bulk_summary_chart, use_container_width=True)
        
        # Environment comparison for bulk
        bulk_env_fig = create_environment_cost_comparison_chart(current_results, analysis_mode)
        if bulk_env_fig:
            st.plotly_chart(bulk_env_fig, use_container_width=True)
    
    # ================================
    # DETAILED COST BREAKDOWN
    # ================================
    
    st.subheader("🔍 Detailed Cost Breakdown")
    
    if analysis_mode == 'single':
        # Environment-by-environment breakdown
        for env_name, result in valid_results.items():
            with st.expander(f"💰 {env_name} Environment Cost Details", expanded=(env_name == 'PROD')):
                cost_breakdown = safe_get(result, 'cost_breakdown', {})
                total_cost = safe_get(result, 'total_cost', 0)
                
                # Create two columns for cost details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**💻 Compute Costs**")
                    
                    if 'writer_monthly' in cost_breakdown:
                        # Aurora/Cluster breakdown
                        writer_cost = safe_get(cost_breakdown, 'writer_monthly', 0)
                        readers_cost = safe_get(cost_breakdown, 'readers_monthly', 0)
                        
                        st.write(f"• Writer Instance: ${writer_cost:,.2f}/month")
                        st.write(f"• Reader Instances: ${readers_cost:,.2f}/month")
                        st.write(f"• **Total Compute: ${writer_cost + readers_cost:,.2f}/month**")
                        
                        # Instance details
                        if 'writer' in result:
                            writer_info = result['writer']
                            st.write(f"• Writer Type: {safe_get_str(writer_info, 'instance_type', 'N/A')}")
                            st.write(f"• Writer Resources: {safe_get(writer_info, 'actual_vCPUs', 0)} vCPUs, {safe_get(writer_info, 'actual_RAM_GB', 0)} GB RAM")
                        
                        if result.get('readers'):
                            reader_count = len(result['readers'])
                            st.write(f"• Readers: {reader_count} x {safe_get_str(result['readers'][0], 'instance_type', 'N/A')}")
                    
                    else:
                        # Standard RDS breakdown
                        instance_cost = safe_get(cost_breakdown, 'instance_monthly', 0)
                        st.write(f"• Database Instance: ${instance_cost:,.2f}/month")
                        st.write(f"• Instance Type: {safe_get_str(result, 'instance_type', 'N/A')}")
                        st.write(f"• Resources: {safe_get(result, 'actual_vCPUs', 0)} vCPUs, {safe_get(result, 'actual_RAM_GB', 0)} GB RAM")
                
                with col2:
                    st.markdown("**💾 Storage & Data Costs**")
                    
                    storage_cost = safe_get(cost_breakdown, 'storage_monthly', 0)
                    backup_cost = safe_get(cost_breakdown, 'backup_monthly', 0)
                    transfer_cost = safe_get(cost_breakdown, 'transfer_monthly', 50)
                    
                    st.write(f"• Primary Storage: ${storage_cost:,.2f}/month")
                    st.write(f"• Backup Storage: ${backup_cost:,.2f}/month")
                    st.write(f"• Data Transfer: ${transfer_cost:,.2f}/month")
                    st.write(f"• **Total Storage: ${storage_cost + backup_cost + transfer_cost:,.2f}/month**")
                    
                    # Storage details
                    storage_gb = safe_get(result, 'storage_GB', 0)
                    st.write(f"• Storage Capacity: {storage_gb:,} GB")
                    if storage_gb > 0:
                        cost_per_gb = storage_cost / storage_gb
                        st.write(f"• Cost per GB: ${cost_per_gb:.4f}/month")
                
                # Summary for this environment
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Monthly Total", f"${total_cost:,.2f}")
                
                with col2:
                    st.metric("Annual Total", f"${total_cost * 12:,.2f}")
                
                with col3:
                    if total_monthly_all_envs > 0:
                        percentage = (total_cost / total_monthly_all_envs) * 100
                        st.metric("% of Total", f"{percentage:.1f}%")
    
    else:  # Bulk analysis breakdown
        # Server-by-server cost summary
        st.markdown("**📊 Server Cost Summary**")
        
        # Create summary table
        server_cost_data = []
        
        for server_name, server_results in current_results.items():
            if 'error' not in server_results:
                # Calculate total cost across all environments for this server
                server_total = 0
                env_costs = {}
                
                for env_name, env_result in server_results.items():
                    if 'error' not in env_result:
                        cost = safe_get(env_result, 'total_cost', 0)
                        server_total += cost
                        env_costs[env_name] = cost
                
                # Add to summary data
                row_data = {
                    'Server Name': server_name,
                    'Total Monthly': f"${server_total:,.2f}",
                    'Total Annual': f"${server_total * 12:,.2f}"
                }
                
                # Add environment-specific costs
                for env in ['DEV', 'QA', 'UAT', 'PREPROD', 'PROD']:
                    if env in env_costs:
                        row_data[f'{env} Monthly'] = f"${env_costs[env]:,.2f}"
                
                server_cost_data.append(row_data)
        
        if server_cost_data:
            server_cost_df = pd.DataFrame(server_cost_data)
            st.dataframe(server_cost_df, use_container_width=True)
        
        # Environment aggregation
        st.markdown("**🏢 Environment Cost Aggregation**")
        
        if environment_totals:
            env_agg_data = []
            
            for env_name in sorted(environment_totals.keys()):
                env_total = environment_totals[env_name]
                server_count_in_env = sum(1 for server_results in current_results.values() 
                                        if 'error' not in server_results and env_name in server_results and 'error' not in server_results[env_name])
                
                env_agg_data.append({
                    'Environment': env_name,
                    'Server Count': server_count_in_env,
                    'Total Monthly': f"${env_total:,.2f}",
                    'Total Annual': f"${env_total * 12:,.2f}",
                    'Avg per Server': f"${env_total / max(server_count_in_env, 1):,.2f}",
                    '% of Total': f"{(env_total / max(total_monthly_bulk, 1)) * 100:.1f}%"
                })
            
            env_agg_df = pd.DataFrame(env_agg_data)
            st.dataframe(env_agg_df, use_container_width=True)
    
    # ================================
    # COST OPTIMIZATION RECOMMENDATIONS
    # ================================
    
    st.subheader("💡 Cost Optimization Recommendations")
    
    # Reserved Instance Savings Calculator
    with st.expander("💰 Reserved Instance Savings Calculator", expanded=True):
        st.markdown("**Calculate potential savings with Reserved Instances:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ri_term = st.selectbox("Reserved Instance Term", ["1 Year", "3 Years"], index=1)
            ri_payment = st.selectbox("Payment Option", ["No Upfront", "Partial Upfront", "All Upfront"], index=1)
        
        with col2:
            # Savings percentages based on typical AWS RI discounts
            ri_savings_map = {
                ("1 Year", "No Upfront"): 0.25,
                ("1 Year", "Partial Upfront"): 0.35,
                ("1 Year", "All Upfront"): 0.40,
                ("3 Years", "No Upfront"): 0.35,
                ("3 Years", "Partial Upfront"): 0.45,
                ("3 Years", "All Upfront"): 0.50
            }
            
            savings_percentage = ri_savings_map.get((ri_term, ri_payment), 0.40)
            
            if analysis_mode == 'single':
                base_annual_cost = prod_monthly * 12
                ri_savings_annual = base_annual_cost * savings_percentage
                ri_cost_after = base_annual_cost - ri_savings_annual
                
                st.metric("Potential Annual Savings", f"${ri_savings_annual:,.0f}")
                st.metric("Cost After RI", f"${ri_cost_after:,.0f}")
            else:
                prod_total = environment_totals.get('PROD', 0)
                base_annual_cost_bulk = prod_total * 12
                ri_savings_annual_bulk = base_annual_cost_bulk * savings_percentage
                ri_cost_after_bulk = base_annual_cost_bulk - ri_savings_annual_bulk
                
                st.metric("PROD Annual Savings", f"${ri_savings_annual_bulk:,.0f}")
                st.metric("PROD Cost After RI", f"${ri_cost_after_bulk:,.0f}")
        
        with col3:
            st.info(f"**Estimated Savings: {savings_percentage*100:.0f}%**")
            st.write(f"• Term: {ri_term}")
            st.write(f"• Payment: {ri_payment}")
            st.write("• Based on typical AWS RDS RI discounts")
    
    # Environment-specific optimization suggestions
    optimization_suggestions = []
    
    if analysis_mode == 'single':
        for env_name, result in valid_results.items():
            total_cost = safe_get(result, 'total_cost', 0)
            
            if env_name in ['DEV', 'QA']:
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Use smaller instances or stop during off-hours',
                    'Potential Savings': f"${total_cost * 0.4:,.0f}/month (40%)",
                    'Implementation': 'Schedule start/stop, use burstable instances',
                    'Risk Level': '🟢 Low'
                })
            elif env_name == 'UAT':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Implement scheduled scaling for business hours only',
                    'Potential Savings': f"${total_cost * 0.25:,.0f}/month (25%)",
                    'Implementation': 'AWS Instance Scheduler, Lambda automation',
                    'Risk Level': '🟢 Low'
                })
            elif env_name == 'PREPROD':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Optimize storage IOPS and enable compression',
                    'Potential Savings': f"${total_cost * 0.15:,.0f}/month (15%)",
                    'Implementation': 'Right-size storage, enable compression features',
                    'Risk Level': '🟡 Medium'
                })
            elif env_name == 'PROD':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Purchase Reserved Instances for predictable workloads',
                    'Potential Savings': f"${total_cost * 0.4:,.0f}/month (40%)",
                    'Implementation': '1-3 year RI commitment with upfront payment',
                    'Risk Level': '🟢 Low'
                })
    
    else:  # Bulk recommendations
        for env_name, env_total in environment_totals.items():
            if env_name in ['DEV', 'QA']:
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Implement bulk stop/start scheduling',
                    'Potential Savings': f"${env_total * 0.5:,.0f}/month (50%)",
                    'Implementation': 'Automated scheduling across all servers',
                    'Risk Level': '🟢 Low'
                })
            elif env_name == 'PROD':
                optimization_suggestions.append({
                    'Environment': env_name,
                    'Recommendation': 'Bulk Reserved Instance purchasing',
                    'Potential Savings': f"${env_total * 0.4:,.0f}/month (40%)",
                    'Implementation': 'Coordinated RI strategy across all PROD servers',
                    'Risk Level': '🟢 Low'
                })
    
    if optimization_suggestions:
        st.markdown("**🎯 Environment-Specific Optimization Opportunities:**")
        opt_df = pd.DataFrame(optimization_suggestions)
        st.dataframe(opt_df, use_container_width=True)
        
        # Calculate total optimization potential
        total_potential_savings = 0
        for suggestion in optimization_suggestions:
            savings_str = suggestion['Potential Savings']
            # Extract numeric value (rough estimation)
            import re
            savings_match = re.search(r'\$([0-9,]+)', savings_str)
            if savings_match:
                savings_value = float(savings_match.group(1).replace(',', ''))
                total_potential_savings += savings_value
        
        if total_potential_savings > 0:
            st.success(f"💰 **Total Optimization Potential: ${total_potential_savings:,.0f}/month (${total_potential_savings * 12:,.0f}/year)**")
    
    # ================================
    # TCO COMPARISON & ROI ANALYSIS
    # ================================
    
    st.subheader("📊 TCO Analysis & ROI Projection")
    
    with st.expander("🏢 Total Cost of Ownership (TCO) Analysis", expanded=False):
        st.markdown("**Compare AWS costs against current on-premises infrastructure:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏢 Current On-Premises Costs (Annual)**")
            
            # Input fields for on-premises costs
            onprem_hardware = st.number_input("Hardware/Licensing", min_value=0, value=50000, step=1000, help="Annual hardware refresh and software licensing")
            onprem_personnel = st.number_input("Personnel Costs", min_value=0, value=120000, step=1000, help="DBA and infrastructure team costs")
            onprem_facilities = st.number_input("Facilities/Power", min_value=0, value=15000, step=1000, help="Data center, power, cooling costs")
            onprem_backup = st.number_input("Backup/DR", min_value=0, value=25000, step=1000, help="Backup systems and disaster recovery")
            onprem_maintenance = st.number_input("Maintenance/Support", min_value=0, value=30000, step=1000, help="Vendor support and maintenance contracts")
            
            total_onprem_annual = onprem_hardware + onprem_personnel + onprem_facilities + onprem_backup + onprem_maintenance
            
            st.metric("**Total On-Premises Annual**", f"${total_onprem_annual:,.0f}")
        
        with col2:
            st.markdown("**☁️ AWS Cloud Costs (Annual)**")
            
            if analysis_mode == 'single':
                aws_infrastructure_annual = prod_monthly * 12
            else:
                aws_infrastructure_annual = environment_totals.get('PROD', 0) * 12
            
            # Estimated additional AWS costs
            aws_migration_onetime = st.number_input("Migration Costs (One-time)", min_value=0, value=int(aws_infrastructure_annual * 0.2), step=1000)
            aws_training_annual = st.number_input("Training/Upskilling", min_value=0, value=15000, step=1000)
            aws_management_annual = st.number_input("Cloud Management Tools", min_value=0, value=int(aws_infrastructure_annual * 0.05), step=1000)
            
            total_aws_annual = aws_infrastructure_annual + aws_training_annual + aws_management_annual
            
            st.metric("Infrastructure (Annual)", f"${aws_infrastructure_annual:,.0f}")
            st.metric("**Total AWS Annual**", f"${total_aws_annual:,.0f}")
            st.metric("Migration Investment", f"${aws_migration_onetime:,.0f}")
        
        # TCO Comparison
        st.markdown("---")
        st.markdown("**📊 3-Year TCO Comparison**")
        
        # Calculate 3-year costs
        three_year_onprem = total_onprem_annual * 3
        three_year_aws = (total_aws_annual * 3) + aws_migration_onetime
        tco_savings = three_year_onprem - three_year_aws
        roi_percentage = (tco_savings / three_year_aws) * 100 if three_year_aws > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("On-Premises 3-Year", f"${three_year_onprem:,.0f}")
        
        with col2:
            st.metric("AWS 3-Year", f"${three_year_aws:,.0f}")
        
        with col3:
            if tco_savings > 0:
                st.metric("💰 Total Savings", f"${tco_savings:,.0f}", delta=f"{(tco_savings/three_year_onprem)*100:.1f}%")
            else:
                st.metric("💰 Additional Cost", f"${abs(tco_savings):,.0f}")
        
        with col4:
            if roi_percentage > 0:
                st.metric("📈 ROI", f"{roi_percentage:.1f}%")
            else:
                st.metric("📈 ROI", "Negative")
        
        # Payback period calculation
        if tco_savings > 0:
            monthly_savings = (total_onprem_annual - total_aws_annual) / 12
            if monthly_savings > 0:
                payback_months = aws_migration_onetime / monthly_savings
                st.info(f"💡 **Payback Period: {payback_months:.1f} months** (excluding migration time)")
    
    # ================================
    # EXPORT FINANCIAL ANALYSIS
    # ================================
    
    st.subheader("📤 Export Financial Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Cost Summary", use_container_width=True):
            # Generate comprehensive cost summary
            cost_summary = {
                'analysis_mode': analysis_mode,
                'generated_at': datetime.now().isoformat(),
                'summary_metrics': {},
                'detailed_breakdown': {},
                'optimization_recommendations': optimization_suggestions
            }
            
            if analysis_mode == 'single':
                cost_summary['summary_metrics'] = {
                    'prod_monthly_cost': prod_monthly,
                    'total_monthly_all_envs': total_monthly_all_envs,
                    'prod_annual_cost': prod_monthly * 12,
                    'three_year_tco': prod_monthly * 36
                }
                
                cost_summary['detailed_breakdown'] = {
                    env_name: {
                        'monthly_cost': safe_get(result, 'total_cost', 0),
                        'annual_cost': safe_get(result, 'total_cost', 0) * 12,
                        'cost_breakdown': safe_get(result, 'cost_breakdown', {}),
                        'instance_details': {
                            'instance_type': safe_get_str(result.get('writer', result), 'instance_type', 'N/A'),
                            'vcpus': safe_get(result.get('writer', result), 'actual_vCPUs', 0),
                            'ram_gb': safe_get(result.get('writer', result), 'actual_RAM_GB', 0)
                        }
                    }
                    for env_name, result in valid_results.items()
                }
            
            else:  # Bulk analysis
                cost_summary['summary_metrics'] = {
                    'total_servers': total_servers,
                    'successful_servers': successful_servers,
                    'total_monthly_cost': total_monthly_bulk,
                    'total_annual_cost': total_monthly_bulk * 12,
                    'average_cost_per_server': total_monthly_bulk / max(successful_servers, 1)
                }
                
                cost_summary['environment_totals'] = environment_totals
                cost_summary['server_breakdown'] = {
                    server_name: {
                        env_name: {
                            'monthly_cost': safe_get(env_result, 'total_cost', 0),
                            'annual_cost': safe_get(env_result, 'total_cost', 0) * 12
                        }
                        for env_name, env_result in server_results.items() if 'error' not in env_result
                    }
                    for server_name, server_results in current_results.items() if 'error' not in server_results
                }
            
            cost_summary_json = json.dumps(cost_summary, indent=2, default=str)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            st.download_button(
                label="📥 Download Cost Summary",
                data=cost_summary_json,
                file_name=f"cost_summary_{analysis_mode}_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("📈 Export TCO Analysis", use_container_width=True):
            # Generate TCO analysis export
            tco_analysis = {
                'analysis_date': datetime.now().isoformat(),
                'analysis_mode': analysis_mode,
                'tco_comparison': {
                    'onpremises_annual': total_onprem_annual if 'total_onprem_annual' in locals() else 0,
                    'aws_annual
# ================================
# TAB 5: AI INSIGHTS
# ================================

with tab5:
    st.header("🤖 AI Insights & Recommendations")
    
    # Check for AI insights availability
    if not st.session_state.ai_insights:
        st.info("💡 Generate sizing recommendations first to enable AI insights.")
        
        # Show what's needed for AI insights
        st.subheader("🔧 AI Insights Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key_status = "✅ Available" if (st.session_state.user_claude_api_key_input or 
                                               ("anthropic" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["anthropic"]) or
                                               os.environ.get('ANTHROPIC_API_KEY')) else "❌ Missing"
            st.write(f"**Anthropic API Key:** {api_key_status}")
            
            analysis_status = "✅ Complete" if (st.session_state.results or st.session_state.bulk_results) else "❌ Pending"
            st.write(f"**Analysis Results:** {analysis_status}")
        
        with col2:
            mode = st.session_state.current_analysis_mode.title()
            st.write(f"**Analysis Mode:** {mode}")
            
            if st.session_state.current_analysis_mode == 'single':
                server_spec_status = "✅ Available" if 'current_server_spec' in st.session_state else "❌ Missing"
                st.write(f"**Server Specs:** {server_spec_status}")
            else:
                server_count = len(st.session_state.on_prem_servers) if st.session_state.on_prem_servers else 0
                st.write(f"**Bulk Servers:** {server_count} configured")
        
        if api_key_status == "❌ Missing":
            st.warning("⚠️ Please provide your Anthropic API key at the top of the page to enable AI insights.")
    
    else:
        ai_insights = st.session_state.ai_insights
        
        # Check for errors in AI insights
        if isinstance(ai_insights, dict) and ai_insights.get("error"):
            st.error(f"❌ Error retrieving AI insights: {ai_insights['error']}")
            
            # Offer to retry AI insights generation
            if st.button("🔄 Retry AI Insights Generation", type="primary"):
                with st.spinner("🤖 Regenerating AI insights..."):
                    try:
                        calculator = st.session_state.calculator
                        
                        if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                            # Retry single server AI insights
                            server_spec = st.session_state.current_server_spec
                            inputs = {
                                "region": st.session_state.region,
                                "target_engine": st.session_state.target_engine,
                                "source_engine": st.session_state.source_engine,
                                "deployment": st.session_state.deployment_option,
                                "storage_type": st.session_state.storage_type,
                                "on_prem_cores": server_spec['cores'],
                                "peak_cpu_percent": server_spec['cpu_util'],
                                "on_prem_ram_gb": server_spec['ram'],
                                "peak_ram_percent": server_spec['ram_util']
                            }
                            
                            ai_insights = asyncio.run(calculator.generate_ai_insights(st.session_state.results, inputs))
                            st.session_state.ai_insights = ai_insights
                            st.success("✅ AI insights regenerated successfully!")
                            st.rerun()
                            
                        elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                            # Retry bulk AI insights using the fixed logic
                            successful_results = {k: v for k, v in st.session_state.bulk_results.items() if 'error' not in v}
                            
                            if successful_results:
                                aggregated_results = {}
                                for server_name, server_data in successful_results.items():
                                    if 'PROD' in server_data:
                                        aggregated_results[server_name] = server_data['PROD']
                                    else:
                                        for env_key, env_result in server_data.items():
                                            if 'error' not in env_result:
                                                aggregated_results[server_name] = env_result
                                                break
                                
                                bulk_inputs = {
                                    "region": st.session_state.region,
                                    "target_engine": st.session_state.target_engine,
                                    "source_engine": st.session_state.source_engine,
                                    "deployment": st.session_state.deployment_option,
                                    "storage_type": st.session_state.storage_type,
                                    "num_servers_analyzed": len(successful_results),
                                    "analysis_mode": "bulk"
                                }
                                
                                ai_insights = asyncio.run(calculator.generate_ai_insights(aggregated_results, bulk_inputs))
                                st.session_state.ai_insights = ai_insights
                                st.success("✅ Bulk AI insights regenerated successfully!")
                                st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Failed to regenerate AI insights: {e}")
                        
        else:
            # Display AI insights successfully
            st.markdown("""
            <div class="ai-insight-card">
                <h3>🤖 AI-Powered Analysis from Claude</h3>
                <p>Leveraging advanced AI to provide deeper insights into your migration.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show analysis type
            analysis_type = st.session_state.current_analysis_mode.title()
            st.subheader(f"📊 {analysis_type} Analysis AI Insights")
            
            # Key metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = ai_insights.get('risk_level', 'N/A')
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{risk_level}</div>
                    <div class="metric-label">Migration Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cost_opt_potential = ai_insights.get('cost_optimization_potential', 0)
                if isinstance(cost_opt_potential, (int, float)):
                    cost_opt_display = f"{cost_opt_potential * 100:.0f}%"
                else:
                    cost_opt_display = "N/A"
                    
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{cost_opt_display}</div>
                    <div class="metric-label">Cost Optimization Potential</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                writers = ai_insights.get('recommended_writers', 'N/A')
                readers = ai_insights.get('recommended_readers', 'N/A')
                
                if writers != 'N/A' and readers != 'N/A':
                    arch_display = f"{writers}W / {readers}R"
                else:
                    arch_display = "Standard RDS"
                    
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{arch_display}</div>
                    <div class="metric-label">AI Recommended Arch.</div>
                </div>
                """, unsafe_allow_html=True)

            # Comprehensive AI Analysis
            st.subheader("🔍 Comprehensive AI Analysis")
            
            ai_analysis_text = ai_insights.get("ai_analysis", "No detailed AI analysis available.")
            
            if ai_analysis_text and ai_analysis_text != "No detailed AI analysis available.":
                st.markdown('<div class="advisory-box">', unsafe_allow_html=True)
                
                # Split long AI analysis into paragraphs for better readability
                if len(ai_analysis_text) > 1000:
                    paragraphs = ai_analysis_text.split('\n\n')
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            st.write(paragraph.strip())
                else:
                    st.write(ai_analysis_text)
                    
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("AI analysis is generating. Please wait or refresh the page.")
            
            # Migration Phases (if available)
            st.subheader("📅 Recommended Migration Phases")
            
            migration_phases = ai_insights.get("recommended_migration_phases")
            if migration_phases and isinstance(migration_phases, list):
                st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)
                for i, phase in enumerate(migration_phases):
                    st.markdown(f"**Phase {i+1}:** {phase}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Provide default migration phases
                default_phases = [
                    "Assessment & Discovery (2-3 weeks)",
                    "Schema Conversion & Testing (3-4 weeks)", 
                    "Data Migration Setup (1-2 weeks)",
                    "Application Code Conversion (4-6 weeks)",
                    "User Acceptance Testing (2-3 weeks)",
                    "Production Cutover (1 week)",
                    "Post-Migration Optimization (2-4 weeks)"
                ]
                
                st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)
                st.info("Using standard migration phases (AI-specific phases not available)")
                for i, phase in enumerate(default_phases):
                    st.markdown(f"**Phase {i+1}:** {phase}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights for bulk analysis
            if st.session_state.current_analysis_mode == 'bulk':
                st.subheader("📊 Bulk Analysis Specific Insights")
                
                total_servers = len(st.session_state.bulk_results) if st.session_state.bulk_results else 0
                successful_servers = len([r for r in st.session_state.bulk_results.values() if 'error' not in r]) if st.session_state.bulk_results else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Servers Analyzed", total_servers)
                
                with col2:
                    st.metric("Successful Analyses", successful_servers)
                
                with col3:
                    success_rate = (successful_servers / total_servers * 100) if total_servers > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                if successful_servers < total_servers:
                    failed_servers = total_servers - successful_servers
                    st.warning(f"⚠️ {failed_servers} servers failed analysis. Check server specifications and try again.")
            
            # Export AI insights
            st.subheader("📄 Export AI Insights")
            
            if st.button("📥 Export AI Insights as Text", use_container_width=True):
                insights_export = f"""# AWS RDS Migration AI Insights
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Mode: {st.session_state.current_analysis_mode.title()}

## Risk Assessment
Migration Risk Level: {ai_insights.get('risk_level', 'N/A')}
Cost Optimization Potential: {cost_opt_display}

## Recommended Architecture
{arch_display}

## Comprehensive Analysis
{ai_analysis_text}

## Migration Phases
"""
                
                if migration_phases:
                    for i, phase in enumerate(migration_phases):
                        insights_export += f"{i+1}. {phase}\n"
                else:
                    insights_export += "Standard migration phases recommended\n"
                
                st.download_button(
                    label="📥 Download AI Insights",
                    data=insights_export,
                    file_name=f"ai_insights_{st.session_state.current_analysis_mode}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ================================
# TAB 6: REPORTS
# ================================

with tab6:
    st.header("📋 PDF Report Generator")

    current_analysis_results = None
    current_server_specs_for_pdf = None

    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_analysis_results = st.session_state.results
        current_server_specs_for_pdf = st.session_state.get('current_server_spec')
        analysis_mode_for_pdf = 'single'
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        current_analysis_results = st.session_state.bulk_results
        current_server_specs_for_pdf = st.session_state.get('on_prem_servers')
        analysis_mode_for_pdf = 'bulk'
    
    # Enhanced PDF Generation with better error handling
    if current_analysis_results:
        # Show current analysis summary
        st.subheader("📊 Current Analysis Summary")
        if analysis_mode_for_pdf == 'single':
            st.info(f"✅ Single server analysis ready for PDF generation")
            if 'current_server_spec' in st.session_state:
                st.write(f"**Server:** {st.session_state.current_server_spec.get('server_name', 'Unknown')}")
        else:
            successful_count = len([r for r in current_analysis_results.values() if 'error' not in r])
            st.info(f"✅ Bulk analysis ready: {successful_count} successful servers")
        
        # AI Insights Status
        if st.session_state.ai_insights:
            st.success("🤖 AI insights available and will be included")
        else:
            st.warning("⚠️ No AI insights available - PDF will be generated without AI analysis")
        
        # Transfer Results Status
        if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
            st.success("🚛 Transfer analysis available and will be included")
        else:
            st.info("💡 No transfer analysis - PDF will be generated without transfer section")
        
        # PDF Generation Options
        st.subheader("⚙️ PDF Generation Options")
        col1, col2 = st.columns(2)
        
        with col1:
            include_detailed_specs = st.checkbox("Include Detailed Server Specifications", value=True)
            include_cost_breakdown = st.checkbox("Include Cost Breakdown", value=True)
        
        with col2:
            include_migration_timeline = st.checkbox("Include Migration Timeline", value=True)
            include_risk_assessment = st.checkbox("Include Risk Assessment", value=True)
        
        # Generate PDF Button with enhanced error handling
        if st.button("📄 Generate Enhanced PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating Enhanced PDF Report..."):
                try:
                    # Validate inputs before PDF generation
                    validation_errors = []
                    
                    if not current_analysis_results:
                        validation_errors.append("No analysis results available")
                    
                    if analysis_mode_for_pdf == 'single' and not current_server_specs_for_pdf:
                        validation_errors.append("Server specifications missing for single analysis")
                    
                    if analysis_mode_for_pdf == 'bulk' and not current_server_specs_for_pdf:
                        validation_errors.append("Server specifications missing for bulk analysis")
                    
                    if validation_errors:
                        st.error("❌ Validation errors:")
                        for error in validation_errors:
                            st.write(f"• {error}")
                        st.stop()
                    
                    # Enhanced PDF generation with better error handling
                    st.info("🔄 Initializing Enhanced Report Generator...")
                    
                    # Create enhanced generator instance
                    comprehensive_generator = ImprovedReportGenerator()
                    
                    st.info("🔄 Preparing report data...")
                    
                    # Prepare AI insights (ensure it's not None)
                    ai_insights_for_pdf = st.session_state.ai_insights if st.session_state.ai_insights else {
                        "risk_level": "Unknown",
                        "cost_optimization_potential": 0,
                        "ai_analysis": "AI insights were not available during analysis generation."
                    }
                    
                    # Prepare transfer results
                    transfer_results_for_pdf = None
                    if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
                        transfer_results_for_pdf = st.session_state.transfer_results
                    
                    st.info("🔄 Generating comprehensive PDF...")
                    
                    # Generate PDF with comprehensive error handling
                    # Generate comprehensive PDF with charts and detailed analysis
                    pdf_bytes = generate_improved_pdf_report(
                    analysis_results=current_analysis_results,
                    analysis_mode=analysis_mode_for_pdf,
                    server_specs=current_server_specs_for_pdf,
                    ai_insights=ai_insights_for_pdf,
                    transfer_results=transfer_results_for_pdf

                    )
                    
                    if pdf_bytes:
                        st.success("✅ Enhanced PDF Report generated successfully!")
                        
                        # Calculate file size
                        file_size_mb = len(pdf_bytes) / (1024 * 1024)
                        st.info(f"📄 PDF Size: {file_size_mb:.2f} MB")
                        
                        # Generate filename with timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"aws_rds_migration_enhanced_{analysis_mode_for_pdf}_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📥 Download Enhanced PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Show report contents summary
                        st.subheader("📋 Report Contents")
                        contents = [
                            "✅ Executive Summary",
                            "✅ Migration Strategy & Planning",
                            "✅ Technical Specifications",
                            "✅ Financial Analysis",
                            "✅ Risk Assessment"
                        ]
                        
                        if ai_insights_for_pdf and 'ai_analysis' in ai_insights_for_pdf:
                            contents.append("✅ AI-Powered Insights")
                        
                        if transfer_results_for_pdf:
                            contents.append("✅ Data Transfer Analysis")
                        
                        for content in contents:
                            st.write(content)
                    
                    else:
                        st.error("❌ Failed to generate PDF. The report generator returned None.")
                        st.info("💡 This could be due to:")
                        st.write("• Missing reportlab dependencies")
                        st.write("• Memory issues with large datasets")
                        st.write("• Data formatting problems")
                        
                        # Offer alternative export
                        st.subheader("🔄 Alternative Export Options")
                        
                        # JSON export as fallback
                        if st.button("📊 Export Analysis as JSON (Fallback)", use_container_width=True):
                            export_data = {
                                'analysis_mode': analysis_mode_for_pdf,
                                'analysis_results': current_analysis_results,
                                'server_specs': current_server_specs_for_pdf,
                                'ai_insights': ai_insights_for_pdf,
                                'transfer_results': transfer_results_for_pdf,
                                'generated_at': datetime.now().isoformat()
                            }
                            
                            json_data = json.dumps(export_data, indent=2, default=str)
                            
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_data,
                                file_name=f"analysis_report_{timestamp}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                
                except ImportError as e:
                    st.error(f"❌ Missing required libraries for PDF generation: {e}")
                    st.info("💡 Please ensure required libraries are installed:")
                    st.code("pip install reportlab kaleido plotly>=5.0.0")
                
                except Exception as e:
                    st.error(f"❌ Error generating enhanced PDF report: {str(e)}")
                    st.code(traceback.format_exc())
                    
                    # Show detailed error information
                    st.subheader("🔍 Debug Information")
                    st.write("**Analysis Mode:**", analysis_mode_for_pdf)
                    st.write("**Has Results:**", bool(current_analysis_results))
                    st.write("**Has Specs:**", bool(current_server_specs_for_pdf))
                    st.write("**Has AI Insights:**", bool(st.session_state.ai_insights))
                    st.write("**Has Transfer Results:**", bool(hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results))
    
    else:
        st.warning("⚠️ Please run an analysis first (Single or Bulk) before generating the PDF report.")
        
        # Show what's needed
        st.subheader("📋 Prerequisites for PDF Generation")
        st.write("1. Configure migration settings in **Migration Planning** tab")
        st.write("2. Set up server specifications in **Server Specifications** tab")
        st.write("3. Run analysis in **Sizing Analysis** tab")
        st.write("4. Optionally generate AI insights and transfer analysis")
        st.write("5. Return to this tab to generate comprehensive PDF report")

    # Additional export options
    st.subheader("📊 Additional Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Executive Summary**")
        if st.button("Generate Executive Summary", use_container_width=True):
            if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
                if valid_results:
                    prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                    
                    exec_summary = f"""
# Executive Summary - AWS RDS Migration Analysis

## Migration Overview
- **Source Engine:** {st.session_state.source_engine}
- **Target Engine:** {st.session_state.target_engine}
- **Migration Type:** {st.session_state.calculator.migration_profile.migration_type.value.title() if st.session_state.calculator.migration_profile else 'Unknown'}

## Cost Analysis
- **Monthly Cost:** ${safe_get(prod_result, 'total_cost', 0):,.2f}
- **Annual Cost:** ${safe_get(prod_result, 'total_cost', 0) * 12:,.2f}

## Recommended Configuration
"""
                    if 'writer' in prod_result:
                        writer_info = prod_result['writer']
                        exec_summary += f"- **Writer Instance:** {safe_get_str(writer_info, 'instance_type', 'N/A')}\n"
                        exec_summary += f"- **Writer Resources:** {safe_get(writer_info, 'actual_vCPUs', 0)} vCPUs, {safe_get(writer_info, 'actual_RAM_GB', 0)} GB RAM\n"
                        if prod_result['readers']:
                            exec_summary += f"- **Reader Instances:** {len(prod_result['readers'])} x {safe_get_str(prod_result['readers'][0], 'instance_type', 'N/A')}\n"
                    else:
                        exec_summary += f"- **Instance Type:** {safe_get_str(prod_result, 'instance_type', 'N/A')}\n"
                        exec_summary += f"- **Resources:** {safe_get(prod_result, 'actual_vCPUs', 0)} vCPUs, {safe_get(prod_result, 'actual_RAM_GB', 0)} GB RAM\n"
                    
                    exec_summary += f"- **Storage:** {safe_get(prod_result, 'storage_GB', 0)} GB\n"
                    
                    if st.session_state.ai_insights and 'ai_analysis' in st.session_state.ai_insights:
                        exec_summary += f"\n## AI Recommendations\n{st.session_state.ai_insights['ai_analysis'][:500]}...\n"
                    
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=exec_summary,
                        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                # Generate Executive Summary for Bulk Analysis
                total_servers_summary = len(st.session_state.bulk_results)
                successful_servers_summary = sum(1 for result in st.session_state.bulk_results.values() if 'error' not in result)
                total_monthly_cost_summary = sum(safe_get(result.get('PROD', {}), 'total_cost', 0) for result in st.session_state.bulk_results.values() if 'error' not in result)
                
                exec_summary_bulk = f"""
# Executive Summary - AWS RDS Bulk Migration Analysis

## Migration Overview
- **Source Engine:** {st.session_state.source_engine}
- **Target Engine:** {st.session_state.target_engine}
- **Migration Type:** {st.session_state.calculator.migration_profile.migration_type.value.title() if st.session_state.calculator.migration_profile else 'Unknown'}

## Aggregate Cost Analysis
- **Total Servers Analyzed:** {total_servers_summary}
- **Successful Analyses:** {successful_servers_summary}
- **Total Monthly Cost (Aggregated):** ${total_monthly_cost_summary:,.2f}
- **Total Annual Cost (Aggregated):** ${total_monthly_cost_summary * 12:,.2f}

## Key AI Insights (Overall Migration)
"""
                if st.session_state.ai_insights and 'ai_analysis' in st.session_state.ai_insights:
                    exec_summary_bulk += f"{st.session_state.ai_insights['ai_analysis'][:500]}...\n"
                else:
                    exec_summary_bulk += "No AI insights available for the overall bulk migration.\n"
                
                st.download_button(
                    label="📥 Download Executive Summary",
                    data=exec_summary_bulk,
                    file_name=f"executive_summary_bulk_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            else:
                st.info("Executive summary available after running an analysis.")
    
    with col2:
        st.markdown("**📋 Technical Specifications**")
        if st.button("Export Technical Specs", use_container_width=True):
            if st.session_state.current_analysis_mode == 'single' and 'current_server_spec' in st.session_state:
                tech_specs = {
                    'server_specification': st.session_state.current_server_spec,
                    'migration_config': {
                        'source_engine': st.session_state.source_engine,
                        'target_engine': st.session_state.target_engine,
                        'deployment_option': st.session_state.deployment_option,
                        'region': st.session_state.region,
                        'storage_type': st.session_state.storage_type
                    },
                    'recommendations': st.session_state.results,
                    'generated_at': datetime.now().isoformat()
                }
                
                tech_specs_json = json.dumps(tech_specs, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Tech Specs",
                    data=tech_specs_json,
                    file_name=f"technical_specifications_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                bulk_tech_specs = {
                    'analysis_mode': 'bulk',
                    'migration_config': {
                        'source_engine': st.session_state.source_engine,
                        'target_engine': st.session_state.target_engine,
                        'deployment_option': st.session_state.deployment_option,
                        'region': st.session_state.region,
                        'storage_type': st.session_state.storage_type
                    },
                    'bulk_servers_specifications': st.session_state.on_prem_servers,
                    'bulk_recommendations': st.session_state.bulk_results,
                    'generated_at': datetime.now().isoformat()
                }
                bulk_tech_specs_json = json.dumps(bulk_tech_specs, indent=2, default=str)
                st.download_button(
                    label="📥 Download Bulk Tech Specs",
                    data=bulk_tech_specs_json,
                    file_name=f"technical_specifications_bulk_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("Technical specifications available after running an analysis.")
    
    with col3:
        st.markdown("**💰 Cost Analysis Report**")
        if st.button("Generate Cost Report", use_container_width=True):
            cost_analysis = {
                'analysis_mode': st.session_state.current_analysis_mode,
                'migration_type': st.session_state.calculator.migration_profile.migration_type.value if st.session_state.calculator.migration_profile else 'unknown',
                'cost_summary': {},
                'generated_at': datetime.now().isoformat()
            }
            
            if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
                cost_analysis['cost_summary'] = {
                    'environments': {},
                    'total_monthly': 0,
                    'total_annual': 0
                }
                
                for env, result in valid_results.items():
                    monthly_cost = safe_get(result, 'total_cost', 0)
                    cost_analysis['cost_summary']['environments'][env] = {
                        'monthly_cost': monthly_cost,
                        'annual_cost': monthly_cost * 12,
                        'cost_breakdown': safe_get(result, 'cost_breakdown', {})
                    }
                    cost_analysis['cost_summary']['total_monthly'] += monthly_cost
                
                cost_analysis['cost_summary']['total_annual'] = cost_analysis['cost_summary']['total_monthly'] * 12
            
            elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                cost_analysis['cost_summary'] = {
                    'servers': {},
                    'total_monthly': 0,
                    'total_annual': 0,
                    'average_monthly_per_server': 0
                }
                
                successful_servers = 0
                
                for server_name, server_results in st.session_state.bulk_results.items():
                    if 'error' not in server_results:
                        result = server_results.get('PROD', list(server_results.values())[0])
                        if 'error' not in result:
                            monthly_cost = safe_get(result, 'total_cost', 0)
                            cost_analysis['cost_summary']['servers'][server_name] = {
                                'monthly_cost': monthly_cost,
                                'annual_cost': monthly_cost * 12
                            }
                            cost_analysis['cost_summary']['total_monthly'] += monthly_cost
                            successful_servers += 1
                
                cost_analysis['cost_summary']['total_annual'] = cost_analysis['cost_summary']['total_monthly'] * 12
                cost_analysis['cost_summary']['average_monthly_per_server'] = cost_analysis['cost_summary']['total_monthly'] / max(successful_servers, 1)
            
            cost_report_json = json.dumps(cost_analysis, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Cost Report",
                data=cost_report_json,
                file_name=f"cost_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

# ================================
# FOOTER
# ================================

st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h5>🚀 Enterprise AWS RDS Migration & Sizing Tool v2.0</h5>
    <p>AI-Powered Database Migration Analysis • Built for Enterprise Scale</p>
    <p>💡 For support and advanced features, contact your AWS solutions architect</p>
</div>
""", unsafe_allow_html=True)
                   