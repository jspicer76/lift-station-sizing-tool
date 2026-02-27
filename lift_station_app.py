"""
Lift Station Sizing Tool - Streamlit Web Application
Engineering design tool for wastewater pump stations with air valve sizing
Includes siphon flow analysis per Smith & Loveless methodology
CORRECTED: HGL calculation ensures HGL = discharge elevation at discharge (P=0)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Lift Station Sizing Tool",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

if 'minor_loss_components' not in st.session_state:
    st.session_state.minor_loss_components = pd.DataFrame({
        'Component': ['Gate Valve (fully open)', 'Check Valve (swing)', '90° Elbow (standard)', 
                     '45° Elbow', 'Tee (flow through run)', 'Entrance (sharp-edged)', 'Exit'],
        'Quantity': [1, 1, 3, 2, 1, 1, 1],
        'K-value': [0.15, 2.0, 0.9, 0.4, 0.6, 0.5, 1.0],
        'Location (ft)': [0, 0, 250, 250, 500, 0, 1000],
        'Description': ['Isolation valve at pump discharge', 'Prevents backflow', 
                       'Elbows along route', '45 degree bends', 'Junction fitting',
                       'Inlet to force main', 'Discharge to open channel/tank']
    })

if 'elevation_profile' not in st.session_state:
    st.session_state.elevation_profile = pd.DataFrame({
        'Station': [0, 1, 2],
        'Distance (ft)': [0.0, 500.0, 1000.0],
        'Elevation (ft)': [0.0, 15.0, 25.0],
        'Description': ['Pump Station', 'Intermediate Point', 'Discharge']
    })

def get_diurnal_peaking_factor(hour):
    """Get EPA diurnal peaking factor for given hour"""
    peaking_factors = [
        0.50, 0.40, 0.35, 0.30, 0.35, 0.50, 0.80, 1.20, 1.60, 1.75, 1.50, 1.20,
        1.10, 1.00, 1.00, 1.10, 1.20, 1.30, 1.50, 1.40, 1.20, 0.90, 0.70, 0.60
    ]
    return peaking_factors[int(hour) % 24]

def calculate_friction_loss_hazen_williams(Q_gpm, pipe_diameter_in, length_ft, C=100):
    """Calculate friction loss using Hazen-Williams equation"""
    if length_ft == 0:
        return 0
    h_f = 10.67 * length_ft * (Q_gpm ** 1.85) / ((C ** 1.85) * (pipe_diameter_in ** 4.87))
    return h_f

def identify_high_points(elevation_df):
    """Identify high points (local maxima) in the elevation profile"""
    high_points = []
    elevations = elevation_df['Elevation (ft)'].values
    
    for i in range(1, len(elevations) - 1):
        if elevations[i] > elevations[i-1] and elevations[i] > elevations[i+1]:
            high_points.append(i)
    
    return high_points

def calculate_air_valve_size(pipe_diameter_in, velocity_fps, elevation_change_ft, segment_length_ft, Q_gpm):
    """Calculate required air valve size based on pipe parameters"""
    
    pipe_diameter_ft = pipe_diameter_in / 12
    pipe_area_sqft = np.pi * (pipe_diameter_ft / 2) ** 2
    
    g = 32.2  # ft/s²
    vacuum_relief_cfm = 0.5 * np.sqrt(2 * g * abs(elevation_change_ft)) * pipe_area_sqft * 60
    air_release_cfm = Q_gpm / 100
    air_elimination_cfm = velocity_fps * pipe_area_sqft * 60
    
    if vacuum_relief_cfm > 10:
        valve_type = "Combination Air Valve (Air Release + Vacuum Relief)"
        primary_function = "Both air release and vacuum relief"
    elif air_release_cfm > 1:
        valve_type = "Air Release Valve"
        primary_function = "Release accumulated air during operation"
    else:
        valve_type = "Small Air Release Valve"
        primary_function = "Release small air pockets"
    
    if pipe_diameter_in <= 4:
        air_release_orifice = "1/16 inch"
        vacuum_orifice = "1 inch"
    elif pipe_diameter_in <= 8:
        air_release_orifice = "1/8 inch"
        vacuum_orifice = "2 inch"
    elif pipe_diameter_in <= 12:
        air_release_orifice = "3/16 inch"
        vacuum_orifice = "3 inch"
    elif pipe_diameter_in <= 16:
        air_release_orifice = "1/4 inch"
        vacuum_orifice = "4 inch"
    else:
        air_release_orifice = "3/8 inch"
        vacuum_orifice = "6 inch"
    
    if pipe_diameter_in <= 6:
        connection_size = "1 inch NPT"
    elif pipe_diameter_in <= 12:
        connection_size = "2 inch NPT"
    elif pipe_diameter_in <= 18:
        connection_size = "3 inch NPT"
    else:
        connection_size = "4 inch NPT"
    
    return {
        'valve_type': valve_type,
        'primary_function': primary_function,
        'air_release_orifice': air_release_orifice,
        'vacuum_orifice': vacuum_orifice,
        'connection_size': connection_size,
        'vacuum_relief_capacity_cfm': vacuum_relief_cfm,
        'air_release_capacity_cfm': air_release_cfm,
        'air_elimination_capacity_cfm': air_elimination_cfm,
        'installation_notes': f"Install at high point, minimum 12 inches above pipe crown"
    }

def calculate_design(Q_avg, Q_peak, Q_min, safety_factor,
                    pipe_diameter, num_pumps, pump_eff, motor_eff, wetwell_diameter,
                    max_cycles, min_drawdown, elevation_df, minor_loss_df,
                    hazen_c, calculate_friction):
    """
    Perform all design calculations with multi-point elevation profile
    Uses Smith & Loveless methodology: TDH to controlling high point, gravity/siphon after
    CORRECTED: HGL calculated backwards from discharge where P=0 (HGL = discharge elevation)
    """
    
    elevation_df = elevation_df.sort_values('Distance (ft)').reset_index(drop=True)
    
    distances = elevation_df['Distance (ft)'].values
    elevations = elevation_df['Elevation (ft)'].values
    
    total_length = distances[-1] - distances[0]
    
    # Discharge is always the last point
    discharge_elevation = elevations[-1]
    pump_elevation = elevations[0]
    
    # Find highest elevation point (controlling point)
    max_elevation_idx = np.argmax(elevations)
    controlling_elevation = elevations[max_elevation_idx]
    controlling_distance = distances[max_elevation_idx]
    
    # Determine flow regime
    high_point_controls = controlling_elevation > discharge_elevation
    
    if high_point_controls:
        # Per Smith & Loveless: TDH only to high point, then gravity/siphon
        design_length = controlling_distance
        static_head = controlling_elevation - pump_elevation
        flow_regime = "Pumped to high point, then gravity/siphon to discharge"
    else:
        # Discharge is highest: pressurized throughout
        design_length = total_length
        static_head = discharge_elevation - pump_elevation
        flow_regime = "Pressurized throughout entire length"
    
    # Identify high points for air valve placement
    high_point_indices = identify_high_points(elevation_df)
    has_high_points = len(high_point_indices) > 0
    
    # Calculate velocity and velocity head
    pipe_area = np.pi * (pipe_diameter / 12 / 2)**2
    pipe_velocity = Q_peak / (449 * pipe_area)
    velocity_head = pipe_velocity**2 / (2 * 32.2)
    
    # Calculate friction loss and minor losses for each segment
    segment_data = []
    
    for i in range(len(distances)):
        if i == 0:
            segment_length = 0
            friction = 0
        else:
            segment_length = distances[i] - distances[i-1]
            if calculate_friction:
                friction = calculate_friction_loss_hazen_williams(Q_peak, pipe_diameter, segment_length, hazen_c)
            else:
                friction = st.session_state.get('manual_friction_loss', 8.0) * (segment_length / total_length)
        
        # Calculate minor losses at this location
        minor_at_point = 0
        if not minor_loss_df.empty:
            for idx, row in minor_loss_df.iterrows():
                component_location = row.get('Location (ft)', 0)
                if abs(component_location - distances[i]) < 1.0:  # Within 1 ft tolerance
                    minor_at_point += row['Quantity'] * row['K-value'] * velocity_head
        
        # Determine if this segment is in pumped or siphon zone
        if high_point_controls:
            if distances[i] <= controlling_distance:
                segment_regime = "Pumped"
                in_design_zone = True
            else:
                segment_regime = "Gravity/Siphon"
                in_design_zone = False
        else:
            segment_regime = "Pressurized"
            in_design_zone = True
        
        segment_data.append({
            'station': i,
            'distance': distances[i],
            'elevation': elevations[i],
            'description': elevation_df.loc[i, 'Description'],
            'segment_length': segment_length,
            'segment_friction': friction,
            'minor_loss_at_point': minor_at_point,
            'flow_regime': segment_regime,
            'in_design_zone': in_design_zone
        })
    
    # ============================================================================
    # CORRECTED HGL CALCULATION - WORK BACKWARDS FROM DISCHARGE
    # At discharge: HGL = discharge_elevation (P = 0)
    # Work upstream adding friction and minor losses
    # ============================================================================
    
    hgl_values = [0] * len(distances)  # Initialize
    pressure_values = [0] * len(distances)  # Initialize
    
    # Start at discharge: HGL = discharge elevation (P = 0)
    hgl_values[-1] = discharge_elevation
    pressure_values[-1] = 0.0
    
    # Work backwards (upstream) from discharge
    cumulative_friction_from_discharge = 0
    cumulative_minor_from_discharge = 0
    
    for i in range(len(distances) - 2, -1, -1):  # Go backwards from second-to-last to first
        # Add friction loss from current point to next downstream point
        next_segment = segment_data[i + 1]
        cumulative_friction_from_discharge += next_segment['segment_friction']
        cumulative_minor_from_discharge += next_segment['minor_loss_at_point']
        
        # HGL increases going upstream due to friction and minor losses
        hgl_values[i] = discharge_elevation + cumulative_friction_from_discharge + cumulative_minor_from_discharge
        
        # Pressure = HGL - elevation
        pressure_values[i] = hgl_values[i] - elevations[i]
    
    # Calculate required TDH based on HGL at pump
    TDH_calculated = hgl_values[0]  # HGL at pump station
    
    # Apply safety factor for equipment selection
    TDH_design = TDH_calculated * safety_factor
    
    # Calculate losses to controlling point for reporting
    cumulative_friction_to_control = 0
    cumulative_minor_to_control = 0
    
    for i in range(1, len(distances)):
        if distances[i] <= design_length:
            cumulative_friction_to_control += segment_data[i]['segment_friction']
            cumulative_minor_to_control += segment_data[i]['minor_loss_at_point']
        else:
            break
    
    friction_loss = cumulative_friction_to_control
    minor_losses = cumulative_minor_to_control
    
    # Total losses for entire system (for reporting)
    total_friction_all = sum(seg['segment_friction'] for seg in segment_data)
    total_minor_all = sum(seg['minor_loss_at_point'] for seg in segment_data)
    
    # Verification: HGL at discharge should equal discharge elevation
    hgl_at_discharge = hgl_values[-1]
    hgl_error = abs(hgl_at_discharge - discharge_elevation)
    
    # Check for negative pressures
    min_pressure_idx = np.argmin(pressure_values)
    min_pressure = pressure_values[min_pressure_idx]
    min_pressure_location = distances[min_pressure_idx]
    has_negative_pressure = min_pressure < -5  # Allow small negative for siphon (vacuum)
    
    # Siphon analysis
    if high_point_controls:
        siphon_start_idx = max_elevation_idx
        siphon_length = distances[-1] - distances[siphon_start_idx]
        siphon_drop = elevations[siphon_start_idx] - elevations[-1]
        
        # Maximum theoretical siphon lift (atmospheric pressure ~33.9 ft water)
        max_siphon_capacity = 33.9  # ft of water at sea level
        siphon_margin = max_siphon_capacity + min_pressure  # How much margin we have
        
        siphon_data = {
            'exists': True,
            'start_station': siphon_start_idx,
            'start_distance': distances[siphon_start_idx],
            'start_elevation': elevations[siphon_start_idx],
            'length': siphon_length,
            'elevation_drop': siphon_drop,
            'min_pressure': min_pressure,
            'min_pressure_location': min_pressure_location,
            'max_vacuum_capacity': max_siphon_capacity,
            'vacuum_margin': siphon_margin,
            'is_stable': siphon_margin > 5  # Need at least 5 ft margin
        }
    else:
        siphon_data = {
            'exists': False
        }
    
    # Pump sizing
    pumps_operating = num_pumps - 1
    Q_pump_gpm = Q_peak / pumps_operating
    
    # Motor power (use TDH_design with safety factor)
    WHP = (Q_pump_gpm * TDH_design) / 3960
    BHP = WHP / pump_eff
    MHP = BHP / motor_eff
    
    standard_motors = [1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
    motor_size = min([m for m in standard_motors if m >= MHP])
    power_kw = motor_size * 0.746
    
    # Wet well sizing
    area_wetwell = np.pi * (wetwell_diameter/2)**2
    cycle_time_min = 60 / max_cycles
    volume_required_gal = (Q_avg * cycle_time_min) / 4
    volume_required_cf = volume_required_gal / 7.48
    drawdown_calculated = volume_required_cf / area_wetwell
    drawdown_design = max(drawdown_calculated, min_drawdown)
    storage_volume_cf = area_wetwell * drawdown_design
    storage_volume_gal = storage_volume_cf * 7.48
    actual_cycle_time = (storage_volume_gal * 4) / Q_avg
    actual_cycles_per_hour = 60 / actual_cycle_time
    
    fill_time_min = storage_volume_gal / Q_avg
    net_outflow_gpm = Q_pump_gpm - Q_avg
    if net_outflow_gpm > 0:
        empty_time_min = storage_volume_gal / net_outflow_gpm
    else:
        empty_time_min = float('inf')
    
    total_cycle_time_calc = fill_time_min + empty_time_min
    
    # Size air valves for each high point
    air_valve_data = []
    for idx in high_point_indices:
        station = distances[idx]
        elevation = elevations[idx]
        
        if idx > 0:
            elevation_change = elevation - elevations[idx-1]
            segment_length = distances[idx] - distances[idx-1]
        else:
            elevation_change = elevation
            segment_length = distances[idx]
        
        valve_info = calculate_air_valve_size(
            pipe_diameter, pipe_velocity, elevation_change, segment_length, Q_peak
        )
        
        # Add pressure information at this point
        pressure_at_point = pressure_values[idx]
        flow_regime_at_point = segment_data[idx]['flow_regime']
        
        air_valve_data.append({
            'station': station,
            'elevation': elevation,
            'description': elevation_df.loc[idx, 'Description'],
            'valve_info': valve_info,
            'pressure_head': pressure_at_point,
            'hgl': hgl_values[idx],
            'flow_regime': flow_regime_at_point
        })
    
    return {
        'Q_avg': Q_avg, 'Q_peak': Q_peak, 'Q_min': Q_min,
        'static_head': static_head, 
        'friction_loss': friction_loss, 
        'minor_losses': minor_losses,
        'total_friction_all': total_friction_all,
        'total_minor_all': total_minor_all,
        'pipe_diameter': pipe_diameter, 'pipe_velocity': pipe_velocity,
        'velocity_head': velocity_head, 'has_high_points': has_high_points,
        'controlling_elevation': controlling_elevation,
        'controlling_distance': controlling_distance,
        'discharge_elevation': discharge_elevation,
        'pump_elevation': pump_elevation,
        'high_point_controls': high_point_controls,
        'flow_regime': flow_regime,
        'design_length': design_length,
        'TDH_calculated': TDH_calculated,
        'TDH_design': TDH_design,
        'safety_factor': safety_factor,
        'num_pumps': num_pumps,
        'pumps_operating': pumps_operating,
        'Q_pump_gpm': Q_pump_gpm,
        'WHP': WHP, 'BHP': BHP, 'MHP': MHP,
        'motor_size': motor_size,
        'power_kw': power_kw,
        'pump_eff': pump_eff,
        'motor_eff': motor_eff,
        'wetwell_diameter': wetwell_diameter,
        'area_wetwell': area_wetwell,
        'drawdown_design': drawdown_design,
        'storage_volume_gal': storage_volume_gal,
        'actual_cycles_per_hour': actual_cycles_per_hour,
        'actual_cycle_time': actual_cycle_time,
        'fill_time_min': fill_time_min,
        'empty_time_min': empty_time_min,
        'total_cycle_time_calc': total_cycle_time_calc,
        'total_length': total_length,
        'hazen_c': hazen_c,
        'elevation_profile': elevation_df,
        'distances': distances,
        'elevations': elevations,
        'hgl_values': hgl_values,
        'pressure_values': pressure_values,
        'segment_data': segment_data,
        'high_point_indices': high_point_indices,
        'air_valve_data': air_valve_data,
        'has_negative_pressure': has_negative_pressure,
        'min_pressure': min_pressure,
        'min_pressure_location': min_pressure_location,
        'siphon_data': siphon_data,
        'hgl_at_discharge': hgl_at_discharge,
        'hgl_error': hgl_error
    }

# Title and header
st.title("Lift Station Sizing Tool")
st.markdown("### Professional Engineering Design Tool for Wastewater Pump Stations")
st.markdown("**With Multi-Point Elevation Profile, Siphon Analysis, and Air Valve Sizing**")
st.markdown("---")

# Add hydraulic principle explanation
with st.expander("IMPORTANT: Smith & Loveless Methodology - TDH and Siphon Flow", expanded=False):
    st.markdown("""
    ### Wastewater Forcemain Design Principles
    
    **Unlike pressurized water systems, wastewater forcemains discharge to atmosphere (P=0)**
    
    #### Design Methodology (Smith & Loveless):
    
    **Case 1: High Point Above Discharge**
    ```
    Pump Station (Elev 0)
        ↓ [PUMPED - TDH calculated for this section only]
    High Point (Elev 40 ft) ← CONTROLLING POINT
        ↓ [GRAVITY/SIPHON FLOW - No additional pumping required]
    Discharge (Elev 25 ft, P=0)
    ```
    
    **TDH = (High Point Elevation) + Friction (to high point) + Minor Losses (to high point)**
    
    **Case 2: Discharge is Highest Point**
    ```
    Pump Station (Elev 0)
        ↓ [PRESSURIZED throughout]
    Discharge (Elev 25 ft, P=0) ← CONTROLLING POINT
    ```
    
    **TDH = (Discharge Elevation) + Friction (full length) + Minor Losses (full length)**
    
    #### Why Siphon Flow Occurs:
    - Water "falls" from high point to lower discharge
    - Creates vacuum/negative pressure in descending section
    - Flow continues by gravity and atmospheric pressure
    - **Air valves CRITICAL** to break siphon when pump stops
    
    #### Siphon Stability:
    - Maximum siphon lift ≈ 33.9 ft (atmospheric pressure at sea level)
    - If vacuum exceeds this, air will be drawn in or pipe may collapse
    - Need minimum 5 ft safety margin
    
    #### At Discharge:
    - **HGL ALWAYS = Discharge Elevation** (pressure = 0, atmospheric)
    - This is fundamentally different from pressurized water distribution
    
    #### CORRECTED HGL CALCULATION:
    - HGL calculated backwards from discharge where P=0
    - Ensures HGL = discharge elevation at discharge point
    - TDH determined from required HGL at pump station
    """)

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    
    st.subheader("Flow Rates (GPM)")
    Q_avg = st.number_input("Average Daily Flow", value=150.0, step=10.0)
    Q_peak = st.number_input("Peak Flow", value=400.0, step=10.0)
    Q_min = st.number_input("Minimum Flow", value=50.0, step=10.0)
    
    st.subheader("Friction Loss Calculation")
    calculate_friction = st.checkbox("Auto-calculate friction loss", value=True, 
                                     help="Use Hazen-Williams equation")
    
    if calculate_friction:
        hazen_c = st.number_input("Hazen-Williams C-factor", value=100, min_value=80, max_value=150, step=5,
                                 help="100=wastewater/old pipe, 120=water, 140=new pipe")
    else:
        manual_friction = st.number_input("Manual Friction Loss (ft)", value=8.0, step=0.5)
        st.session_state.manual_friction_loss = manual_friction
        hazen_c = 100
    
    safety_factor = st.number_input("Safety Factor", value=1.15, step=0.05)
    
    st.subheader("Forcemain Parameters")
    pipe_diameter = st.number_input("Pipe Diameter (inches)", value=8.0, step=1.0)
    
    st.subheader("Pump Configuration")
    num_pumps = st.selectbox("Number of Pumps", [2, 3, 4], index=0)
    pump_eff = st.slider("Pump Efficiency", 0.60, 0.85, 0.70, 0.01)
    motor_eff = st.slider("Motor Efficiency", 0.85, 0.95, 0.90, 0.01)
    
    st.subheader("Wet Well Parameters")
    wetwell_diameter = st.number_input("Diameter (ft)", value=6.0, step=1.0)
    max_cycles = st.number_input("Max Cycles/Hour", value=6.0, step=1.0)
    min_drawdown = st.number_input("Min Drawdown (ft)", value=2.0, step=0.5)
    
    st.markdown("---")
    calculate_button = st.button("Calculate Design", type="primary", use_container_width=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Results", 
    "Elevation Profile (10 pts)", 
    "Minor Losses with Locations",
    "Siphon Analysis",
    "Air Valve Design",
    "Analysis & HGL", 
    "Export"
])

# Tab 1: Results
with tab1:
    if calculate_button:
        with st.spinner("Calculating design parameters..."):
            try:
                results = calculate_design(
                    Q_avg, Q_peak, Q_min, safety_factor,
                    pipe_diameter, num_pumps, pump_eff, motor_eff, wetwell_diameter,
                    max_cycles, min_drawdown, st.session_state.elevation_profile,
                    st.session_state.minor_loss_components, hazen_c, calculate_friction
                )
                st.session_state.results = results
                st.success("Design calculations completed successfully!")
                
                # Display flow regime
                if results['high_point_controls']:
                    st.warning(f"⚠️ HIGH POINT CONTROLS DESIGN - Siphon flow occurs after high point")
                    st.info(f"TDH calculated to station {results['controlling_distance']:.0f} ft only. "
                           f"Flow is gravity/siphon from there to discharge.")
                else:
                    st.info("Discharge controls design - Pressurized flow throughout")
                
                if results['has_negative_pressure'] and results['siphon_data']['exists']:
                    vacuum_psi = abs(results['min_pressure']) * 0.433
                    st.error(f"⚠️ VACUUM in siphon section: {abs(results['min_pressure']):.1f} ft ({vacuum_psi:.1f} psi) "
                            f"at station {results['min_pressure_location']:.0f} ft")
                
            except Exception as e:
                st.error(f"Error in calculations: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    if st.session_state.results:
        r = st.session_state.results
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Dynamic Head", f"{r['TDH_design']:.1f} ft")
        with col2:
            st.metric("Pump Capacity", f"{r['Q_pump_gpm']:.0f} GPM")
        with col3:
            st.metric("Motor Size", f"{r['motor_size']:.1f} HP")
        with col4:
            st.metric("Pipe Velocity", f"{r['pipe_velocity']:.2f} ft/s")
        with col5:
            st.metric("Design Length", f"{r['design_length']:.0f} ft")
        
        st.markdown("---")
        
        # HGL Verification - CORRECTED
        with st.expander("✅ HGL Verification (CORRECTED METHOD)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**HGL at Pump:** {r['hgl_values'][0]:.2f} ft")
                st.write(f"**HGL at Discharge:** {r['hgl_values'][-1]:.6f} ft")
                st.write(f"**Discharge Elevation:** {r['discharge_elevation']:.6f} ft")
                
                # Verify HGL = elevation at discharge
                hgl_error = r['hgl_error']
                if hgl_error < 0.001:  # Very tight tolerance
                    st.success(f"✅ PERFECT: HGL = Discharge Elevation (error: {hgl_error:.8f} ft)")
                elif hgl_error < 0.1:
                    st.success(f"✅ GOOD: HGL ≈ Discharge Elevation (error: {hgl_error:.4f} ft)")
                else:
                    st.error(f"❌ ERROR: HGL ≠ Discharge Elevation (error: {hgl_error:.2f} ft)")
            
            with col2:
                st.write(f"**TDH (from HGL calculation):** {r['TDH_calculated']:.2f} ft")
                st.write(f"**TDH (design, SF={r['safety_factor']}):** {r['TDH_design']:.2f} ft")
                st.info("""
                **CORRECTED METHOD:**
                ✅ HGL calculated backwards from discharge (P=0)
                ✅ HGL at discharge = discharge elevation EXACTLY
                ✅ TDH = HGL at pump station
                ✅ Safety factor applied for equipment selection only
                """)
        
        # Flow Regime Summary
        with st.expander("Flow Regime Summary", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Flow Regime:** {r['flow_regime']}")
                st.write(f"**Controlling Point:** {r['controlling_elevation']:.1f} ft at {r['controlling_distance']:.0f} ft")
                st.write(f"**Discharge Elevation:** {r['discharge_elevation']:.1f} ft")
                st.write(f"**Design Length (for TDH):** {r['design_length']:.0f} ft")
                
                if r['high_point_controls']:
                    st.error(f"**HIGH POINT CONTROLS**")
                    st.write(f"• Pumped section: 0 to {r['design_length']:.0f} ft")
                    st.write(f"• Siphon section: {r['design_length']:.0f} to {r['total_length']:.0f} ft")
                    st.write(f"• Elevation drop in siphon: {r['controlling_elevation'] - r['discharge_elevation']:.1f} ft")
                else:
                    st.success(f"**DISCHARGE CONTROLS (no siphon)**")
            
            with col2:
                st.write(f"**Static Head:** {r['static_head']:.1f} ft")
                st.write(f"**Friction Loss (to control):** {r['friction_loss']:.2f} ft")
                st.write(f"**Minor Losses (to control):** {r['minor_losses']:.2f} ft")
                st.write(f"**Calculated TDH:** {r['TDH_calculated']:.1f} ft")
                st.write(f"**Design TDH (SF={r['safety_factor']}):** {r['TDH_design']:.1f} ft")
                
                if r['high_point_controls']:
                    st.caption(f"Note: Additional friction loss in siphon section: {r['total_friction_all'] - r['friction_loss']:.2f} ft (not included in TDH)")
        
        # Point-by-Point Analysis Table
        with st.expander("Point-by-Point Hydraulic Analysis", expanded=True):
            analysis_data = []
            for seg in r['segment_data']:
                analysis_data.append({
                    'Station': seg['station'],
                    'Distance (ft)': f"{seg['distance']:.1f}",
                    'Elevation (ft)': f"{seg['elevation']:.1f}",
                    'Description': seg['description'],
                    'Flow Regime': seg['flow_regime'],
                    'HGL (ft)': f"{r['hgl_values'][seg['station']]:.2f}",
                    'Pressure (ft)': f"{r['pressure_values'][seg['station']]:.2f}"
                })
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
        
        # Flow Parameters
        with st.expander("Flow Parameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Average Flow:** {r['Q_avg']:.2f} GPM")
                st.write(f"**Peak Flow:** {r['Q_peak']:.2f} GPM")
                st.write(f"**Minimum Flow:** {r['Q_min']:.2f} GPM")
            with col2:
                st.write(f"**Peak/Average Ratio:** {r['Q_peak']/r['Q_avg']:.2f}")
                st.write(f"**Pipe Velocity:** {r['pipe_velocity']:.2f} ft/s")
                if r['pipe_velocity'] < 2:
                    st.warning("Velocity below 2 ft/s - risk of solids settling")
                elif r['pipe_velocity'] > 8:
                    st.warning("Velocity above 8 ft/s - risk of erosion")
                else:
                    st.success("Velocity within recommended range (2-8 ft/s)")
        
        # Pump Specifications
        with st.expander("Pump Specifications", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Number of Pumps:** {r['num_pumps']}")
                st.write(f"**Operating Configuration:** {r['pumps_operating']} (N-1 redundancy)")
                st.write(f"**Pump Capacity (each):** {r['Q_pump_gpm']:.2f} GPM")
                st.write(f"**Pump Head:** {r['TDH_design']:.2f} ft")
            with col2:
                st.write(f"**Water Horsepower:** {r['WHP']:.2f} HP")
                st.write(f"**Brake Horsepower:** {r['BHP']:.2f} HP")
                st.write(f"**Selected Motor Size:** {r['motor_size']:.1f} HP")
                st.write(f"**Power per Pump:** {r['power_kw']:.2f} kW")
        
        # Wet Well Design
        with st.expander("Wet Well Design", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Physical Dimensions:**")
                st.write(f"**Diameter:** {r['wetwell_diameter']:.1f} ft")
                st.write(f"**Area:** {r['area_wetwell']:.2f} ft²")
                st.write(f"**Drawdown Depth:** {r['drawdown_design']:.2f} ft")
                st.write(f"**Storage Volume:** {r['storage_volume_gal']:.0f} gallons")
            with col2:
                st.markdown("**Operating Cycles:**")
                st.write(f"**Cycle Time:** {r['actual_cycle_time']:.2f} minutes")
                st.write(f"**Cycles per Hour:** {r['actual_cycles_per_hour']:.2f}")
                st.write(f"**Fill Time:** {r['fill_time_min']:.2f} minutes")
                st.write(f"**Empty Time:** {r['empty_time_min']:.2f} minutes")
    else:
        st.info("Enter parameters in the sidebar and click 'Calculate Design' to see results")

        # Tab 2: Elevation Profile (10 points)
with tab2:
    st.subheader("Forcemain Elevation Profile Configuration (Up to 10 Points)")
    st.markdown("Define up to 10 elevation points along the forcemain alignment. System will identify controlling point and siphon zones.")
    
    st.info("**Tip:** Add points at significant elevation changes, high points, and low points. First point = Pump Station, Last point = Discharge.")
    
    edited_profile = st.data_editor(
        st.session_state.elevation_profile,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Station": st.column_config.NumberColumn(
                "Station #",
                help="Sequential station number",
                min_value=0,
                max_value=20,
                step=1,
                format="%d"
            ),
            "Distance (ft)": st.column_config.NumberColumn(
                "Distance from Pump (ft)",
                help="Horizontal distance from pump station",
                min_value=0.0,
                step=10.0,
                format="%.1f"
            ),
            "Elevation (ft)": st.column_config.NumberColumn(
                "Elevation (ft)",
                help="Elevation above pump centerline",
                step=1.0,
                format="%.1f"
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                help="Brief description of this location",
                max_chars=100
            )
        },
        hide_index=True,
        key="elevation_editor"
    )
    
    # Clean and validate
    if not edited_profile.empty:
        edited_profile = edited_profile.dropna(subset=['Distance (ft)', 'Elevation (ft)'], how='all')
        
        if 'Station' in edited_profile.columns:
            edited_profile['Station'] = edited_profile['Station'].fillna(method='ffill').fillna(0).astype(int)
        if 'Distance (ft)' in edited_profile.columns:
            edited_profile['Distance (ft)'] = pd.to_numeric(edited_profile['Distance (ft)'], errors='coerce').fillna(0.0)
        if 'Elevation (ft)' in edited_profile.columns:
            edited_profile['Elevation (ft)'] = pd.to_numeric(edited_profile['Elevation (ft)'], errors='coerce').fillna(0.0)
        if 'Description' in edited_profile.columns:
            edited_profile['Description'] = edited_profile['Description'].fillna('Point')
    
    if len(edited_profile) > 10:
        st.error("Maximum 10 elevation points allowed. Please remove excess points.")
    elif len(edited_profile) < 2:
        st.warning("Minimum 2 points required (pump station and discharge).")
    else:
        st.session_state.elevation_profile = edited_profile
        st.success(f"Elevation profile saved with {len(edited_profile)} points")
        
        # Profile preview
        st.markdown("### Profile Preview")
        fig = go.Figure()
        
        sorted_profile = edited_profile.sort_values('Distance (ft)')
        
        fig.add_trace(go.Scatter(
            x=sorted_profile['Distance (ft)'],
            y=sorted_profile['Elevation (ft)'],
            mode='lines+markers',
            name='Elevation Profile',
            line=dict(color='blue', width=3),
            marker=dict(size=10, color='blue')
        ))
        
        # Identify high point
        max_idx = sorted_profile['Elevation (ft)'].idxmax()
        max_elev = sorted_profile.loc[max_idx, 'Elevation (ft)']
        max_dist = sorted_profile.loc[max_idx, 'Distance (ft)']
        
        fig.add_trace(go.Scatter(
            x=[max_dist],
            y=[max_elev],
            mode='markers+text',
            name='Controlling Point',
            marker=dict(size=15, color='red', symbol='star'),
            text=['HIGH POINT'],
            textposition='top center'
        ))
        
        for idx, row in sorted_profile.iterrows():
            fig.add_annotation(
                x=row['Distance (ft)'],
                y=row['Elevation (ft)'],
                text=row['Description'],
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                bgcolor="yellow",
                opacity=0.8
            )
        
        fig.update_layout(
            xaxis_title="Distance Along Forcemain (ft)",
            yaxis_title="Elevation (ft)",
            height=450,
            showlegend=True,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset to Default", use_container_width=True):
            st.session_state.elevation_profile = pd.DataFrame({
                'Station': [0, 1, 2],
                'Distance (ft)': [0.0, 500.0, 1000.0],
                'Elevation (ft)': [0.0, 15.0, 25.0],
                'Description': ['Pump Station', 'Intermediate', 'Discharge']
            })
            st.rerun()
    
    with col2:
        if st.button("Load Example: High Point", use_container_width=True):
            st.session_state.elevation_profile = pd.DataFrame({
                'Station': [0, 1, 2, 3, 4],
                'Distance (ft)': [0.0, 300.0, 600.0, 800.0, 1200.0],
                'Elevation (ft)': [0.0, 15.0, 35.0, 20.0, 25.0],
                'Description': ['Pump Station', 'Rising', 'High Point', 'Descending', 'Discharge']
            })
            st.rerun()
    
    with col3:
        if st.button("Load Example: Complex Profile", use_container_width=True):
            st.session_state.elevation_profile = pd.DataFrame({
                'Station': [0, 1, 2, 3, 4, 5, 6, 7],
                'Distance (ft)': [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0],
                'Elevation (ft)': [0.0, 10.0, 25.0, 40.0, 35.0, 20.0, 15.0, 30.0],
                'Description': ['Pump', 'Point 1', 'Point 2', 'High Point', 'Point 4', 'Point 5', 'Low Point', 'Discharge']
            })
            st.rerun()
# Tab 3: Minor Losses with Locations
with tab3:
    st.subheader("Minor Loss Components with Locations")
    st.markdown("Configure fittings, valves, and appurtenances. **Each row represents component(s) at a specific location.**")
    
    st.info("""
    **How to use:**
    - Each row can have multiple identical components (use Quantity) at ONE location
    - Add multiple rows for same component type at different locations
    - Example: "90° Elbow, Qty=2, Location=250 ft" and "90° Elbow, Qty=1, Location=600 ft"
    """)
    
    # Enhanced data editor
    edited_minor = st.data_editor(
        st.session_state.minor_loss_components,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Component": st.column_config.SelectboxColumn(
                "Component Type",
                help="Select component type or enter custom",
                width="medium",
                options=[
                    "Gate Valve (fully open)",
                    "Gate Valve (3/4 open)",
                    "Gate Valve (1/2 open)",
                    "Check Valve (swing)",
                    "Check Valve (ball)",
                    "90° Elbow (standard)",
                    "90° Elbow (long radius)",
                    "45° Elbow",
                    "Tee (flow through run)",
                    "Tee (flow through branch)",
                    "Reducer",
                    "Enlargement",
                    "Entrance (sharp-edged)",
                    "Entrance (bell-mouth)",
                    "Exit",
                    "Butterfly Valve",
                    "Custom"
                ]
            ),
            "Quantity": st.column_config.NumberColumn(
                "Qty",
                help="Number of identical components at this location",
                min_value=1, 
                max_value=10, 
                step=1,
                width="small"
            ),
            "K-value": st.column_config.NumberColumn(
                "K-value",
                help="Loss coefficient",
                min_value=0.0, 
                max_value=20.0, 
                step=0.1, 
                format="%.2f",
                width="small"
            ),
            "Location (ft)": st.column_config.NumberColumn(
                "Location (ft)",
                help="Distance from pump where component(s) are located",
                min_value=0.0,
                step=10.0,
                format="%.0f",
                width="small"
            ),
            "Description": st.column_config.TextColumn(
                "Description/Notes",
                help="Additional notes",
                width="large"
            )
        },
        hide_index=True,
        key="minor_loss_editor"
    )
    
    st.session_state.minor_loss_components = edited_minor
    
    # Analysis by location
    st.markdown("---")
    st.markdown("### Minor Loss Summary by Location")
    
    if not edited_minor.empty:
        # Group by location
        location_summary = edited_minor.groupby('Location (ft)').apply(
            lambda x: pd.Series({
                'Components': ', '.join([f"{row['Quantity']}x {row['Component']}" for _, row in x.iterrows()]),
                'Total K-value': (x['Quantity'] * x['K-value']).sum(),
                'Count': len(x)
            })
        ).reset_index()
        
        st.dataframe(location_summary, use_container_width=True)
        
        # Calculate total K-value and show by zone if applicable
        total_k = (edited_minor['Quantity'] * edited_minor['K-value']).sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Minor Loss K-value", f"{total_k:.2f}")
        
        if st.session_state.results and st.session_state.results['high_point_controls']:
            control_dist = st.session_state.results['design_length']
            
            k_pumped = edited_minor[edited_minor['Location (ft)'] <= control_dist].apply(
                lambda row: row['Quantity'] * row['K-value'], axis=1
            ).sum()
            
            k_siphon = edited_minor[edited_minor['Location (ft)'] > control_dist].apply(
                lambda row: row['Quantity'] * row['K-value'], axis=1
            ).sum()
            
            with col2:
                st.success(f"**Pumped Zone K:** {k_pumped:.2f}")
                st.caption(f"(0 to {control_dist:.0f} ft)")
            
            with col3:
                st.info(f"**Siphon Zone K:** {k_siphon:.2f}")
                st.caption(f"({control_dist:.0f} to end)")
                st.caption("(Not included in TDH)")
    
    # Quick-add buttons for common components
    st.markdown("---")
    st.markdown("### Quick Add Common Components")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➕ Add 90° Elbow", use_container_width=True):
            new_row = pd.DataFrame([{
                'Component': '90° Elbow (standard)',
                'Quantity': 1,
                'K-value': 0.9,
                'Location (ft)': 0,
                'Description': 'Standard radius elbow'
            }])
            st.session_state.minor_loss_components = pd.concat([
                st.session_state.minor_loss_components, new_row
            ], ignore_index=True)
            st.rerun()
    
    with col2:
        if st.button("➕ Add 45° Elbow", use_container_width=True):
            new_row = pd.DataFrame([{
                'Component': '45° Elbow',
                'Quantity': 1,
                'K-value': 0.4,
                'Location (ft)': 0,
                'Description': '45 degree bend'
            }])
            st.session_state.minor_loss_components = pd.concat([
                st.session_state.minor_loss_components, new_row
            ], ignore_index=True)
            st.rerun()
    
    with col3:
        if st.button("➕ Add Check Valve", use_container_width=True):
            new_row = pd.DataFrame([{
                'Component': 'Check Valve (swing)',
                'Quantity': 1,
                'K-value': 2.0,
                'Location (ft)': 0,
                'Description': 'Swing check valve'
            }])
            st.session_state.minor_loss_components = pd.concat([
                st.session_state.minor_loss_components, new_row
            ], ignore_index=True)
            st.rerun()
    
    with col4:
        if st.button("➕ Add Gate Valve", use_container_width=True):
            new_row = pd.DataFrame([{
                'Component': 'Gate Valve (fully open)',
                'Quantity': 1,
                'K-value': 0.15,
                'Location (ft)': 0,
                'Description': 'Isolation valve'
            }])
            st.session_state.minor_loss_components = pd.concat([
                st.session_state.minor_loss_components, new_row
            ], ignore_index=True)
            st.rerun()
    
    # K-value reference table
    with st.expander("📚 K-value Reference Table (Crane TP-410)", expanded=False):
        reference_data = {
            'Component': [
                'Gate Valve (fully open)', 'Gate Valve (3/4 open)', 'Gate Valve (1/2 open)',
                'Globe Valve (fully open)', 'Check Valve (swing)', 'Check Valve (ball)',
                'Butterfly Valve (fully open)', '90° Elbow (standard)', '90° Elbow (long radius)',
                '45° Elbow', 'Tee (flow through run)', 'Tee (flow through branch)',
                'Reducer (sudden)', 'Enlargement (sudden)', 'Entrance (sharp-edged)',
                'Entrance (bell-mouth)', 'Exit'
            ],
            'K-value': [
                0.15, 0.9, 4.5, 6.0, 2.0, 10.0, 0.3, 0.9, 0.6, 0.4, 0.6, 1.8,
                0.5, 1.0, 0.5, 0.05, 1.0
            ],
            'Notes': [
                'Minimal restriction', 'Partially closed', 'Half closed',
                'High resistance', 'Standard swing check', 'High resistance check',
                'Quarter-turn valve', 'r/D = 1', 'r/D = 1.5',
                'Less restriction', 'Flow straight through', 'Flow turns 90°',
                'Contraction', 'Expansion', 'No rounding',
                'Streamlined inlet', 'Discharge to atmosphere'
            ]
        }
        reference_df = pd.DataFrame(reference_data)
        st.dataframe(reference_df, use_container_width=True)
    
    # Reset and template buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset to Default Configuration", use_container_width=True):
            st.session_state.minor_loss_components = pd.DataFrame({
                'Component': ['Gate Valve (fully open)', 'Check Valve (swing)', 'Entrance (sharp-edged)', 'Exit'],
                'Quantity': [1, 1, 1, 1],
                'K-value': [0.15, 2.0, 0.5, 1.0],
                'Location (ft)': [0, 0, 0, 1000],
                'Description': ['Isolation valve at pump', 'Prevents backflow', 'Pump station inlet', 'Discharge point']
            })
            st.rerun()
    
    with col2:
        if st.button("📋 Load Typical Force Main Template", use_container_width=True):
            st.session_state.minor_loss_components = pd.DataFrame({
                'Component': [
                    'Entrance (sharp-edged)', 'Gate Valve (fully open)', 'Check Valve (swing)',
                    '90° Elbow (standard)', '90° Elbow (standard)', '45° Elbow',
                    '90° Elbow (standard)', '45° Elbow', 'Exit'
                ],
                'Quantity': [1, 1, 1, 2, 1, 2, 2, 1, 1],
                'K-value': [0.5, 0.15, 2.0, 0.9, 0.9, 0.4, 0.9, 0.4, 1.0],
                'Location (ft)': [0, 0, 0, 150, 350, 350, 650, 850, 1000],
                'Description': [
                    'Pump inlet', 'Isolation valve', 'Check valve',
                    'Direction change', 'Elbow at station', 'Reduce angle',
                    'Alignment bends', 'Final approach', 'Discharge'
                ]
            })
            st.rerun()
# Tab 4: Siphon Analysis
with tab4:
    st.subheader("Siphon Flow Analysis")
    
    if st.session_state.results:
        r = st.session_state.results
        siphon = r['siphon_data']
        
        if siphon['exists']:
            st.success("✅ Siphon flow detected in system")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Siphon Characteristics")
                st.write(f"**Siphon Start:** Station {siphon['start_station']} at {siphon['start_distance']:.0f} ft")
                st.write(f"**Start Elevation:** {siphon['start_elevation']:.1f} ft")
                st.write(f"**Siphon Length:** {siphon['length']:.0f} ft")
                st.write(f"**Elevation Drop:** {siphon['elevation_drop']:.1f} ft")
                
                st.markdown("---")
                st.markdown("### Vacuum Analysis")
                st.write(f"**Minimum Pressure in Siphon:** {siphon['min_pressure']:.2f} ft")
                if siphon['min_pressure'] < 0:
                    vacuum_psi = abs(siphon['min_pressure']) * 0.433
                    st.error(f"**Vacuum:** {abs(siphon['min_pressure']):.1f} ft ({vacuum_psi:.1f} psi)")
                st.write(f"**Location:** {siphon['min_pressure_location']:.0f} ft")
                
                st.write(f"**Max Siphon Capacity:** {siphon['max_vacuum_capacity']:.1f} ft (atmospheric)")
                st.write(f"**Available Margin:** {siphon['vacuum_margin']:.1f} ft")
                
                if siphon['is_stable']:
                    st.success(f"✅ Siphon is stable (margin > 5 ft)")
                else:
                    st.error(f"⚠️ Siphon may be unstable (margin < 5 ft)")
                    st.warning("**Recommendation:** Reduce elevation at high point or increase TDH")
            
            with col2:
                st.markdown("### Siphon Diagram")
                st.code(f"""
    HIGH POINT (Control Point)
    Elev: {siphon['start_elevation']:.1f} ft
    Station: {siphon['start_distance']:.0f} ft
         |
         | <-- SIPHON ZONE ({siphon['length']:.0f} ft)
         |     • Gravity flow
         |     • Vacuum: {abs(siphon['min_pressure']):.1f} ft
         |     • Flow driven by elevation drop
         |
    DISCHARGE
    Elev: {r['discharge_elevation']:.1f} ft
    Station: {r['total_length']:.0f} ft
    Pressure: 0 (atmospheric)
                """, language="text")
                
                st.markdown("### Siphon Operation")
                st.info("""
                **During Pump Operation:**
                - Pump fills pipe to high point
                - Gravity pulls water down from high point
                - Vacuum develops in descending section
                - Air valve releases trapped air
                
                **When Pump Stops:**
                - Air valve admits air at high point
                - Breaks siphon
                - Prevents backflow
                - Prevents pipe collapse from vacuum
                """)
            
            # Siphon pressure profile
            st.markdown("---")
            st.markdown("### Pressure Profile in Siphon Zone")
            
            siphon_segments = [seg for seg in r['segment_data'] if seg['distance'] >= siphon['start_distance']]
            
            siphon_distances = [seg['distance'] for seg in siphon_segments]
            siphon_pressures = [r['pressure_values'][seg['station']] for seg in siphon_segments]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=siphon_distances,
                y=siphon_pressures,
                mode='lines+markers',
                name='Pressure Head',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Atmospheric (0 ft)")
            fig.add_hline(y=-siphon['max_vacuum_capacity'], line_dash="dash", line_color="darkred", 
                         annotation_text=f"Max Vacuum (-{siphon['max_vacuum_capacity']:.1f} ft)")
            
            fig.update_layout(
                title="Pressure Head in Siphon Zone",
                xaxis_title="Distance (ft)",
                yaxis_title="Pressure Head (ft)",
                height=400,
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Design recommendations
            st.markdown("---")
            st.markdown("### Design Recommendations")
            
            if siphon['min_pressure'] < -15:
                st.error("⚠️ **CRITICAL:** Excessive vacuum may cause:")
                st.markdown("""
                - Pipe collapse risk
                - Potential for water column separation
                - Increased cavitation risk
                - **ACTION:** Reduce high point elevation or add booster pump
                """)
            elif siphon['min_pressure'] < -5:
                st.warning("⚠️ **CAUTION:** Moderate vacuum present:")
                st.markdown("""
                - Monitor for air accumulation
                - Ensure air valves are properly sized
                - Consider pipe material strength
                """)
            else:
                st.success("✅ Vacuum levels acceptable")
            
            st.markdown("""
            **Required Air Valve Functions:**
            1. **Air Release:** Remove trapped air during system filling and operation
            2. **Vacuum Relief:** Admit air when pump stops to break siphon
            3. **Combination:** Most high points require combination air/vacuum valves
            
            **Critical:** Air valve at high point is MANDATORY for safe operation!
            """)
            
        else:
            st.info("No siphon flow in this system - discharge is the highest point")
            st.markdown("""
            ### Pressurized Flow Throughout
            
            This system maintains positive pressure along entire length because:
            - Discharge point is at highest elevation (or equal to highest point)
            - No descending sections after high points
            - Conventional pressurized forcemain design
            
            **TDH Calculation:**
            - Based on full pipe length to discharge
            - Includes all friction and minor losses
            - No gravity assist from elevation drops
            """)
    else:
        st.info("Calculate design first to see siphon analysis")

# Tab 5: Air Valve Design
with tab5:
    st.subheader("Air Valve Design and Sizing")
    
    if st.session_state.results:
        r = st.session_state.results
        
        if r['has_high_points']:
            st.success(f"✅ {len(r['air_valve_data'])} air valve location(s) identified")
            
            if r['siphon_data']['exists']:
                st.error("⚠️ **CRITICAL:** Air valves are MANDATORY at high point(s) in siphon systems!")
                st.markdown("""
                **Air Valve Functions in Siphon Systems:**
                1. **During Filling:** Release air as pipe fills
                2. **During Operation:** Release accumulated air pockets
                3. **During Shutdown:** Admit air to break siphon and prevent backflow
                4. **Vacuum Relief:** Prevent pipe collapse from vacuum
                """)
            
            for i, valve in enumerate(r['air_valve_data'], 1):
                with st.expander(f"Air Valve #{i} - Station {valve['station']:.0f} ft (Elev: {valve['elevation']:.1f} ft) - {valve['flow_regime']}", expanded=True):
                    st.markdown(f"**Location:** {valve['description']}")
                    st.markdown(f"**Flow Regime:** {valve['flow_regime']}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### Valve Specifications")
                        valve_info = valve['valve_info']
                        
                        st.markdown(f"**Valve Type:** {valve_info['valve_type']}")
                        st.markdown(f"**Primary Function:** {valve_info['primary_function']}")
                        st.markdown(f"**Connection Size:** {valve_info['connection_size']}")
                        
                        st.markdown("---")
                        st.markdown("**Hydraulic Conditions:**")
                        st.write(f"• HGL: **{valve['hgl']:.2f} ft**")
                        st.write(f"• Pressure Head: **{valve['pressure_head']:.2f} ft**")
                        
                        if valve['pressure_head'] < 0:
                            vacuum_psi = abs(valve['pressure_head']) * 0.433
                            st.error(f"• Vacuum: **{abs(valve['pressure_head']):.1f} ft ({vacuum_psi:.1f} psi)**")
                        
                        st.markdown("---")
                        st.markdown("**Orifice Sizes:**")
                        st.write(f"• Air Release Orifice: **{valve_info['air_release_orifice']}**")
                        st.write(f"• Vacuum Relief Orifice: **{valve_info['vacuum_orifice']}**")
                        
                        st.markdown("---")
                        st.markdown("**Capacity Requirements:**")
                        st.write(f"• Air Release: **{valve_info['air_release_capacity_cfm']:.1f} CFM**")
                        st.write(f"• Vacuum Relief: **{valve_info['vacuum_relief_capacity_cfm']:.1f} CFM**")
                        st.write(f"• Air Elimination: **{valve_info['air_elimination_capacity_cfm']:.1f} CFM**")
                    
                    with col2:
                        st.markdown("### Installation")
                        st.info("""
    **AIR VALVE**
         |
    Isolation 
      Valve
         |
    === PIPE ===
                        """)
                    
                    st.info(f"**Installation Notes:** {valve_info['installation_notes']}")
                    
                    if valve['flow_regime'] == "Gravity/Siphon":
                        st.error("""
                        **⚠️ CRITICAL FOR SIPHON ZONE:**
                        • MUST admit air when pump stops
                        • Prevents siphon backflow
                        • Prevents vacuum pipe collapse
                        • Combination valve REQUIRED
                        • Regular inspection MANDATORY
                        """)
                    
                    st.warning("""
                    **Design Considerations:**
                    • Install at least 12 inches above pipe crown
                    • Provide isolation valve for maintenance
                    • Ensure drain connection to prevent freezing
                    • Consider vandalism protection
                    • Verify manufacturer capacity ratings
                    • Test vacuum relief function regularly
                    """)
            
            # Summary table
            st.markdown("---")
            st.markdown("### Air Valve Summary Table")
            
            valve_summary = []
            for i, valve in enumerate(r['air_valve_data'], 1):
                valve_summary.append({
                    'Valve #': i,
                    'Station (ft)': valve['station'],
                    'Elevation (ft)': valve['elevation'],
                    'Flow Regime': valve['flow_regime'],
                    'Pressure (ft)': f"{valve['pressure_head']:.2f}",
                    'Type': valve['valve_info']['valve_type'],
                    'Connection': valve['valve_info']['connection_size']
                })
            
            valve_summary_df = pd.DataFrame(valve_summary)
            st.dataframe(valve_summary_df, use_container_width=True)
            
        else:
            st.info("No intermediate high points detected - no air valves required at intermediate locations")
            st.markdown("""
            **Note:** Air valves may still be required at:
            - Pump discharge (for air release during startup)
            - Any manual high points in the system
            - Long horizontal runs (every 1000-2000 ft)
            """)
    else:
        st.info("Calculate design first to see air valve requirements")
# Tab 6: Analysis & HGL
with tab6:
    if st.session_state.results:
        r = st.session_state.results
        
        st.subheader("Force Main System Profile with Corrected HGL")
        
        fig = go.Figure()
        
        # Ground/pipe profile
        fig.add_trace(go.Scatter(
            x=r['distances'], 
            y=r['elevations'], 
            fill='tozeroy', 
            name='Force Main Profile',
            line=dict(color='brown', width=3),
            fillcolor='rgba(139, 69, 19, 0.3)'
        ))
        
        # Hydraulic grade line - CORRECTED
        fig.add_trace(go.Scatter(
            x=r['distances'], 
            y=r['hgl_values'], 
            name='Hydraulic Grade Line (HGL)',
            line=dict(color='red', dash='dash', width=3),
            mode='lines+markers'
        ))
        
        # Mark pump station
        fig.add_trace(go.Scatter(
            x=[r['distances'][0]], 
            y=[r['elevations'][0]],
            mode='markers+text',
            name='Pump Station',
            marker=dict(size=15, color='green', symbol='square'),
            text=['Pump<br>Station'],
            textposition='top center',
            showlegend=False
        ))
        
        # Mark controlling point
        if r['high_point_controls']:
            control_idx = np.argmax(r['elevations'])
            fig.add_trace(go.Scatter(
                x=[r['controlling_distance']], 
                y=[r['controlling_elevation']],
                mode='markers+text',
                name='Controlling Point (High Point)',
                marker=dict(size=18, color='red', symbol='star'),
                text=[f'HIGH POINT<br>{r["controlling_elevation"]:.0f} ft'],
                textposition='top center',
                showlegend=False
            ))
            
            # Show siphon zone
            siphon_distances = [d for d in r['distances'] if d >= r['controlling_distance']]
            siphon_elevations = [r['elevations'][i] for i, d in enumerate(r['distances']) if d >= r['controlling_distance']]
            
            fig.add_trace(go.Scatter(
                x=siphon_distances,
                y=siphon_elevations,
                mode='lines',
                name='Siphon/Gravity Zone',
                line=dict(color='purple', width=5, dash='dot'),
                showlegend=True
            ))
        
        # Mark high points with air valves
        if r['has_high_points']:
            for idx in r['high_point_indices']:
                fig.add_trace(go.Scatter(
                    x=[r['distances'][idx]], 
                    y=[r['elevations'][idx]],
                    mode='markers+text',
                    name=f'Air Valve Required',
                    marker=dict(size=15, color='orange', symbol='triangle-up'),
                    text=[f'AIR VALVE<br>{r["elevations"][idx]:.0f} ft'],
                    textposition='bottom center',
                    showlegend=False
                ))
        
        # Mark discharge - VERIFICATION POINT
        fig.add_trace(go.Scatter(
            x=[r['distances'][-1]], 
            y=[r['elevations'][-1]],
            mode='markers+text',
            name='Discharge (P=0)',
            marker=dict(size=15, color='blue', symbol='diamond'),
            text=['Discharge<br>P=0<br>HGL=Elevation'],
            textposition='top center',
            showlegend=False
        ))
        
        # Add line showing HGL = discharge elevation at discharge
        fig.add_hline(
            y=r['discharge_elevation'], 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"HGL = Discharge Elevation = {r['discharge_elevation']:.1f} ft",
            annotation_position="right"
        )
        
        fig.update_layout(
            xaxis_title="Distance Along Forcemain (feet)",
            yaxis_title="Elevation (ft)",
            height=600,
            hovermode='x unified',
            showlegend=True
        )
        
        # Add verification annotation
        annotation_text = (
            f"<b>HGL VERIFICATION:</b><br>"
            f"HGL at Discharge: {r['hgl_values'][-1]:.6f} ft<br>"
            f"Discharge Elevation: {r['discharge_elevation']:.6f} ft<br>"
            f"Error: {r['hgl_error']:.8f} ft<br>"
            f"Status: {'✅ PASS' if r['hgl_error'] < 0.1 else '❌ FAIL'}<br><br>"
            f"<b>System:</b><br>"
            f"TDH: {r['TDH_design']:.1f} ft<br>"
            f"Flow: {r['Q_peak']:.0f} GPM<br>"
            f"Velocity: {r['pipe_velocity']:.2f} ft/s<br>"
            f"Flow Regime: {r['flow_regime'][:15]}..."
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=annotation_text,
            showarrow=False,
            bgcolor="lightgreen" if r['hgl_error'] < 0.1 else "lightcoral",
            bordercolor="black",
            borderwidth=1,
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

            # Pressure profile chart
        st.markdown("---")
        st.subheader("Pressure Head Profile")
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=r['distances'],
            y=r['pressure_values'],
            mode='lines+markers',
            name='Pressure Head',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            fill='tozeroy'
        ))
        
        fig2.add_hline(y=0, line_dash="solid", line_color="black", annotation_text="Atmospheric (0 ft)")
        
        if r['siphon_data']['exists']:
            fig2.add_hline(
                y=-r['siphon_data']['max_vacuum_capacity'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Max Vacuum Limit (-{r['siphon_data']['max_vacuum_capacity']:.0f} ft)"
            )
            
            # Shade siphon zone
            fig2.add_vrect(
                x0=r['controlling_distance'], 
                x1=r['total_length'],
                fillcolor="purple", 
                opacity=0.1,
                annotation_text="Siphon Zone", 
                annotation_position="top left"
            )
        
        fig2.update_layout(
            xaxis_title="Distance (ft)",
            yaxis_title="Pressure Head (ft)",
            height=400,
            hovermode='x'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Pump curves
        st.markdown("---")
        st.subheader("Pump Performance Curves")
        
        Q_range = np.linspace(0, r['Q_pump_gpm'] * 1.5, 100)
        H_shutoff = r['TDH_design'] * 1.2
        H_curve = H_shutoff - (H_shutoff - r['TDH_design']) * (Q_range / r['Q_pump_gpm'])**1.5
        H_system = r['static_head'] + (r['TDH_design'] - r['static_head']) * (Q_range / r['Q_pump_gpm'])**2
        eff_curve = r['pump_eff'] * 100 * np.exp(-((Q_range/r['Q_pump_gpm'] - 1)**2) / 0.5)
        
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Pump Curve', 'Efficiency Curve'))
        
        fig3.add_trace(go.Scatter(x=Q_range, y=H_curve, name='Pump Curve', 
                                  line=dict(color='blue', width=3)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=Q_range, y=H_system, name='System Curve', 
                                  line=dict(color='red', dash='dash', width=2)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=[r['Q_pump_gpm']], y=[r['TDH_design']], name='Operating Point', 
                                  mode='markers', marker=dict(size=15, color='green')), row=1, col=1)
        
        fig3.add_trace(go.Scatter(x=Q_range, y=eff_curve, name='Efficiency', 
                                  line=dict(color='green', width=3)), row=1, col=2)
        fig3.add_vline(x=r['Q_pump_gpm'], line_dash="dash", line_color="red", row=1, col=2)
        
        fig3.update_xaxes(title_text="Flow Rate (GPM)", row=1, col=1)
        fig3.update_xaxes(title_text="Flow Rate (GPM)", row=1, col=2)
        fig3.update_yaxes(title_text="Head (ft)", row=1, col=1)
        fig3.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        fig3.update_layout(height=400, showlegend=True)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # EPA Diurnal Flow Pattern
        st.markdown("---")
        st.subheader("EPA Diurnal Flow Pattern")
        
        hours = np.arange(0, 24, 0.25)
        peaking_factors = [get_diurnal_peaking_factor(h) for h in hours]
        flows = [r['Q_avg'] * pf for pf in peaking_factors]
        
        fig4 = make_subplots(rows=2, cols=1, subplot_titles=('Flow Pattern', 'Peaking Factors'),
                           vertical_spacing=0.12)
        
        fig4.add_trace(go.Scatter(x=hours, y=flows, fill='tozeroy', name='Diurnal Flow',
                                line=dict(color='blue', width=2)), row=1, col=1)
        fig4.add_hline(y=r['Q_avg'], line_dash="dash", line_color="green", 
                     annotation_text="Average", row=1, col=1)
        fig4.add_hline(y=r['Q_peak'], line_dash="dash", line_color="red", 
                     annotation_text="Design Peak", row=1, col=1)
        
        fig4.add_trace(go.Scatter(x=hours, y=peaking_factors, fill='tozeroy', 
                                name='Peaking Factor', line=dict(color='purple', width=2)), row=2, col=1)
        fig4.add_hline(y=1.0, line_dash="dash", line_color="green", row=2, col=1)
        
        fig4.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig4.update_yaxes(title_text="Flow (GPM)", row=1, col=1)
        fig4.update_yaxes(title_text="Peaking Factor", row=2, col=1)
        fig4.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig4, use_container_width=True)
        
    else:
        st.info("Calculate design first to see analysis")

# Tab 7: Export
with tab7:
    if st.session_state.results:
        r = st.session_state.results
        
        st.subheader("Export Design Results")
        
        # HGL Verification Status - PROMINENT DISPLAY
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if r['hgl_error'] < 0.001:
                st.success(f"✅ HGL VERIFICATION: PERFECT MATCH")
                st.success(f"HGL at discharge = {r['hgl_values'][-1]:.6f} ft")
                st.success(f"Discharge elevation = {r['discharge_elevation']:.6f} ft")
                st.success(f"Error = {r['hgl_error']:.8f} ft")
            elif r['hgl_error'] < 0.1:
                st.success(f"✅ HGL VERIFICATION: PASSED")
                st.info(f"Error = {r['hgl_error']:.6f} ft")
            else:
                st.error(f"❌ HGL VERIFICATION: FAILED")
                st.error(f"Error = {r['hgl_error']:.6f} ft")
        
        st.markdown("---")
        
        # Summary data
        summary_data = {
            'Parameter': [
                'HGL Verification Status',
                'HGL at Discharge (ft)',
                'Discharge Elevation (ft)', 
                'HGL Error (ft)',
                'Flow Regime',
                'Total Dynamic Head (calculated)',
                'Total Dynamic Head (design)',
                'Safety Factor',
                'Average Flow', 'Peak Flow', 'Pipe Velocity',
                'Total Forcemain Length', 'Design Length (for TDH)', 
                'Number of Elevation Points', 'Number of High Points',
                'Pump Elevation', 'Discharge Elevation', 'Controlling Elevation',
                'High Point Controls Design',
                'Static Head', 'Friction Loss (to control)', 'Minor Losses (to control)',
                'Siphon Exists', 'Siphon Length', 'Min Pressure in Siphon',
                'Number of Pumps', 'Pump Capacity', 'Motor Size', 'Power per Pump',
                'Wet Well Diameter', 'Storage Volume', 'Cycles per Hour',
                'Air Valves Required'
            ],
            'Value': [
                'PERFECT' if r['hgl_error'] < 0.001 else ('PASS' if r['hgl_error'] < 0.1 else 'FAIL'),
                f"{r['hgl_values'][-1]:.8f}",
                f"{r['discharge_elevation']:.8f}",
                f"{r['hgl_error']:.8f}",
                r['flow_regime'],
                f"{r['TDH_calculated']:.2f} ft",
                f"{r['TDH_design']:.2f} ft",
                f"{r['safety_factor']:.2f}",
                f"{r['Q_avg']:.0f} GPM", f"{r['Q_peak']:.0f} GPM", f"{r['pipe_velocity']:.2f} ft/s",
                f"{r['total_length']:.0f} ft", f"{r['design_length']:.0f} ft",
                len(r['distances']), len(r['air_valve_data']),
                f"{r['pump_elevation']:.1f} ft", f"{r['discharge_elevation']:.1f} ft", 
                f"{r['controlling_elevation']:.1f} ft",
                "YES" if r['high_point_controls'] else "NO",
                f"{r['static_head']:.1f} ft", f"{r['friction_loss']:.2f} ft", 
                f"{r['minor_losses']:.2f} ft",
                "YES" if r['siphon_data']['exists'] else "NO",
                f"{r['siphon_data']['length']:.0f} ft" if r['siphon_data']['exists'] else "N/A",
                f"{r['siphon_data']['min_pressure']:.2f} ft" if r['siphon_data']['exists'] else "N/A",
                r['num_pumps'], f"{r['Q_pump_gpm']:.0f} GPM", 
                f"{r['motor_size']:.1f} HP", f"{r['power_kw']:.1f} kW",
                f"{r['wetwell_diameter']:.1f} ft", f"{r['storage_volume_gal']:.0f} gal",
                f"{r['actual_cycles_per_hour']:.1f}",
                len(r['air_valve_data'])
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="Download Summary (CSV)",
                data=csv,
                file_name=f"lift_station_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Segment data
                segment_export = pd.DataFrame(r['segment_data'])
                segment_export['HGL'] = r['hgl_values']
                segment_export['Pressure'] = r['pressure_values']
                segment_export.to_excel(writer, sheet_name='Segment Analysis', index=False)
                
                # Minor losses
                st.session_state.minor_loss_components.to_excel(writer, sheet_name='Minor Losses', index=False)
                
                # Elevation profile
                r['elevation_profile'].to_excel(writer, sheet_name='Elevation Profile', index=False)
                
                # Air valves
                if r['air_valve_data']:
                    valve_export = []
                    for i, valve in enumerate(r['air_valve_data'], 1):
                        valve_export.append({
                            'Valve Number': i,
                            'Station (ft)': valve['station'],
                            'Elevation (ft)': valve['elevation'],
                            'Flow Regime': valve['flow_regime'],
                            'HGL (ft)': valve['hgl'],
                            'Pressure (ft)': valve['pressure_head'],
                            'Description': valve['description'],
                            'Valve Type': valve['valve_info']['valve_type'],
                            'Connection Size': valve['valve_info']['connection_size'],
                            'Air Release Orifice': valve['valve_info']['air_release_orifice'],
                            'Vacuum Orifice': valve['valve_info']['vacuum_orifice']
                        })
                    pd.DataFrame(valve_export).to_excel(writer, sheet_name='Air Valves', index=False)
                
                # Siphon data
                if r['siphon_data']['exists']:
                    siphon_export = pd.DataFrame([{
                        'Siphon Start Distance (ft)': r['siphon_data']['start_distance'],
                        'Siphon Start Elevation (ft)': r['siphon_data']['start_elevation'],
                        'Siphon Length (ft)': r['siphon_data']['length'],
                        'Elevation Drop (ft)': r['siphon_data']['elevation_drop'],
                        'Min Pressure (ft)': r['siphon_data']['min_pressure'],
                        'Min Pressure Location (ft)': r['siphon_data']['min_pressure_location'],
                        'Max Vacuum Capacity (ft)': r['siphon_data']['max_vacuum_capacity'],
                        'Vacuum Margin (ft)': r['siphon_data']['vacuum_margin'],
                        'Is Stable': 'YES' if r['siphon_data']['is_stable'] else 'NO'
                    }])
                    siphon_export.to_excel(writer, sheet_name='Siphon Analysis', index=False)
                
                # HGL Verification
                hgl_verification = pd.DataFrame([{
                    'HGL at Discharge (ft)': f"{r['hgl_at_discharge']:.8f}",
                    'Discharge Elevation (ft)': f"{r['discharge_elevation']:.8f}",
                    'HGL Error (ft)': f"{r['hgl_error']:.8f}",
                    'Verification Status': 'PERFECT' if r['hgl_error'] < 0.001 else ('PASS' if r['hgl_error'] < 0.1 else 'FAIL'),
                    'TDH Calculated (ft)': r['TDH_calculated'],
                    'TDH Design (ft)': r['TDH_design'],
                    'Safety Factor': r['safety_factor'],
                    'Calculation Method': 'Backwards from discharge where P=0'
                }])
                hgl_verification.to_excel(writer, sheet_name='HGL Verification', index=False)
            
            st.download_button(
                label="Download Full Report (Excel)",
                data=buffer.getvalue(),
                file_name=f"lift_station_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Technical specification report
            spec_text = "LIFT STATION DESIGN REPORT\n"
            spec_text += "="*80 + "\n\n"
            spec_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            spec_text += "HGL VERIFICATION\n"
            spec_text += "-"*80 + "\n"
            spec_text += f"HGL at Discharge: {r['hgl_at_discharge']:.8f} ft\n"
            spec_text += f"Discharge Elevation: {r['discharge_elevation']:.8f} ft\n"
            spec_text += f"HGL Error: {r['hgl_error']:.8f} ft\n"
            if r['hgl_error'] < 0.001:
                spec_text += f"Verification: PERFECT MATCH\n"
            elif r['hgl_error'] < 0.1:
                spec_text += f"Verification: PASS\n"
            else:
                spec_text += f"Verification: FAIL\n"
            spec_text += f"Method: Backwards calculation from discharge (P=0)\n\n"
            
            spec_text += "HYDRAULIC DESIGN SUMMARY\n"
            spec_text += "-"*80 + "\n"
            spec_text += f"Flow Regime: {r['flow_regime']}\n"
            spec_text += f"Total Dynamic Head (calculated): {r['TDH_calculated']:.2f} ft\n"
            spec_text += f"Total Dynamic Head (design): {r['TDH_design']:.2f} ft\n"
            spec_text += f"Safety Factor: {r['safety_factor']:.2f}\n"
            spec_text += f"Design Length: {r['design_length']:.0f} ft\n"
            spec_text += f"Total Length: {r['total_length']:.0f} ft\n"
            spec_text += f"Static Head: {r['static_head']:.1f} ft\n"
            spec_text += f"Friction Loss: {r['friction_loss']:.2f} ft\n"
            spec_text += f"Minor Losses: {r['minor_losses']:.2f} ft\n\n"
            
            if r['siphon_data']['exists']:
                spec_text += "SIPHON ANALYSIS\n"
                spec_text += "-"*80 + "\n"
                spec_text += f"Siphon Length: {r['siphon_data']['length']:.0f} ft\n"
                spec_text += f"Elevation Drop: {r['siphon_data']['elevation_drop']:.1f} ft\n"
                spec_text += f"Minimum Pressure: {r['siphon_data']['min_pressure']:.2f} ft\n"
                spec_text += f"Vacuum Margin: {r['siphon_data']['vacuum_margin']:.1f} ft\n"
                spec_text += f"Stability: {'STABLE' if r['siphon_data']['is_stable'] else 'UNSTABLE'}\n\n"
            
            spec_text += "PUMP SPECIFICATIONS\n"
            spec_text += "-"*80 + "\n"
            spec_text += f"Number of Pumps: {r['num_pumps']}\n"
            spec_text += f"Operating Pumps: {r['pumps_operating']} (N-1 redundancy)\n"
            spec_text += f"Capacity (each): {r['Q_pump_gpm']:.0f} GPM @ {r['TDH_design']:.1f} ft TDH\n"
            spec_text += f"Motor Size: {r['motor_size']:.1f} HP\n"
            spec_text += f"Power: {r['power_kw']:.1f} kW\n\n"
            
            if r['air_valve_data']:
                spec_text += "AIR VALVE SCHEDULE\n"
                spec_text += "-"*80 + "\n"
                for i, valve in enumerate(r['air_valve_data'], 1):
                    spec_text += f"\nAir Valve #{i}\n"
                    spec_text += f"  Location: Station {valve['station']:.0f} ft, Elevation {valve['elevation']:.1f} ft\n"
                    spec_text += f"  Flow Regime: {valve['flow_regime']}\n"
                    spec_text += f"  Type: {valve['valve_info']['valve_type']}\n"
                    spec_text += f"  Connection: {valve['valve_info']['connection_size']}\n"
                    spec_text += f"  HGL: {valve['hgl']:.2f} ft\n"
                    spec_text += f"  Pressure: {valve['pressure_head']:.2f} ft\n"
            
            spec_text += "\n" + "="*80 + "\n"
            spec_text += "Report generated by Lift Station Sizing Tool v4.0\n"
            spec_text += "Smith & Loveless Methodology with CORRECTED HGL Calculation\n"
            spec_text += "HGL calculated backwards from discharge where P=0\n"
            spec_text += "="*80 + "\n"
            
            st.download_button(
                label="Download Technical Specs (TXT)",
                data=spec_text,
                file_name=f"lift_station_specs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Calculate design first to export results")

# Footer
st.markdown("---")
st.markdown("**Lift Station Sizing Tool** | Professional Engineering Design Software | Version 4.0")
st.caption("With Siphon Flow Analysis, Multi-Point Elevation Profile, Smith & Loveless Methodology, and CORRECTED HGL Calculation")
st.caption("✅ **CORRECTED:** HGL at discharge now properly equals discharge elevation (P=0)")
st.caption("🔧 **Method:** Backwards calculation from discharge ensures hydraulic accuracy")
        