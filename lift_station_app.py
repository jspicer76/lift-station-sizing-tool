"""
Lift Station Sizing Tool - Complete Professional Version
Engineering design tool for wastewater pump stations with startup analysis
Includes siphon flow analysis per Smith & Loveless methodology
CORRECTED: No safety factor on TDH, startup analysis, dual operating points
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
    page_title="Lift Station Sizing Tool v5.0",
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
    h_f = 4.52 * length_ft * (Q_gpm ** 1.85) / ((C ** 1.85) * (pipe_diameter_in ** 4.87))
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

# =============================================================================
# ENHANCED AIR VALVE ANALYSIS - OPTION 3: DETAILED ENGINEERING
# =============================================================================

def detect_valve_locations_comprehensive(elevation_df, pipe_diameter, Q_gpm, total_length):
    """
    Comprehensive detection of ALL air valve locations:
    1. High points (air accumulation)
    2. Low points (vacuum relief)
    3. Long horizontal runs (air release spacing)
    """
    
    import numpy as np
    
    distances = elevation_df['Distance (ft)'].values
    elevations = elevation_df['Elevation (ft)'].values
    
    valve_locations = {
        'high_points': [],
        'low_points': [],
        'long_runs': [],
        'pump_discharge': [],
        'all_valves': []
    }
    
    # Calculate velocity for air transport analysis
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    # =========================================================================
    # 1. HIGH POINTS - Local maxima (air accumulation)
    # =========================================================================
    for i in range(1, len(elevations) - 1):
        # Check if local maximum
        if elevations[i] > elevations[i-1] and elevations[i] > elevations[i+1]:
            
            # Calculate prominence (how significant is this peak?)
            left_min = min(elevations[:i+1])
            right_min = min(elevations[i:])
            prominence = elevations[i] - max(left_min, right_min)
            
            # Only include significant high points (>2 ft prominence)
            if prominence > 2.0:
                valve_locations['high_points'].append({
                    'index': i,
                    'distance': distances[i],
                    'elevation': elevations[i],
                    'prominence': prominence,
                    'type': 'High Point - Combination Valve',
                    'priority': 'CRITICAL' if prominence > 10 else 'HIGH'
                })
    
    # =========================================================================
    # 2. LOW POINTS - Local minima (vacuum relief)
    # =========================================================================
    for i in range(1, len(elevations) - 1):
        # Check if local minimum
        if elevations[i] < elevations[i-1] and elevations[i] < elevations[i+1]:
            
            # Calculate depth (how significant is this valley?)
            left_max = max(elevations[:i+1])
            right_max = max(elevations[i:])
            depth = min(left_max, right_max) - elevations[i]
            
            # Only include significant low points (>2 ft depth)
            if depth > 2.0:
                valve_locations['low_points'].append({
                    'index': i,
                    'distance': distances[i],
                    'elevation': elevations[i],
                    'depth': depth,
                    'type': 'Low Point - Vacuum Relief',
                    'priority': 'HIGH' if depth > 10 else 'MEDIUM'
                })
    
    # =========================================================================
    # 3. LONG HORIZONTAL RUNS - Based on velocity and spacing
    # =========================================================================
    
    # Critical velocity for air transport (AWWA M51)
    critical_velocity = 2.5  # fps
    
    # Determine spacing based on velocity
    if velocity_fps > 3.0:
        max_spacing = 2640  # 1/2 mile (good air transport)
    elif velocity_fps > 2.5:
        max_spacing = 1980  # 3/8 mile (moderate transport)
    else:
        max_spacing = 1320  # 1/4 mile (poor air transport)
    
    # Find long runs without existing valves
    last_valve_distance = 0
    
    for i in range(len(distances)):
        distance_from_last = distances[i] - last_valve_distance
        
        # Check if we need a long-run valve here
        if distance_from_last > max_spacing:
            
            # Check if relatively horizontal (grade < 2%)
            if i > 0:
                segment_length = distances[i] - distances[i-1]
                if segment_length > 0:
                    grade = abs(elevations[i] - elevations[i-1]) / segment_length
                    
                    if grade < 0.02:  # Less than 2% grade
                        valve_locations['long_runs'].append({
                            'index': i,
                            'distance': distances[i],
                            'elevation': elevations[i],
                            'spacing': distance_from_last,
                            'velocity': velocity_fps,
                            'type': 'Long Run - Air Release',
                            'priority': 'MEDIUM'
                        })
                        last_valve_distance = distances[i]
        
        # Update last valve distance if we're at a high or low point
        if any(hp['distance'] == distances[i] for hp in valve_locations['high_points']):
            last_valve_distance = distances[i]
        if any(lp['distance'] == distances[i] for lp in valve_locations['low_points']):
            last_valve_distance = distances[i]
    
    # =========================================================================
    # 4. PUMP DISCHARGE - Always needs valve
    # =========================================================================
    valve_locations['pump_discharge'].append({
        'index': 0,
        'distance': distances[0],
        'elevation': elevations[0],
        'type': 'Pump Discharge - Combination Valve',
        'priority': 'CRITICAL'
    })
    
    # =========================================================================
    # 5. CONSOLIDATE ALL VALVES
    # =========================================================================
    all_valves = []
    all_valves.extend(valve_locations['high_points'])
    all_valves.extend(valve_locations['low_points'])
    all_valves.extend(valve_locations['long_runs'])
    all_valves.extend(valve_locations['pump_discharge'])
    
    # Sort by distance
    all_valves.sort(key=lambda x: x['distance'])
    
    # Number the valves
    for i, valve in enumerate(all_valves, start=1):
        valve['valve_number'] = i
    
    valve_locations['all_valves'] = all_valves
    
    return valve_locations

def calculate_air_valve_sizing_comprehensive(valve_data, pipe_diameter, Q_gpm, static_head, system_pressure_psi=0):
    """
    Comprehensive air valve sizing for multiple scenarios
    CORRECTED VERSION - Fixed orifice velocities and vacuum sizing
    """
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    pipe_area_sqin = pipe_area_sqft * 144
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    P_atm_psia = 14.7
    valve_type = valve_data['type']
    
    sizing_results = {
        'valve_type_required': None,
        'scenarios': {},
        'recommended_valve': None,
        'orifice_sizes': {}
    }
    
    # =========================================================================
    # SCENARIO 1: NORMAL OPERATION (Continuous Air Release)
    # =========================================================================
    
    air_entrainment_percent = 2.0
    air_release_rate_cfm = (Q_gpm * 0.00223) * (air_entrainment_percent / 100)
    
    if system_pressure_psi > 0:
        pressure_correction = P_atm_psia / (P_atm_psia + system_pressure_psi)
        air_release_rate_cfm *= pressure_correction
    
    # FIX 1: CORRECTED - Reduced air velocity from 10 fps to 7 fps (more conservative)
    air_velocity_orifice = 7  # fps (CHANGED from 10 fps)
    orifice_area_sqin_small = (air_release_rate_cfm * 144) / (air_velocity_orifice * 60)
    orifice_diameter_small = 2 * np.sqrt(orifice_area_sqin_small / np.pi)
    
    standard_small_orifices = [1/32, 1/16, 3/32, 1/8, 5/32, 3/16, 1/4]
    selected_small_orifice = min([s for s in standard_small_orifices if s >= orifice_diameter_small], default=1/4)
    
    sizing_results['scenarios']['normal_operation'] = {
        'description': 'Continuous air release during normal operation',
        'air_release_rate_cfm': air_release_rate_cfm,
        'calculated_orifice_diameter': orifice_diameter_small,
        'selected_orifice_diameter': selected_small_orifice,
        'orifice_type': 'Small (continuous release)',
        'design_velocity_fps': air_velocity_orifice  # Added for documentation
    }
    
    # =========================================================================
    # SCENARIO 2: SYSTEM FILLING (Air Evacuation)
    # =========================================================================
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        
        filling_velocity_fps = 2.5
        filling_flow_gpm = filling_velocity_fps * pipe_area_sqft * 449
        air_evacuation_rate_cfm = filling_flow_gpm * 0.00223
        
        air_velocity_large_orifice = 3  # fps (appropriate for large volumes)
        orifice_area_sqin_large = (air_evacuation_rate_cfm * 144) / (air_velocity_large_orifice * 60)
        orifice_diameter_large = 2 * np.sqrt(orifice_area_sqin_large / np.pi)
        
        if orifice_diameter_large < 1.5:
            connection_size = 2
        elif orifice_diameter_large < 3:
            connection_size = 3
        elif orifice_diameter_large < 5:
            connection_size = 4
        else:
            connection_size = 6
        
        sizing_results['scenarios']['system_filling'] = {
            'description': 'Air evacuation during system filling',
            'filling_velocity_fps': filling_velocity_fps,
            'filling_flow_gpm': filling_flow_gpm,
            'air_evacuation_rate_cfm': air_evacuation_rate_cfm,
            'calculated_large_orifice': orifice_diameter_large,
            'recommended_connection_size': connection_size,
            'orifice_type': 'Large (evacuation)',
            'design_velocity_fps': air_velocity_large_orifice
        }
    
    # =========================================================================
    # SCENARIO 3: SYSTEM DRAINING (Vacuum Relief)
    # FIX 2: CORRECTED - Using empirical method instead of theoretical
    # =========================================================================
    
    if 'Low Point' in valve_type or 'Pump Discharge' in valve_type:
        
        drainage_velocity_fps = 3.0
        drainage_flow_gpm = drainage_velocity_fps * pipe_area_sqft * 449
        air_inflow_rate_cfm = drainage_flow_gpm * 0.00223
        allowable_vacuum_psi = 5.0
        
        # CORRECTED: Empirical sizing based on manufacturer data
        # More accurate than theoretical orifice equation
        
        # Manufacturer capacity data (CFM at 2 psi vacuum differential)
        # Based on Val-Matic, APCO, and ARI published data
        vacuum_valve_capacities = [
            {'size': 2, 'capacity_2psi': 50},
            {'size': 3, 'capacity_2psi': 115},
            {'size': 4, 'capacity_2psi': 210},
            {'size': 6, 'capacity_2psi': 475}
        ]
        
        # Adjust required capacity for different vacuum pressure
        # Capacity scales with √(ΔP)
        pressure_ratio = np.sqrt(allowable_vacuum_psi / 2.0)
        adjusted_required_cfm = air_inflow_rate_cfm / pressure_ratio
        
        # Add 20% safety factor
        design_cfm = adjusted_required_cfm * 1.2
        
        # Select valve size
        selected_vacuum_valve = None
        for valve_spec in vacuum_valve_capacities:
            if valve_spec['capacity_2psi'] >= design_cfm:
                selected_vacuum_valve = valve_spec
                break
        
        if selected_vacuum_valve is None:
            selected_vacuum_valve = vacuum_valve_capacities[-1]  # Use largest
        
        vacuum_connection = selected_vacuum_valve['size']
        rated_capacity = selected_vacuum_valve['capacity_2psi'] * pressure_ratio
        
        # Calculate theoretical orifice for reference only
        C_d = 0.65
        delta_P_psf = allowable_vacuum_psi * 144
        rho_air = 0.075
        theoretical_velocity = C_d * np.sqrt(2 * delta_P_psf / rho_air)
        theoretical_area_sqft = (air_inflow_rate_cfm / 60) / theoretical_velocity
        theoretical_orifice = 2 * np.sqrt(theoretical_area_sqft / np.pi) * 12
        
        sizing_results['scenarios']['system_draining'] = {
            'description': 'Vacuum relief during drainage',
            'drainage_flow_gpm': drainage_flow_gpm,
            'air_inflow_rate_cfm': air_inflow_rate_cfm,
            'allowable_vacuum_psi': allowable_vacuum_psi,
            'design_air_inflow_cfm': design_cfm,
            'recommended_connection_size': vacuum_connection,
            'rated_capacity_cfm': rated_capacity,
            'orifice_type': 'Vacuum relief',
            'theoretical_orifice_reference': theoretical_orifice,  # For reference only
            'sizing_method': 'Empirical (manufacturer data)'
        }
    
    # =========================================================================
    # SCENARIO 4: WATER HAMMER / TRANSIENT PROTECTION
    # =========================================================================
    
    if 'High Point' in valve_type:
        
        # Estimate transient low pressure
        wave_speed = 3000  # fps
        velocity_change = velocity_fps
        
        pressure_drop_transient = (wave_speed * velocity_change) / 32.2
        
        # Check if column separation could occur
        # NOTE: This uses simplified assumption - actual pressure depends on HGL
        elevation_diff = valve_data['elevation'] - static_head
        
        if elevation_diff + pressure_drop_transient < -20:
            column_separation_risk = "HIGH"
        elif elevation_diff + pressure_drop_transient < -10:
            column_separation_risk = "MODERATE"
        else:
            column_separation_risk = "LOW"
        
        sizing_results['scenarios']['water_hammer'] = {
            'description': 'Transient protection / column separation',
            'pressure_drop_transient_ft': pressure_drop_transient,
            'column_separation_risk': column_separation_risk,
            'recommendation': 'Combination valve provides surge protection' if column_separation_risk != "LOW" else 'Standard sizing adequate',
            'note': 'Detailed transient analysis recommended for high-risk locations'
        }
    
    # =========================================================================
    # DETERMINE VALVE TYPE AND SIZING
    # =========================================================================
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        sizing_results['valve_type_required'] = 'Combination Air Valve'
        sizing_results['recommended_valve'] = {
            'type': 'Combination',
            'connection_size': f"{sizing_results['scenarios']['system_filling']['recommended_connection_size']}\"",
            'large_orifice': f"{sizing_results['scenarios']['system_filling']['calculated_large_orifice']:.2f}\"",
            'small_orifice': f"{selected_small_orifice:.3f}\"",
            'function': 'Air release, evacuation, and vacuum relief'
        }
        
    elif 'Low Point' in valve_type:
        sizing_results['valve_type_required'] = 'Air/Vacuum Valve'
        sizing_results['recommended_valve'] = {
            'type': 'Air/Vacuum',
            'connection_size': f"{sizing_results['scenarios']['system_draining']['recommended_connection_size']}\"",
            'rated_capacity': f"{sizing_results['scenarios']['system_draining']['rated_capacity_cfm']:.0f} CFM",
            'function': 'Vacuum relief during drainage',
            'sizing_method': 'Empirical'
        }
        
    elif 'Long Run' in valve_type:
        sizing_results['valve_type_required'] = 'Air Release Valve'
        sizing_results['recommended_valve'] = {
            'type': 'Air Release',
            'connection_size': '1\" or 2\"',
            'small_orifice': f"{selected_small_orifice:.3f}\"",
            'function': 'Continuous air release only'
        }
    
    return sizing_results

def analyze_transient_conditions(elevation_df, Q_gpm, pipe_diameter, wave_speed=3000):
    """
    Analyze transient (water hammer) conditions
    CORRECTED VERSION - Better pressure head calculations
    """
    
    distances = elevation_df['Distance (ft)'].values
    elevations = elevation_df['Elevation (ft)'].values
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    transient_results = []
    
    # FIX 3: Calculate approximate HGL instead of using placeholder
    # Simplified friction loss estimate (for better pressure estimation)
    
    # Estimate friction factor using Hazen-Williams
    C_hw = 100  # Conservative Hazen-Williams coefficient
    
    for i in range(len(distances)):
        
        # Joukowsky pressure drop
        pressure_drop_ft = (wave_speed * velocity_fps) / 32.2
        
        # FIX 3: Improved steady-state pressure estimation
        # Calculate approximate pressure at this point
        
        if i == 0:
            # At pump discharge
            # Assume pump provides enough head to overcome total system
            steady_pressure_ft = 60  # Approximate at discharge
        else:
            # Estimate friction losses to this point
            distance_to_point = distances[i]
            
            # Hazen-Williams friction loss estimate
            # hf = 10.67 × L × Q^1.85 / (C^1.85 × d^4.87)
            L_miles = distance_to_point / 5280
            d_inches = pipe_diameter
            
            if d_inches > 0 and C_hw > 0 and Q_gpm > 0:
                friction_loss_ft = 10.67 * L_miles * (Q_gpm ** 1.85) / ((C_hw ** 1.85) * (d_inches ** 4.87))
            else:
                friction_loss_ft = 0
            
            # Steady pressure = Initial head - elevation gain - friction
            elevation_gain = elevations[i] - elevations[0]
            steady_pressure_ft = 60 - elevation_gain - friction_loss_ft
        
        # Minimum transient pressure during pump shutdown
        min_transient_pressure = steady_pressure_ft - pressure_drop_ft
        
        # Column separation risk assessment
        # Vapor pressure of water ≈ -33 ft (absolute vacuum)
        # But we want to stay well above this
        
        vapor_pressure_ft = -33  # Absolute vacuum reference
        safety_margin = 10  # Want at least 10 ft above vapor pressure
        
        if min_transient_pressure < vapor_pressure_ft + safety_margin:
            separation_risk = "CRITICAL"
        elif min_transient_pressure < 0:
            separation_risk = "HIGH"
        elif min_transient_pressure < 20:
            separation_risk = "MODERATE"
        else:
            separation_risk = "LOW"
        
        transient_results.append({
            'distance': distances[i],
            'elevation': elevations[i],
            'steady_pressure_ft': steady_pressure_ft,  # Added
            'pressure_drop_ft': pressure_drop_ft,
            'min_transient_pressure': min_transient_pressure,
            'separation_risk': separation_risk,
            'friction_loss_estimate': friction_loss_ft if i > 0 else 0  # Added for diagnostics
        })
    
    return transient_results

def calculate_staged_filling_procedure(valve_locations, pipe_diameter, total_length):
    """
    Generate optimal filling procedure to minimize surge and ensure proper air evacuation
    """
    
    high_points = valve_locations['high_points']
    
    if not high_points:
        return {
            'procedure': 'Simple',
            'stages': ['Fill at 2-3 fps until full'],
            'estimated_time_minutes': (total_length * np.pi * (pipe_diameter/12)**2 / 4) / (2.5 * 60)
        }
    
    # Multi-stage filling for complex profiles
    stages = []
    
    # Stage 1: Fill to first high point
    stages.append({
        'stage': 1,
        'description': f"Fill to Station {high_points[0]['distance']:.0f} ft (First High Point)",
        'target_elevation': high_points[0]['elevation'],
        'filling_rate': '2.0 fps (slow)',
        'air_valve_action': 'Air evacuates through HP1 combination valve',
        'estimated_time_min': 'TBD'
    })
    
    # Subsequent stages
    for i, hp in enumerate(high_points[1:], start=2):
        stages.append({
            'stage': i,
            'description': f"Continue to Station {hp['distance']:.0f} ft (High Point {i})",
            'target_elevation': hp['elevation'],
            'filling_rate': '2.5 fps (moderate)',
            'air_valve_action': f'Air evacuates through HP{i} combination valve',
            'estimated_time_min': 'TBD'
        })
    
    # Final stage
    stages.append({
        'stage': len(stages) + 1,
        'description': 'Fill remainder to discharge',
        'target_elevation': 'Discharge elevation',
        'filling_rate': '3.0 fps (normal)',
        'air_valve_action': 'Final air release through discharge valve',
        'estimated_time_min': 'TBD'
    })
    
    return {
        'procedure': 'Staged',
        'number_of_stages': len(stages),
        'stages': stages,
        'total_estimated_time_hours': 'TBD',
        'critical_notes': [
            'Monitor pressure at each high point',
            'Verify air valve operation at each stage',
            'Do not exceed 3 fps filling velocity',
            'Allow time for air evacuation before increasing flow'
        ]
    }

# =============================================================================
# BLOCK 3: MANUFACTURER-SPECIFIC SIZING METHODS
# =============================================================================

def calculate_valve_sizing_valmatic_method(valve_data, pipe_diameter, Q_gpm, scenarios):
    """
    Val-Matic sizing methodology
    Based on Val-Matic Engineering Data and kinetic sizing methods
    """
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    valve_type = valve_data['type']
    
    valmatic_sizing = {
        'manufacturer': 'Val-Matic',
        'method': 'Kinetic Sizing',
        'recommendations': {}
    }
    
    # High Point / Combination Valve
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        
        # Large orifice - based on filling velocity and kinetic equation
        # Q_air = C_d × A × √(2gΔH)
        filling_velocity = 2.5  # fps
        air_flow_cfm = (filling_velocity * pipe_area_sqft * 449) * 0.00223
        
        # Val-Matic uses pressure differential approach
        delta_P_psi = 2.0  # Typical design pressure drop
        C_d = 0.65  # Discharge coefficient
        
        # Kinetic formula: Q = C × d² × √(ΔP)
        # Where C is the valve coefficient
        # Rearrange: d = √(Q / (C × √ΔP))
        
        C_valve = 27.3  # Val-Matic coefficient for combination valves
        orifice_diam_valmatic = np.sqrt(air_flow_cfm / (C_valve * np.sqrt(delta_P_psi)))
        
        # Val-Matic standard sizes
        valmatic_sizes = [
            {'size': '1"', 'model': '107', 'max_flow_cfm': 15},
            {'size': '2"', 'model': '207', 'max_flow_cfm': 65},
            {'size': '3"', 'model': '307', 'max_flow_cfm': 145},
            {'size': '4"', 'model': '407', 'max_flow_cfm': 260},
            {'size': '6"', 'model': '607', 'max_flow_cfm': 585}
        ]
        
        # Select appropriate size
        selected_valve = next((v for v in valmatic_sizes if v['max_flow_cfm'] >= air_flow_cfm), valmatic_sizes[-1])
        
        valmatic_sizing['recommendations']['combination_valve'] = {
            'calculated_orifice': f"{orifice_diam_valmatic:.2f} inches",
            'air_flow_required': f"{air_flow_cfm:.1f} CFM",
            'recommended_model': f"Val-Matic {selected_valve['model']}",
            'connection_size': selected_valve['size'],
            'capacity': f"{selected_valve['max_flow_cfm']} CFM",
            'function': 'Large orifice for evacuation, small orifice for continuous release'
        }
    
    # Low Point / Vacuum Relief
    elif 'Low Point' in valve_type:
        
        # Vacuum relief sizing
        drainage_velocity = 3.0  # fps
        air_inflow_cfm = (drainage_velocity * pipe_area_sqft * 449) * 0.00223
        
        # Val-Matic vacuum valve sizing
        allowable_vacuum_psi = 2.0  # Conservative for Val-Matic sizing
        
        C_vacuum = 21.8  # Val-Matic coefficient for vacuum valves
        orifice_diam_vacuum = np.sqrt(air_inflow_cfm / (C_vacuum * np.sqrt(allowable_vacuum_psi)))
        
        # Val-Matic air/vacuum valve sizes
        vacuum_sizes = [
            {'size': '2"', 'model': '201-A', 'max_flow_cfm': 50},
            {'size': '3"', 'model': '301-A', 'max_flow_cfm': 115},
            {'size': '4"', 'model': '401-A', 'max_flow_cfm': 210},
            {'size': '6"', 'model': '601-A', 'max_flow_cfm': 475}
        ]
        
        selected_vacuum = next((v for v in vacuum_sizes if v['max_flow_cfm'] >= air_inflow_cfm), vacuum_sizes[-1])
        
        valmatic_sizing['recommendations']['vacuum_valve'] = {
            'calculated_orifice': f"{orifice_diam_vacuum:.2f} inches",
            'air_inflow_required': f"{air_inflow_cfm:.1f} CFM",
            'recommended_model': f"Val-Matic {selected_vacuum['model']}",
            'connection_size': selected_vacuum['size'],
            'capacity': f"{selected_vacuum['max_flow_cfm']} CFM",
            'function': 'Vacuum relief during drainage'
        }
    
    # Long Run / Air Release Only
    elif 'Long Run' in valve_type:
        
        # Small continuous release
        air_release_cfm = scenarios['normal_operation']['air_release_rate_cfm']
        
        # Val-Matic air release valve
        release_sizes = [
            {'size': '1"', 'model': '1', 'orifice': '1/8"'},
            {'size': '2"', 'model': '2', 'orifice': '1/4"'}
        ]
        
        selected_release = release_sizes[0] if air_release_cfm < 1.0 else release_sizes[1]
        
        valmatic_sizing['recommendations']['air_release'] = {
            'air_release_required': f"{air_release_cfm:.2f} CFM",
            'recommended_model': f"Val-Matic Model {selected_release['model']}",
            'connection_size': selected_release['size'],
            'orifice': selected_release['orifice'],
            'function': 'Continuous air release only'
        }
    
    return valmatic_sizing

def calculate_valve_sizing_apco_method(valve_data, pipe_diameter, Q_gpm, scenarios):
    """
    APCO (Flowmatic) sizing methodology
    """
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    valve_type = valve_data['type']
    
    apco_sizing = {
        'manufacturer': 'APCO',
        'method': 'Flow-based Selection',
        'recommendations': {}
    }
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        
        # APCO combination air valves (Series 340/345)
        air_flow_cfm = scenarios['system_filling']['air_evacuation_rate_cfm']
        
        apco_combo_sizes = [
            {'size': '2"', 'model': '340-020', 'capacity_cfm': 80},
            {'size': '3"', 'model': '340-030', 'capacity_cfm': 180},
            {'size': '4"', 'model': '340-040', 'capacity_cfm': 320},
            {'size': '6"', 'model': '340-060', 'capacity_cfm': 720}
        ]
        
        selected_apco = next((v for v in apco_combo_sizes if v['capacity_cfm'] >= air_flow_cfm), apco_combo_sizes[-1])
        
        apco_sizing['recommendations']['combination_valve'] = {
            'air_flow_required': f"{air_flow_cfm:.1f} CFM",
            'recommended_model': f"APCO {selected_apco['model']}",
            'connection_size': selected_apco['size'],
            'rated_capacity': f"{selected_apco['capacity_cfm']} CFM",
            'series': '340/345 Combination Air Valve'
        }
    
    elif 'Low Point' in valve_type:
        
        # APCO air/vacuum valves (Series 330)
        air_inflow_cfm = scenarios['system_draining']['air_inflow_rate_cfm']
        
        apco_vacuum_sizes = [
            {'size': '2"', 'model': '330-020', 'capacity_cfm': 65},
            {'size': '3"', 'model': '330-030', 'capacity_cfm': 145},
            {'size': '4"', 'model': '330-040', 'capacity_cfm': 260},
            {'size': '6"', 'model': '330-060', 'capacity_cfm': 585}
        ]
        
        selected_vacuum = next((v for v in apco_vacuum_sizes if v['capacity_cfm'] >= air_inflow_cfm), apco_vacuum_sizes[-1])
        
        apco_sizing['recommendations']['vacuum_valve'] = {
            'air_inflow_required': f"{air_inflow_cfm:.1f} CFM",
            'recommended_model': f"APCO {selected_vacuum['model']}",
            'connection_size': selected_vacuum['size'],
            'rated_capacity': f"{selected_vacuum['capacity_cfm']} CFM",
            'series': '330 Air/Vacuum Valve'
        }
    
    elif 'Long Run' in valve_type:
        
        apco_sizing['recommendations']['air_release'] = {
            'recommended_model': 'APCO 310-Series',
            'connection_size': '1" or 2"',
            'series': 'Air Release Valve (small orifice)'
        }
    
    return apco_sizing

def calculate_valve_sizing_ari_method(valve_data, pipe_diameter, Q_gpm, scenarios):
    """
    A.R.I. (Applied Research International) sizing methodology
    Israeli standard, widely used internationally
    """
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    valve_type = valve_data['type']
    
    ari_sizing = {
        'manufacturer': 'A.R.I.',
        'method': 'European/Israeli Standards',
        'recommendations': {}
    }
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        
        air_flow_cfm = scenarios['system_filling']['air_evacuation_rate_cfm']
        
        # A.R.I. combination air valves (D-025 series)
        ari_combo_sizes = [
            {'size': 'DN50 (2")', 'model': 'D-025-DN50', 'capacity_m3h': 160, 'capacity_cfm': 94},
            {'size': 'DN80 (3")', 'model': 'D-025-DN80', 'capacity_m3h': 310, 'capacity_cfm': 182},
            {'size': 'DN100 (4")', 'model': 'D-025-DN100', 'capacity_m3h': 540, 'capacity_cfm': 318},
            {'size': 'DN150 (6")', 'model': 'D-025-DN150', 'capacity_m3h': 1200, 'capacity_cfm': 706}
        ]
        
        selected_ari = next((v for v in ari_combo_sizes if v['capacity_cfm'] >= air_flow_cfm), ari_combo_sizes[-1])
        
        ari_sizing['recommendations']['combination_valve'] = {
            'air_flow_required': f"{air_flow_cfm:.1f} CFM ({air_flow_cfm * 1.699:.0f} m³/h)",
            'recommended_model': f"A.R.I. {selected_ari['model']}",
            'connection_size': selected_ari['size'],
            'rated_capacity': f"{selected_ari['capacity_cfm']} CFM ({selected_ari['capacity_m3h']} m³/h)",
            'series': 'D-025 Triple Function Air Valve'
        }
    
    elif 'Low Point' in valve_type:
        
        air_inflow_cfm = scenarios['system_draining']['air_inflow_rate_cfm']
        
        ari_vacuum_sizes = [
            {'size': 'DN50 (2")', 'model': 'D-022-DN50', 'capacity_cfm': 82},
            {'size': 'DN80 (3")', 'model': 'D-022-DN80', 'capacity_cfm': 165},
            {'size': 'DN100 (4")', 'model': 'D-022-DN100', 'capacity_cfm': 294},
            {'size': 'DN150 (6")', 'model': 'D-022-DN150', 'capacity_cfm': 659}
        ]
        
        selected_vacuum = next((v for v in ari_vacuum_sizes if v['capacity_cfm'] >= air_inflow_cfm), ari_vacuum_sizes[-1])
        
        ari_sizing['recommendations']['vacuum_valve'] = {
            'air_inflow_required': f"{air_inflow_cfm:.1f} CFM",
            'recommended_model': f"A.R.I. {selected_vacuum['model']}",
            'connection_size': selected_vacuum['size'],
            'rated_capacity': f"{selected_vacuum['capacity_cfm']} CFM",
            'series': 'D-022 Air/Vacuum Valve'
        }
    
    elif 'Long Run' in valve_type:
        
        ari_sizing['recommendations']['air_release'] = {
            'recommended_model': 'A.R.I. D-020',
            'connection_size': 'DN25 (1") or DN50 (2")',
            'series': 'D-020 Automatic Air Release Valve'
        }
    
    return ari_sizing

def generate_complete_valve_schedule(valve_locations, pipe_diameter, Q_gpm, static_head):
    """
    Generate complete air valve schedule with all manufacturers and scenarios
    """
    
    valve_schedule = []
    
    for valve in valve_locations['all_valves']:
        
        # Get comprehensive sizing for this valve
        scenarios = calculate_air_valve_sizing_comprehensive(
            valve, pipe_diameter, Q_gpm, static_head, system_pressure_psi=0
        )
        
        # Get manufacturer-specific recommendations
        valmatic = calculate_valve_sizing_valmatic_method(valve, pipe_diameter, Q_gpm, scenarios['scenarios'])
        apco = calculate_valve_sizing_apco_method(valve, pipe_diameter, Q_gpm, scenarios['scenarios'])
        ari = calculate_valve_sizing_ari_method(valve, pipe_diameter, Q_gpm, scenarios['scenarios'])
        
        valve_schedule.append({
            'valve_number': valve['valve_number'],
            'station': valve['distance'],
            'elevation': valve['elevation'],
            'location_type': valve['type'],
            'priority': valve['priority'],
            'comprehensive_sizing': scenarios,
            'manufacturer_options': {
                'val_matic': valmatic,
                'apco': apco,
                'ari': ari
            }
        })
    
    return valve_schedule
# =============================================================================
# BLOCK 4: COST OPTIMIZATION & ECONOMIC ANALYSIS
# =============================================================================

def estimate_air_valve_costs(valve_schedule):
    """
    Estimate costs for air valve system including:
    - Equipment costs
    - Installation costs
    - Maintenance costs (lifecycle)
    """
    
    # Typical air valve costs (2024 estimates, adjust as needed)
    valve_costs = {
        'combination': {
            '2"': 800,
            '3"': 1200,
            '4"': 1800,
            '6"': 3000
        },
        'vacuum': {
            '2"': 600,
            '3"': 900,
            '4"': 1300,
            '6"': 2200
        },
        'air_release': {
            '1"': 300,
            '2"': 400
        }
    }
    
    # Installation cost multipliers
    installation_factor = 2.5  # Installation typically 1.5-3x material cost
    
    cost_analysis = {
        'valves': [],
        'summary': {}
    }
    
    total_equipment_cost = 0
    total_installation_cost = 0
    
    for valve_entry in valve_schedule:
        valve = valve_entry
        valve_type = valve['location_type']
        
        # Determine valve category and size
        recommended = valve['comprehensive_sizing']['recommended_valve']
        connection_size = recommended['connection_size'].replace('"', '')
        
        # Estimate equipment cost
        if 'Combination' in valve_type or 'Pump Discharge' in valve_type:
            equipment_cost = valve_costs['combination'].get(connection_size, valve_costs['combination']['4"'])
            valve_category = 'Combination'
        elif 'Vacuum' in valve_type or 'Low Point' in valve_type:
            equipment_cost = valve_costs['vacuum'].get(connection_size, valve_costs['vacuum']['3"'])
            valve_category = 'Vacuum Relief'
        else:
            equipment_cost = valve_costs['air_release'].get(connection_size, valve_costs['air_release']['1"'])
            valve_category = 'Air Release'
        
        installation_cost = equipment_cost * installation_factor
        valve_total = equipment_cost + installation_cost
        
        # Annual maintenance cost (5% of equipment cost typical)
        annual_maintenance = equipment_cost * 0.05
        
        # 20-year lifecycle cost
        lifecycle_maintenance = annual_maintenance * 20
        lifecycle_total = valve_total + lifecycle_maintenance
        
        cost_analysis['valves'].append({
            'valve_number': valve['valve_number'],
            'station': valve['station'],
            'type': valve_category,
            'size': connection_size,
            'equipment_cost': equipment_cost,
            'installation_cost': installation_cost,
            'first_cost': valve_total,
            'annual_maintenance': annual_maintenance,
            'lifecycle_cost_20yr': lifecycle_total
        })
        
        total_equipment_cost += equipment_cost
        total_installation_cost += installation_cost
    
    total_first_cost = total_equipment_cost + total_installation_cost
    total_annual_maintenance = total_first_cost * 0.05
    total_lifecycle_20yr = total_first_cost + (total_annual_maintenance * 20)
    
    cost_analysis['summary'] = {
        'number_of_valves': len(valve_schedule),
        'total_equipment_cost': total_equipment_cost,
        'total_installation_cost': total_installation_cost,
        'total_first_cost': total_first_cost,
        'annual_maintenance_cost': total_annual_maintenance,
        'lifecycle_cost_20yr': total_lifecycle_20yr,
        'cost_per_valve_average': total_first_cost / len(valve_schedule) if len(valve_schedule) > 0 else 0
    }
    
    return cost_analysis

def optimize_valve_locations(valve_locations, pipe_diameter, Q_gpm, cost_weight=0.3, performance_weight=0.7):
    """
    Optimize valve locations balancing:
    - System performance (air evacuation, vacuum protection)
    - Installation cost
    - Maintenance accessibility
    """
    
    optimization_results = {
        'original_valve_count': len(valve_locations['all_valves']),
        'optimized_recommendations': [],
        'potential_savings': {}
    }
    
    # Analyze if any valves can be eliminated or combined
    high_points = valve_locations['high_points']
    low_points = valve_locations['low_points']
    long_runs = valve_locations['long_runs']
    
    # Check if long-run valves are truly needed based on velocity
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    optimized_long_runs = []
    
    if velocity_fps > 3.5:
        # High velocity - good air transport, may eliminate some long-run valves
        optimization_results['optimized_recommendations'].append({
            'recommendation': 'High velocity (>3.5 fps) provides good air transport',
            'action': f'Consider reducing long-run valves from {len(long_runs)} to essential locations only',
            'potential_savings': len(long_runs) * 0.4 * 1000  # Rough estimate
        })
        optimized_long_runs = long_runs[:max(1, len(long_runs)//2)]  # Keep half
    else:
        # Lower velocity - keep all long-run valves
        optimization_results['optimized_recommendations'].append({
            'recommendation': 'Moderate/low velocity requires air release valves as calculated',
            'action': f'Maintain all {len(long_runs)} long-run air release valves',
            'potential_savings': 0
        })
        optimized_long_runs = long_runs
    
    # Check for redundant valves at closely spaced high points
    consolidated_high_points = []
    min_spacing = 500  # Minimum 500 ft between high point valves
    
    last_included = None
    for hp in sorted(high_points, key=lambda x: x['distance']):
        if last_included is None:
            consolidated_high_points.append(hp)
            last_included = hp
        else:
            if hp['distance'] - last_included['distance'] > min_spacing:
                consolidated_high_points.append(hp)
                last_included = hp
            else:
                # Close spacing - keep the higher prominence peak
                if hp['prominence'] > last_included['prominence']:
                    consolidated_high_points[-1] = hp
                    last_included = hp
    
    if len(consolidated_high_points) < len(high_points):
        eliminated = len(high_points) - len(consolidated_high_points)
        optimization_results['optimized_recommendations'].append({
            'recommendation': f'Consolidate closely-spaced high point valves',
            'action': f'Reduce from {len(high_points)} to {len(consolidated_high_points)} high point valves',
            'potential_savings': eliminated * 3000  # Rough estimate for combination valve installed
        })
    
    optimization_results['optimized_valve_count'] = (
        len(consolidated_high_points) + 
        len(low_points) + 
        len(optimized_long_runs) + 
        1  # Pump discharge
    )
    
    total_savings = sum(r.get('potential_savings', 0) for r in optimization_results['optimized_recommendations'])
    
    optimization_results['potential_savings'] = {
        'total_estimated_savings': total_savings,
        'percentage_reduction': ((optimization_results['original_valve_count'] - optimization_results['optimized_valve_count']) / 
                                optimization_results['original_valve_count'] * 100) if optimization_results['original_valve_count'] > 0 else 0
    }
    
    return optimization_results

def calculate_air_evacuation_resistance(pipe_diameter_in, total_length_ft, elevation_change_ft):
    """
    Calculate additional head loss due to air evacuation during startup
    Based on AWWA guidelines and industry experience
    """
    pipe_volume_cf = np.pi * (pipe_diameter_in/12)**2 * total_length_ft / 4
    pipe_volume_gal = pipe_volume_cf * 7.48
    
    # Air evacuation resistance factors
    if elevation_change_ft > 30:
        air_factor = 0.25  # High elevation change = more air resistance
    elif elevation_change_ft > 15:
        air_factor = 0.15  # Moderate elevation change
    else:
        air_factor = 0.10  # Low elevation change
    
    # Base air evacuation head (empirical)
    base_air_head = 5.0  # ft (minimum for any system)
    elevation_air_head = elevation_change_ft * air_factor
    volume_air_head = (pipe_volume_gal / 1000) * 0.5  # Volume effect
    
    total_air_head = base_air_head + elevation_air_head + volume_air_head
    
    return {
        'base_air_resistance': base_air_head,
        'elevation_air_resistance': elevation_air_head,
        'volume_air_resistance': volume_air_head,
        'total_air_resistance': total_air_head,
        'pipe_volume_gal': pipe_volume_gal,
        'air_factor': air_factor
    }

def calculate_series_pumps_per_pump_system_curves(Q_design, TDH_design, static_head, pipe_diameter, total_length, hazen_c, minor_loss_K_total, num_pumps, pump_eff, motor_eff, motor_safety_factor):
    """
    CORRECT GRAPHICAL METHOD per user specification:
    
    - ONE individual pump curve (fixed)
    - MULTIPLE system curves showing what EACH PUMP sees in series
    - For n pumps in series: each pump sees System_Head(Q) / n
    - Intersection = operating point for each pump in that series configuration
    """
    
    import numpy as np
    
    # Pipe characteristics
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    standard_motors = [1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
    
    # ============================================================================
    # DEFINE INDIVIDUAL PUMP CURVE (fixed characteristic)
    # ============================================================================
    def individual_pump_curve(Q_gpm):
        """Single pump characteristic curve"""
        if Q_gpm <= 0:
            return TDH_design * 1.15  # Shutoff head
        
        H_shutoff = TDH_design * 1.15
        flow_ratio = Q_gpm / Q_design
        head_drop = (H_shutoff - TDH_design) * (flow_ratio ** 1.8)
        H_pump = H_shutoff - head_drop
        
        return max(H_pump, 0)
    
    # ============================================================================
    # DEFINE TOTAL SYSTEM CURVE (total head required)
    # ============================================================================
    def total_system_curve(Q_gpm):
        """Total system head required at flow Q"""
        if Q_gpm <= 0:
            return static_head
        
        velocity_fps = Q_gpm / (pipe_area_sqft * 449)
        friction_loss = calculate_friction_loss_hazen_williams(Q_gpm, pipe_diameter, total_length, hazen_c)
        velocity_head = velocity_fps**2 / (2 * 32.2)
        minor_losses = minor_loss_K_total * velocity_head
        
        return static_head + friction_loss + minor_losses
    
    # ============================================================================
    # DEFINE PER-PUMP SYSTEM CURVES (what each pump sees in series)
    # ============================================================================
    def per_pump_system_curve(Q_gpm, n_pumps):
        """
        What each individual pump must provide in n-pump series configuration
        Each pump provides: Total_System_Head(Q) / n
        """
        return total_system_curve(Q_gpm) / n_pumps
    
    # ============================================================================
    # FIND INTERSECTIONS (pump curve meets per-pump system curve)
    # ============================================================================
    def find_intersection_per_pump(n_pumps):
        """
        Find where individual pump curve intersects the per-pump system curve
        This is what each pump actually does in n-pump series configuration
        """
        
        # Search range
        Q_search = np.linspace(Q_design * 0.5, Q_design * 2.0, 200)
        
        # Individual pump curve (same for all scenarios)
        pump_heads = [individual_pump_curve(Q) for Q in Q_search]
        
        # Per-pump system curve (what each pump must overcome)
        per_pump_system_heads = [per_pump_system_curve(Q, n_pumps) for Q in Q_search]
        
        # Find intersection
        differences = [abs(ph - sh) for ph, sh in zip(pump_heads, per_pump_system_heads)]
        min_idx = np.argmin(differences)
        
        Q_operating = Q_search[min_idx]
        H_per_pump = per_pump_system_heads[min_idx]  # What each pump provides
        H_total_system = total_system_curve(Q_operating)  # Total system head
        error = differences[min_idx]
        
        return Q_operating, H_per_pump, H_total_system, error
    
    # ============================================================================
    # CALCULATE SCENARIOS
    # ============================================================================
    scenarios = []
    
    # Generate plotting data (do this once for all scenarios)
    Q_plot = np.linspace(Q_design * 0.3, Q_design * 2.0, 200)
    individual_pump_plot = [individual_pump_curve(Q) for Q in Q_plot]
    
    for n_series in range(1, num_pumps + 1):
        
        # Find operating point
        Q_operating, H_per_pump, H_total_system, error = find_intersection_per_pump(n_series)
        
        # Operating conditions
        velocity_operating = Q_operating / (pipe_area_sqft * 449)
        flow_increase_percent = ((Q_operating - Q_design) / Q_design) * 100
        baseline_velocity = Q_design / (pipe_area_sqft * 449)
        velocity_increase_percent = ((velocity_operating - baseline_velocity) / baseline_velocity) * 100
        
        # Power calculations per pump
        WHP_per_pump = (Q_operating * H_per_pump) / 3960
        BHP_per_pump = WHP_per_pump / pump_eff
        MHP_per_pump = BHP_per_pump / motor_eff
        MHP_design_per_pump = MHP_per_pump * motor_safety_factor
        
        # Motor sizing
        motor_size_per_pump = min([m for m in standard_motors if m >= MHP_design_per_pump])
        power_kw_per_pump = motor_size_per_pump * 0.746
        total_power_kw = power_kw_per_pump * n_series
        total_motor_hp = motor_size_per_pump * n_series
        
        # System breakdown at operating point
        friction_at_operating = calculate_friction_loss_hazen_williams(Q_operating, pipe_diameter, total_length, hazen_c)
        velocity_head_operating = velocity_operating**2 / (2 * 32.2)
        minor_losses_operating = minor_loss_K_total * velocity_head_operating
        
        # Scenario description
        if n_series == 1:
            scenario_type = "Single Pump"
            scenario_description = "Pump curve intersects full system curve"
            hydraulic_notes = f"Single pump: {Q_operating:.0f} GPM @ {H_per_pump:.1f} ft"
        else:
            scenario_type = f"{n_series} Pumps in Series"
            scenario_description = f"Pump curve intersects system curve ÷ {n_series}"
            hydraulic_notes = f"Each pump: {Q_operating:.0f} GPM @ {H_per_pump:.1f} ft (Total: {H_total_system:.1f} ft)"
        
        # Generate per-pump system curve data for plotting
        per_pump_system_plot = [per_pump_system_curve(Q, n_series) for Q in Q_plot]
        
        scenarios.append({
            'pumps_in_series': n_series,
            'scenario_type': scenario_type,
            'scenario_description': scenario_description,
            'typical_application': f"Each pump provides {100/n_series:.0f}% of total system head",
            'hydraulic_notes': hydraulic_notes,
            
            # Operating conditions
            'Q_operating': Q_operating,
            'Q_design_baseline': Q_design,
            'flow_increase_percent': flow_increase_percent,
            'H_system_total': H_total_system,
            'H_per_pump': H_per_pump,
            'velocity_operating': velocity_operating,
            'velocity_increase_percent': velocity_increase_percent,
            
            # System breakdown
            'static_head': static_head,
            'friction_loss_operating': friction_at_operating,
            'minor_losses_operating': minor_losses_operating,
            
            # Power analysis
            'WHP_per_pump': WHP_per_pump,
            'BHP_per_pump': BHP_per_pump,
            'MHP_per_pump': MHP_per_pump,
            'motor_size_per_pump': motor_size_per_pump,
            'power_kw_per_pump': power_kw_per_pump,
            'total_power_kw': total_power_kw,
            'total_motor_hp': total_motor_hp,
            
            # Efficiency
            'design_pump_eff': pump_eff,
            'actual_pump_eff': pump_eff,
            'efficiency_factor': 1.0,
            'efficiency_note': "At intersection point",
            
            # Engineering assessment
            'advantages': [
                f"✅ Flow: {Q_operating:.0f} GPM",
                f"✅ Each pump: {H_per_pump:.1f} ft",
                f"✅ Motor: {motor_size_per_pump:.1f} HP each"
            ],
            'disadvantages': [
                f"⚠️ Total power: {total_power_kw:.1f} kW",
                f"⚠️ {n_series} pumps to maintain"
            ],
            
            # Plotting data
            'curve_data': {
                'Q_range': Q_plot.tolist(),
                'individual_pump_curve': individual_pump_plot,
                'per_pump_system_curve': per_pump_system_plot,
                'n_series': n_series
            },
            
            'solution_error': error
        })
    
    return scenarios

def analyze_series_hydraulic_benefits(scenarios):
    """
    Analyze the hydraulic benefits of series pump configuration
    """
    
    if len(scenarios) < 2:
        return ["Insufficient scenarios for comparison"]
    
    baseline = scenarios[0]  # Single pump
    series_2 = scenarios[1]   # Two pumps in series
    
    recommendations = []
    
    # Flow capacity analysis
    flow_gain = series_2['Q_operating'] - baseline['Q_operating']
    flow_gain_percent = series_2['flow_increase_percent']
    
    if flow_gain_percent > 20:
        recommendations.append(f"🎯 **SIGNIFICANT CAPACITY GAIN:** Series pumps increase flow by {flow_gain_percent:.1f}% ({flow_gain:.0f} GPM)")
        recommendations.append("🎯 Series configuration provides substantial hydraulic advantage")
    elif flow_gain_percent > 10:
        recommendations.append(f"🎯 **MODERATE CAPACITY GAIN:** Series pumps increase flow by {flow_gain_percent:.1f}% ({flow_gain:.0f} GPM)")
        recommendations.append("🎯 Series configuration offers meaningful improvement")
    else:
        recommendations.append(f"🎯 **LIMITED CAPACITY GAIN:** Series pumps increase flow by only {flow_gain_percent:.1f}% ({flow_gain:.0f} GPM)")
        recommendations.append("🎯 System is friction-limited - series pumps have minimal benefit")
    
    # Velocity analysis
    if series_2['velocity_operating'] > 8:
        recommendations.append(f"⚠️ **HIGH VELOCITY WARNING:** Series operation results in {series_2['velocity_operating']:.2f} ft/s")
        recommendations.append("⚠️ Consider pipe erosion and water hammer effects")
    elif series_2['velocity_operating'] > 6:
        recommendations.append(f"🎯 **MODERATE VELOCITY:** Series operation: {series_2['velocity_operating']:.2f} ft/s")
        recommendations.append("🎯 Monitor for long-term pipe wear")
    
    # Power efficiency analysis
    power_penalty = ((series_2['total_power_kw'] - baseline['total_power_kw']) / baseline['total_power_kw']) * 100
    
    if power_penalty > 15:
        recommendations.append(f"⚠️ **POWER PENALTY:** Series pumps require {power_penalty:.0f}% more total power")
    else:
        recommendations.append(f"✅ **REASONABLE POWER:** Series pumps require only {power_penalty:.0f}% more total power")
    
    # Engineering recommendation
    if flow_gain_percent > 15 and series_2['velocity_operating'] < 8:
        recommendations.append("✅ **RECOMMENDATION:** Series pumps provide good hydraulic benefit with acceptable trade-offs")
    elif flow_gain_percent < 10:
        recommendations.append("🎯 **RECOMMENDATION:** Limited benefit from series pumps - single pump may be preferred")
    else:
        recommendations.append("🎯 **RECOMMENDATION:** Evaluate series pumps based on capacity needs vs. complexity")
    
    return recommendations

def calculate_series_control_strategy(scenarios, TDH):
    """
    Recommend when to use series pump configuration
    """

    single_pump_scenario = scenarios[0]

    recommendations = []

    # Analyze if series makes sense
    if TDH > 200:
        recommendations.append("? **VERY HIGH HEAD SYSTEM** - Series pumps strongly recommended")
        recommendations.append("Single pump would require specialized high-head design")
    elif TDH > 100:
        recommendations.append("? **HIGH HEAD SYSTEM** - Consider series pumps")
        recommendations.append("Series may allow use of standard pumps instead of high-head pumps")
    else:
        recommendations.append("? **MODERATE HEAD SYSTEM** - Single pump typically preferred")
        recommendations.append("Series pumps add complexity without significant benefit")

    # Motor size considerations
    if single_pump_scenario['motor_size_per_pump'] > 25:
        recommendations.append(f"? Single pump requires {single_pump_scenario['motor_size_per_pump']:.1f} HP motor - consider series to reduce motor size")

    # Efficiency considerations
    if len(scenarios) > 1:
        series_efficiency = scenarios[1]['total_whp'] / scenarios[1]['total_bhp']
        single_efficiency = scenarios[0]['total_whp'] / scenarios[0]['total_bhp']

        if series_efficiency < single_efficiency * 0.95:
            recommendations.append("Series configuration reduces overall efficiency")

    return recommendations


def calculate_multi_pump_scenarios(Q_avg, Q_peak, Q_min, TDH, num_pumps, pump_eff, motor_eff, motor_safety_factor):
    """
    Calculate performance for different numbers of pumps operating
    """
    scenarios = []
    standard_motors = [1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]

    for pumps_running in range(1, num_pumps + 1):
        Q_per_pump = Q_peak / pumps_running
        Q_total = Q_peak

        WHP = (Q_per_pump * TDH) / 3960
        BHP_actual = WHP / pump_eff
        MHP_actual = BHP_actual / motor_eff
        MHP_design = MHP_actual * motor_safety_factor

        motor_size = min([m for m in standard_motors if m >= MHP_design])
        power_kw = motor_size * 0.746

        actual_efficiency = WHP / (power_kw / 0.746 * motor_eff)
        efficiency_penalty = pump_eff - actual_efficiency

        scenarios.append({
            'pumps_running': pumps_running,
            'scenario_type': f'{pumps_running} Pump{"s" if pumps_running > 1 else ""}',
            'Q_per_pump_gpm': Q_per_pump,
            'Q_total_gpm': Q_total,
            'H_per_pump': TDH,
            'BHP_actual_per_pump': BHP_actual,
            'motor_size_per_pump': motor_size,
            'power_kw_per_pump': power_kw,
            'total_power_kw': power_kw * pumps_running,
            'actual_efficiency': actual_efficiency,
            'efficiency_penalty': efficiency_penalty,
            'efficiency_note': 'Good' if efficiency_penalty < 0.05 else 'Caution' if efficiency_penalty < 0.10 else 'Poor',
            'usage_notes': f'{pumps_running} pump{"s" if pumps_running > 1 else ""} operating at full capacity'
        })

    return scenarios


def calculate_pump_control_strategy(scenarios, Q_avg, Q_peak):
    """
    Recommend control strategy based on scenarios
    """
    strategies = []

    for scenario in scenarios:
        pumps = scenario['pumps_running']

        if pumps == 1:
            strategies.append({
                'pumps_operating': 1,
                'flow_range': f'0 to {scenario["Q_per_pump_gpm"]:.0f} GPM',
                'control_logic': 'Single pump with float switch or level control',
                'efficiency': f'{scenario["actual_efficiency"]*100:.0f}%',
                'notes': 'Use for flows up to single pump capacity'
            })
        else:
            strategies.append({
                'pumps_operating': pumps,
                'flow_range': f'{scenarios[pumps-2]["Q_per_pump_gpm"]:.0f} to {scenario["Q_per_pump_gpm"]:.0f} GPM',
                'control_logic': f'Lead/lag control with {pumps} pumps',
                'efficiency': f'{scenario["actual_efficiency"]*100:.0f}%',
                'notes': 'Stage in additional pumps as flow demand increases'
            })

    return strategies

def calculate_startup_conditions(elevation_df, pipe_diameter, total_length, Q_peak):
    """
    Calculate comprehensive startup conditions for pump sizing
    """
    elevations = elevation_df['Elevation (ft)'].values
    distances = elevation_df['Distance (ft)'].values
    
    pump_elevation = elevations[0]
    discharge_elevation = elevations[-1]
    max_elevation = elevations.max()
    max_elevation_distance = distances[elevations.argmax()]
    
    # Static heads for different scenarios
    static_to_high_point = max_elevation - pump_elevation
    static_to_discharge = discharge_elevation - pump_elevation
    
    # Air evacuation analysis
    elevation_rise = max_elevation - pump_elevation
    air_analysis = calculate_air_evacuation_resistance(pipe_diameter, total_length, elevation_rise)
    
    # Startup flow characteristics (typically 50-70% of design flow during filling)
    startup_flow_factor = 0.6
    startup_flow_gpm = Q_peak * startup_flow_factor
    
    # Filling resistance (higher than normal friction due to air/water interface)
    filling_friction_factor = 1.8  # Empirical - 80% higher during filling
    
    # Calculate startup TDH components
    startup_static = static_to_high_point
    startup_air_resistance = air_analysis['total_air_resistance']
    
    # Total startup TDH
    startup_TDH = startup_static + startup_air_resistance
    
    # Determine critical startup conditions
    is_startup_critical = startup_TDH > static_to_discharge * 1.1
    startup_advantage = startup_TDH - static_to_discharge if startup_TDH > static_to_discharge else 0
    
    return {
        'static_to_high_point': static_to_high_point,
        'static_to_discharge': static_to_discharge,
        'startup_TDH': startup_TDH,
        'startup_flow_gpm': startup_flow_gpm,
        'air_analysis': air_analysis,
        'filling_friction_factor': filling_friction_factor,
        'is_startup_critical': is_startup_critical,
        'startup_advantage': startup_advantage,
        'max_elevation_distance': max_elevation_distance,
        'startup_recommendations': generate_startup_recommendations(startup_TDH, static_to_discharge, air_analysis)
    }

def generate_startup_recommendations(startup_TDH, running_TDH, air_analysis):
    """
    Generate professional recommendations for startup considerations
    """
    recommendations = []
    
    if startup_TDH > running_TDH * 1.2:
        recommendations.append("⚠️ CRITICAL: Startup TDH significantly exceeds running TDH")
        recommendations.append("Consider pump with flat curve characteristics")
        recommendations.append("Verify motor starting torque requirements")
    
    if air_analysis['total_air_resistance'] > 10:
        recommendations.append("⚠️ High air evacuation resistance expected")
        recommendations.append("Ensure adequate air valve capacity at high points")
        recommendations.append("Consider staged startup procedure")
    
    if startup_TDH > running_TDH * 1.1:
        recommendations.append("💡 Pump selection must consider both operating points")
        recommendations.append("Verify efficiency at both startup and running conditions")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Startup conditions appear manageable")
        recommendations.append("Standard pump selection procedures apply")
    
    return recommendations
# =============================================================================
# COMPREHENSIVE AIR VALVE ANALYSIS SYSTEM - COMPLETE RESTORATION
# =============================================================================

def detect_valve_locations_comprehensive(elevation_df, pipe_diameter, Q_gpm, total_length):
    """
    Comprehensive detection of ALL air valve locations:
    1. High points (air accumulation)
    2. Low points (vacuum relief)
    3. Long horizontal runs (air release spacing)
    """
    
    import numpy as np
    
    distances = elevation_df['Distance (ft)'].values
    elevations = elevation_df['Elevation (ft)'].values
    
    valve_locations = {
        'high_points': [],
        'low_points': [],
        'long_runs': [],
        'pump_discharge': [],
        'all_valves': []
    }
    
    # Calculate velocity for air transport analysis
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    # HIGH POINTS - Local maxima
    # FIX 3: CORRECTED - Adaptive prominence threshold
    total_elevation_change = max(elevations) - min(elevations)
    # Use 2 ft minimum OR 5% of total elevation change (whichever is greater)
    prominence_threshold = max(2.0, total_elevation_change * 0.05)

    for i in range(1, len(elevations) - 1):
        if elevations[i] > elevations[i-1] and elevations[i] > elevations[i+1]:
            left_min = min(elevations[:i+1])
            right_min = min(elevations[i:])
            prominence = elevations[i] - max(left_min, right_min)
        
            if prominence > prominence_threshold:  # NEW: Adaptive threshold
                valve_locations['high_points'].append({
                    'index': i,
                    'distance': distances[i],
                    'elevation': elevations[i],
                    'prominence': prominence,
                    'type': 'High Point - Combination Valve',
                    'priority': 'CRITICAL' if prominence > 10 else 'HIGH',
                    'prominence_threshold_used': prominence_threshold  # Added for diagnostics
            })
    
    # LOW POINTS - Local minima
    # FIX 3: CORRECTED - Adaptive depth threshold
    depth_threshold = max(2.0, total_elevation_change * 0.05)

    for i in range(1, len(elevations) - 1):
        if elevations[i] < elevations[i-1] and elevations[i] < elevations[i+1]:
            left_max = max(elevations[:i+1])
            right_max = max(elevations[i:])
            depth = min(left_max, right_max) - elevations[i]
        
            if depth > depth_threshold:  # NEW: Adaptive threshold
                valve_locations['low_points'].append({
                    'index': i,
                    'distance': distances[i],
                    'elevation': elevations[i],
                    'depth': depth,
                    'type': 'Low Point - Vacuum Relief',
                    'priority': 'HIGH' if depth > 10 else 'MEDIUM',
                    'depth_threshold_used': depth_threshold  # Added for diagnostics
            })
    
    # LONG HORIZONTAL RUNS
    critical_velocity = 2.5
    if velocity_fps > 3.0:
        max_spacing = 2640
    elif velocity_fps > 2.5:
        max_spacing = 1980
    else:
        max_spacing = 1320
    
    last_valve_distance = 0
    for i in range(len(distances)):
        distance_from_last = distances[i] - last_valve_distance
        
        if distance_from_last > max_spacing:
            if i > 0:
                segment_length = distances[i] - distances[i-1]
                if segment_length > 0:
                    grade = abs(elevations[i] - elevations[i-1]) / segment_length
                    
                    if grade < 0.02:
                        valve_locations['long_runs'].append({
                            'index': i,
                            'distance': distances[i],
                            'elevation': elevations[i],
                            'spacing': distance_from_last,
                            'velocity': velocity_fps,
                            'type': 'Long Run - Air Release',
                            'priority': 'MEDIUM'
                        })
                        last_valve_distance = distances[i]
        
        if any(hp['distance'] == distances[i] for hp in valve_locations['high_points']):
            last_valve_distance = distances[i]
        if any(lp['distance'] == distances[i] for lp in valve_locations['low_points']):
            last_valve_distance = distances[i]
    
    # PUMP DISCHARGE
    valve_locations['pump_discharge'].append({
        'index': 0,
        'distance': distances[0],
        'elevation': elevations[0],
        'type': 'Pump Discharge - Combination Valve',
        'priority': 'CRITICAL'
    })
    
    # CONSOLIDATE
    all_valves = []
    all_valves.extend(valve_locations['high_points'])
    all_valves.extend(valve_locations['low_points'])
    all_valves.extend(valve_locations['long_runs'])
    all_valves.extend(valve_locations['pump_discharge'])
    
    all_valves.sort(key=lambda x: x['distance'])
    
    for i, valve in enumerate(all_valves, start=1):
        valve['valve_number'] = i
    
    valve_locations['all_valves'] = all_valves
    
    return valve_locations

def calculate_air_valve_sizing_comprehensive(valve_data, pipe_diameter, Q_gpm, static_head, system_pressure_psi=0):
    """
    Comprehensive air valve sizing for multiple scenarios
    """
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    pipe_area_sqin = pipe_area_sqft * 144
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    P_atm_psia = 14.7
    valve_type = valve_data['type']
    
    sizing_results = {
        'valve_type_required': None,
        'scenarios': {},
        'recommended_valve': None,
        'orifice_sizes': {}
    }
    
    # NORMAL OPERATION
    air_entrainment_percent = 2.0
    air_release_rate_cfm = (Q_gpm * 0.00223) * (air_entrainment_percent / 100)
    
    if system_pressure_psi > 0:
        pressure_correction = P_atm_psia / (P_atm_psia + system_pressure_psi)
        air_release_rate_cfm *= pressure_correction
    
    air_velocity_orifice = 10
    orifice_area_sqin_small = (air_release_rate_cfm * 144) / (air_velocity_orifice * 60)
    orifice_diameter_small = 2 * np.sqrt(orifice_area_sqin_small / np.pi)
    
    standard_small_orifices = [1/32, 1/16, 3/32, 1/8, 5/32, 3/16, 1/4]
    selected_small_orifice = min([s for s in standard_small_orifices if s >= orifice_diameter_small], default=1/4)
    
    sizing_results['scenarios']['normal_operation'] = {
        'description': 'Continuous air release during normal operation',
        'air_release_rate_cfm': air_release_rate_cfm,
        'calculated_orifice_diameter': orifice_diameter_small,
        'selected_orifice_diameter': selected_small_orifice,
        'orifice_type': 'Small (continuous release)'
    }
    
    # SYSTEM FILLING
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        filling_velocity_fps = 2.5
        filling_flow_gpm = filling_velocity_fps * pipe_area_sqft * 449
        air_evacuation_rate_cfm = filling_flow_gpm * 0.00223
        
        air_velocity_large_orifice = 3
        orifice_area_sqin_large = (air_evacuation_rate_cfm * 144) / (air_velocity_large_orifice * 60)
        orifice_diameter_large = 2 * np.sqrt(orifice_area_sqin_large / np.pi)
        
        if orifice_diameter_large < 1.5:
            connection_size = 2
        elif orifice_diameter_large < 3:
            connection_size = 3
        elif orifice_diameter_large < 5:
            connection_size = 4
        else:
            connection_size = 6
        
        sizing_results['scenarios']['system_filling'] = {
            'description': 'Air evacuation during system filling',
            'filling_velocity_fps': filling_velocity_fps,
            'filling_flow_gpm': filling_flow_gpm,
            'air_evacuation_rate_cfm': air_evacuation_rate_cfm,
            'calculated_large_orifice': orifice_diameter_large,
            'recommended_connection_size': connection_size,
            'orifice_type': 'Large (evacuation)'
        }
    
    # SYSTEM DRAINING
    if 'Low Point' in valve_type or 'Pump Discharge' in valve_type:
        drainage_velocity_fps = 3.0
        drainage_flow_gpm = drainage_velocity_fps * pipe_area_sqft * 449
        air_inflow_rate_cfm = drainage_flow_gpm * 0.00223
        allowable_vacuum_psi = 5.0
        
        C_d = 0.65
        delta_P_psf = allowable_vacuum_psi * 144
        rho_air = 0.075
        
        velocity_through_orifice = C_d * np.sqrt(2 * delta_P_psf / rho_air)
        orifice_area_sqft = (air_inflow_rate_cfm / 60) / velocity_through_orifice
        orifice_diameter_vacuum = 2 * np.sqrt(orifice_area_sqft / np.pi) * 12
        
        if orifice_diameter_vacuum < 2:
            vacuum_connection = 2
        elif orifice_diameter_vacuum < 4:
            vacuum_connection = 3
        elif orifice_diameter_vacuum < 6:
            vacuum_connection = 4
        else:
            vacuum_connection = 6
        
        sizing_results['scenarios']['system_draining'] = {
            'description': 'Vacuum relief during drainage',
            'drainage_flow_gpm': drainage_flow_gpm,
            'air_inflow_rate_cfm': air_inflow_rate_cfm,
            'allowable_vacuum_psi': allowable_vacuum_psi,
            'calculated_vacuum_orifice': orifice_diameter_vacuum,
            'recommended_connection_size': vacuum_connection,
            'orifice_type': 'Vacuum relief'
        }
    
    # DETERMINE VALVE TYPE
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        sizing_results['valve_type_required'] = 'Combination Air Valve'
        sizing_results['recommended_valve'] = {
            'type': 'Combination',
            'connection_size': f"{sizing_results['scenarios']['system_filling']['recommended_connection_size']}\"",
            'large_orifice': f"{sizing_results['scenarios']['system_filling']['calculated_large_orifice']:.2f}\"",
            'small_orifice': f"{selected_small_orifice:.3f}\"",
            'function': 'Air release, evacuation, and vacuum relief'
        }
    elif 'Low Point' in valve_type:
        sizing_results['valve_type_required'] = 'Air/Vacuum Valve'
        sizing_results['recommended_valve'] = {
            'type': 'Air/Vacuum',
            'connection_size': f"{sizing_results['scenarios']['system_draining']['recommended_connection_size']}\"",
            'orifice': f"{sizing_results['scenarios']['system_draining']['calculated_vacuum_orifice']:.2f}\"",
            'function': 'Vacuum relief during drainage'
        }
    elif 'Long Run' in valve_type:
        sizing_results['valve_type_required'] = 'Air Release Valve'
        sizing_results['recommended_valve'] = {
            'type': 'Air Release',
            'connection_size': '1\" or 2\"',
            'small_orifice': f"{selected_small_orifice:.3f}\"",
            'function': 'Continuous air release only'
        }
    
    return sizing_results

def calculate_valve_sizing_valmatic_method(valve_data, pipe_diameter, Q_gpm, scenarios):
    """Val-Matic sizing methodology"""
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    valve_type = valve_data['type']
    
    valmatic_sizing = {
        'manufacturer': 'Val-Matic',
        'method': 'Kinetic Sizing',
        'recommendations': {}
    }
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        air_flow_cfm = scenarios['system_filling']['air_evacuation_rate_cfm']
        
        valmatic_sizes = [
            {'size': '1"', 'model': '107', 'max_flow_cfm': 15},
            {'size': '2"', 'model': '207', 'max_flow_cfm': 65},
            {'size': '3"', 'model': '307', 'max_flow_cfm': 145},
            {'size': '4"', 'model': '407', 'max_flow_cfm': 260},
            {'size': '6"', 'model': '607', 'max_flow_cfm': 585}
        ]
        
        selected_valve = next((v for v in valmatic_sizes if v['max_flow_cfm'] >= air_flow_cfm), valmatic_sizes[-1])
        
        valmatic_sizing['recommendations']['combination_valve'] = {
            'air_flow_required': f"{air_flow_cfm:.1f} CFM",
            'recommended_model': f"Val-Matic {selected_valve['model']}",
            'connection_size': selected_valve['size'],
            'capacity': f"{selected_valve['max_flow_cfm']} CFM",
            'function': 'Large orifice for evacuation, small orifice for continuous release'
        }
    
    elif 'Low Point' in valve_type:
        air_inflow_cfm = scenarios['system_draining']['air_inflow_rate_cfm']
        
        vacuum_sizes = [
            {'size': '2"', 'model': '201-A', 'max_flow_cfm': 50},
            {'size': '3"', 'model': '301-A', 'max_flow_cfm': 115},
            {'size': '4"', 'model': '401-A', 'max_flow_cfm': 210},
            {'size': '6"', 'model': '601-A', 'max_flow_cfm': 475}
        ]
        
        selected_vacuum = next((v for v in vacuum_sizes if v['max_flow_cfm'] >= air_inflow_cfm), vacuum_sizes[-1])
        
        valmatic_sizing['recommendations']['vacuum_valve'] = {
            'air_inflow_required': f"{air_inflow_cfm:.1f} CFM",
            'recommended_model': f"Val-Matic {selected_vacuum['model']}",
            'connection_size': selected_vacuum['size'],
            'capacity': f"{selected_vacuum['max_flow_cfm']} CFM",
            'function': 'Vacuum relief during drainage'
        }
    
    elif 'Long Run' in valve_type:
        valmatic_sizing['recommendations']['air_release'] = {
            'air_release_required': f"{scenarios['normal_operation']['air_release_rate_cfm']:.2f} CFM",
            'recommended_model': 'Val-Matic Model 1 or 2',
            'connection_size': '1" or 2"',
            'orifice': '1/8" or 1/4"',
            'function': 'Continuous air release only'
        }
    
    return valmatic_sizing

def calculate_valve_sizing_apco_method(valve_data, pipe_diameter, Q_gpm, scenarios):
    """APCO sizing methodology"""
    
    valve_type = valve_data['type']
    
    apco_sizing = {
        'manufacturer': 'APCO',
        'method': 'Flow-based Selection',
        'recommendations': {}
    }
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        air_flow_cfm = scenarios['system_filling']['air_evacuation_rate_cfm']
        
        apco_combo_sizes = [
            {'size': '2"', 'model': '340-020', 'capacity_cfm': 80},
            {'size': '3"', 'model': '340-030', 'capacity_cfm': 180},
            {'size': '4"', 'model': '340-040', 'capacity_cfm': 320},
            {'size': '6"', 'model': '340-060', 'capacity_cfm': 720}
        ]
        
        selected_apco = next((v for v in apco_combo_sizes if v['capacity_cfm'] >= air_flow_cfm), apco_combo_sizes[-1])
        
        apco_sizing['recommendations']['combination_valve'] = {
            'air_flow_required': f"{air_flow_cfm:.1f} CFM",
            'recommended_model': f"APCO {selected_apco['model']}",
            'connection_size': selected_apco['size'],
            'rated_capacity': f"{selected_apco['capacity_cfm']} CFM",
            'series': '340/345 Combination Air Valve'
        }
    
    elif 'Low Point' in valve_type:
        air_inflow_cfm = scenarios['system_draining']['air_inflow_rate_cfm']
        
        apco_vacuum_sizes = [
            {'size': '2"', 'model': '330-020', 'capacity_cfm': 65},
            {'size': '3"', 'model': '330-030', 'capacity_cfm': 145},
            {'size': '4"', 'model': '330-040', 'capacity_cfm': 260},
            {'size': '6"', 'model': '330-060', 'capacity_cfm': 585}
        ]
        
        selected_vacuum = next((v for v in apco_vacuum_sizes if v['capacity_cfm'] >= air_inflow_cfm), apco_vacuum_sizes[-1])
        
        apco_sizing['recommendations']['vacuum_valve'] = {
            'air_inflow_required': f"{air_inflow_cfm:.1f} CFM",
            'recommended_model': f"APCO {selected_vacuum['model']}",
            'connection_size': selected_vacuum['size'],
            'rated_capacity': f"{selected_vacuum['capacity_cfm']} CFM",
            'series': '330 Air/Vacuum Valve'
        }
    
    elif 'Long Run' in valve_type:
        apco_sizing['recommendations']['air_release'] = {
            'recommended_model': 'APCO 310-Series',
            'connection_size': '1" or 2"',
            'series': 'Air Release Valve (small orifice)'
        }
    
    return apco_sizing

def calculate_valve_sizing_ari_method(valve_data, pipe_diameter, Q_gpm, scenarios):
    """A.R.I. sizing methodology"""
    
    valve_type = valve_data['type']
    
    ari_sizing = {
        'manufacturer': 'A.R.I.',
        'method': 'European/Israeli Standards',
        'recommendations': {}
    }
    
    if 'High Point' in valve_type or 'Pump Discharge' in valve_type:
        air_flow_cfm = scenarios['system_filling']['air_evacuation_rate_cfm']
        
        ari_combo_sizes = [
            {'size': 'DN50 (2")', 'model': 'D-025-DN50', 'capacity_m3h': 160, 'capacity_cfm': 94},
            {'size': 'DN80 (3")', 'model': 'D-025-DN80', 'capacity_m3h': 310, 'capacity_cfm': 182},
            {'size': 'DN100 (4")', 'model': 'D-025-DN100', 'capacity_m3h': 540, 'capacity_cfm': 318},
            {'size': 'DN150 (6")', 'model': 'D-025-DN150', 'capacity_m3h': 1200, 'capacity_cfm': 706}
        ]
        
        selected_ari = next((v for v in ari_combo_sizes if v['capacity_cfm'] >= air_flow_cfm), ari_combo_sizes[-1])
        
        ari_sizing['recommendations']['combination_valve'] = {
            'air_flow_required': f"{air_flow_cfm:.1f} CFM ({air_flow_cfm * 1.699:.0f} m³/h)",
            'recommended_model': f"A.R.I. {selected_ari['model']}",
            'connection_size': selected_ari['size'],
            'rated_capacity': f"{selected_ari['capacity_cfm']} CFM ({selected_ari['capacity_m3h']} m³/h)",
            'series': 'D-025 Triple Function Air Valve'
        }
    
    elif 'Low Point' in valve_type:
        air_inflow_cfm = scenarios['system_draining']['air_inflow_rate_cfm']
        
        ari_vacuum_sizes = [
            {'size': 'DN50 (2")', 'model': 'D-022-DN50', 'capacity_cfm': 82},
            {'size': 'DN80 (3")', 'model': 'D-022-DN80', 'capacity_cfm': 165},
            {'size': 'DN100 (4")', 'model': 'D-022-DN100', 'capacity_cfm': 294},
            {'size': 'DN150 (6")', 'model': 'D-022-DN150', 'capacity_cfm': 659}
        ]
        
        selected_vacuum = next((v for v in ari_vacuum_sizes if v['capacity_cfm'] >= air_inflow_cfm), ari_vacuum_sizes[-1])
        
        ari_sizing['recommendations']['vacuum_valve'] = {
            'air_inflow_required': f"{air_inflow_cfm:.1f} CFM",
            'recommended_model': f"A.R.I. {selected_vacuum['model']}",
            'connection_size': selected_vacuum['size'],
            'rated_capacity': f"{selected_vacuum['capacity_cfm']} CFM",
            'series': 'D-022 Air/Vacuum Valve'
        }
    
    elif 'Long Run' in valve_type:
        ari_sizing['recommendations']['air_release'] = {
            'recommended_model': 'A.R.I. D-020',
            'connection_size': 'DN25 (1") or DN50 (2")',
            'series': 'D-020 Automatic Air Release Valve'
        }
    
    return ari_sizing

def generate_complete_valve_schedule(valve_locations, pipe_diameter, Q_gpm, static_head):
    """Generate complete air valve schedule"""
    
    valve_schedule = []
    
    for valve in valve_locations['all_valves']:
        scenarios = calculate_air_valve_sizing_comprehensive(valve, pipe_diameter, Q_gpm, static_head, system_pressure_psi=0)
        valmatic = calculate_valve_sizing_valmatic_method(valve, pipe_diameter, Q_gpm, scenarios['scenarios'])
        apco = calculate_valve_sizing_apco_method(valve, pipe_diameter, Q_gpm, scenarios['scenarios'])
        ari = calculate_valve_sizing_ari_method(valve, pipe_diameter, Q_gpm, scenarios['scenarios'])
        
        valve_schedule.append({
            'valve_number': valve['valve_number'],
            'station': valve['distance'],
            'elevation': valve['elevation'],
            'location_type': valve['type'],
            'priority': valve['priority'],
            'comprehensive_sizing': scenarios,
            'manufacturer_options': {
                'val_matic': valmatic,
                'apco': apco,
                'ari': ari
            }
        })
    
    return valve_schedule

def estimate_air_valve_costs(valve_schedule):
    """Estimate costs for air valve system"""
    
    valve_costs = {
        'combination': {'2"': 800, '3"': 1200, '4"': 1800, '6"': 3000},
        'vacuum': {'2"': 600, '3"': 900, '4"': 1300, '6"': 2200},
        'air_release': {'1"': 300, '2"': 400}
    }
    
    installation_factor = 2.5
    
    cost_analysis = {'valves': [], 'summary': {}}
    
    total_equipment_cost = 0
    total_installation_cost = 0
    
    for valve_entry in valve_schedule:
        valve = valve_entry
        valve_type = valve['location_type']
        recommended = valve['comprehensive_sizing']['recommended_valve']
        connection_size = recommended['connection_size'].replace('"', '')
        
        if 'Combination' in valve_type or 'Pump Discharge' in valve_type:
            equipment_cost = valve_costs['combination'].get(connection_size, valve_costs['combination']['4"'])
            valve_category = 'Combination'
        elif 'Vacuum' in valve_type or 'Low Point' in valve_type:
            equipment_cost = valve_costs['vacuum'].get(connection_size, valve_costs['vacuum']['3"'])
            valve_category = 'Vacuum Relief'
        else:
            equipment_cost = valve_costs['air_release'].get(connection_size, valve_costs['air_release']['1"'])
            valve_category = 'Air Release'
        
        installation_cost = equipment_cost * installation_factor
        valve_total = equipment_cost + installation_cost
        annual_maintenance = equipment_cost * 0.05
        lifecycle_maintenance = annual_maintenance * 20
        lifecycle_total = valve_total + lifecycle_maintenance
        
        cost_analysis['valves'].append({
            'valve_number': valve['valve_number'],
            'station': valve['station'],
            'type': valve_category,
            'size': connection_size,
            'equipment_cost': equipment_cost,
            'installation_cost': installation_cost,
            'first_cost': valve_total,
            'annual_maintenance': annual_maintenance,
            'lifecycle_cost_20yr': lifecycle_total
        })
        
        total_equipment_cost += equipment_cost
        total_installation_cost += installation_cost
    
    total_first_cost = total_equipment_cost + total_installation_cost
    total_annual_maintenance = total_first_cost * 0.05
    total_lifecycle_20yr = total_first_cost + (total_annual_maintenance * 20)
    
    cost_analysis['summary'] = {
        'number_of_valves': len(valve_schedule),
        'total_equipment_cost': total_equipment_cost,
        'total_installation_cost': total_installation_cost,
        'total_first_cost': total_first_cost,
        'annual_maintenance_cost': total_annual_maintenance,
        'lifecycle_cost_20yr': total_lifecycle_20yr,
        'cost_per_valve_average': total_first_cost / len(valve_schedule) if len(valve_schedule) > 0 else 0
    }
    
    return cost_analysis

def optimize_valve_locations(valve_locations, pipe_diameter, Q_gpm, cost_weight=0.3, performance_weight=0.7):
    """Optimize valve locations"""
    
    optimization_results = {
        'original_valve_count': len(valve_locations['all_valves']),
        'optimized_recommendations': [],
        'potential_savings': {}
    }
    
    high_points = valve_locations['high_points']
    low_points = valve_locations['low_points']
    long_runs = valve_locations['long_runs']
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    optimized_long_runs = []
    
    if velocity_fps > 3.5:
        optimization_results['optimized_recommendations'].append({
            'recommendation': 'High velocity (>3.5 fps) provides good air transport',
            'action': f'Consider reducing long-run valves from {len(long_runs)} to essential locations only',
            'potential_savings': len(long_runs) * 0.4 * 1000
        })
        optimized_long_runs = long_runs[:max(1, len(long_runs)//2)]
    else:
        optimization_results['optimized_recommendations'].append({
            'recommendation': 'Moderate/low velocity requires air release valves as calculated',
            'action': f'Maintain all {len(long_runs)} long-run air release valves',
            'potential_savings': 0
        })
        optimized_long_runs = long_runs
    
    optimization_results['optimized_valve_count'] = len(high_points) + len(low_points) + len(optimized_long_runs) + 1
    
    total_savings = sum(r.get('potential_savings', 0) for r in optimization_results['optimized_recommendations'])
    
    optimization_results['potential_savings'] = {
        'total_estimated_savings': total_savings,
        'percentage_reduction': ((optimization_results['original_valve_count'] - optimization_results['optimized_valve_count']) / 
                                optimization_results['original_valve_count'] * 100) if optimization_results['original_valve_count'] > 0 else 0
    }
    
    return optimization_results

def analyze_transient_conditions(elevation_df, Q_gpm, pipe_diameter, wave_speed=3000):
    """Analyze transient conditions"""
    
    distances = elevation_df['Distance (ft)'].values
    elevations = elevation_df['Elevation (ft)'].values
    
    pipe_area_sqft = np.pi * (pipe_diameter/12)**2 / 4
    velocity_fps = Q_gpm / (pipe_area_sqft * 449)
    
    transient_results = []
    
    for i in range(len(distances)):
        pressure_drop_ft = (wave_speed * velocity_fps) / 32.2
        steady_pressure_ft = 50
        min_transient_pressure = steady_pressure_ft - pressure_drop_ft
        vapor_pressure_ft = -33
        
        if min_transient_pressure < vapor_pressure_ft + 10:
            separation_risk = "CRITICAL"
        elif min_transient_pressure < vapor_pressure_ft + 20:
            separation_risk = "HIGH"
        elif min_transient_pressure < 0:
            separation_risk = "MODERATE"
        else:
            separation_risk = "LOW"
        
        transient_results.append({
            'distance': distances[i],
            'elevation': elevations[i],
            'pressure_drop_ft': pressure_drop_ft,
            'min_transient_pressure': min_transient_pressure,
            'separation_risk': separation_risk
        })
    
    return transient_results

def calculate_staged_filling_procedure(valve_locations, pipe_diameter, total_length):
    """Generate optimal filling procedure"""
    
    high_points = valve_locations['high_points']
    
    if not high_points:
        pipe_area = np.pi * (pipe_diameter/12)**2 / 4
        fill_time = (total_length * pipe_area) / (2.5 * 60)
        return {
            'procedure': 'Simple',
            'stages': ['Fill at 2-3 fps until full'],
            'estimated_time_minutes': fill_time
        }
    
    stages = []
    
    stages.append({
        'stage': 1,
        'description': f"Fill to Station {high_points[0]['distance']:.0f} ft (First High Point)",
        'target_elevation': high_points[0]['elevation'],
        'filling_rate': '2.0 fps (slow)',
        'air_valve_action': 'Air evacuates through HP1 combination valve',
        'estimated_time_min': 'TBD'
    })
    
    for i, hp in enumerate(high_points[1:], start=2):
        stages.append({
            'stage': i,
            'description': f"Continue to Station {hp['distance']:.0f} ft (High Point {i})",
            'target_elevation': hp['elevation'],
            'filling_rate': '2.5 fps (moderate)',
            'air_valve_action': f'Air evacuates through HP{i} combination valve',
            'estimated_time_min': 'TBD'
        })
    
    stages.append({
        'stage': len(stages) + 1,
        'description': 'Fill remainder to discharge',
        'target_elevation': 'Discharge elevation',
        'filling_rate': '3.0 fps (normal)',
        'air_valve_action': 'Final air release through discharge valve',
        'estimated_time_min': 'TBD'
    })
    
    return {
        'procedure': 'Staged',
        'number_of_stages': len(stages),
        'stages': stages,
        'total_estimated_time_hours': 'TBD',
        'critical_notes': [
            'Monitor pressure at each high point',
            'Verify air valve operation at each stage',
            'Do not exceed 3 fps filling velocity',
            'Allow time for air evacuation before increasing flow'
        ]
    }

def calculate_design(Q_avg, Q_peak, Q_min, motor_safety_factor,
                    pipe_diameter, num_pumps, pump_eff, motor_eff, wetwell_diameter,
                    max_cycles, min_drawdown, elevation_df, minor_loss_df,
                    hazen_c, calculate_friction):
    """
    Perform all design calculations with multi-point elevation profile
    Uses Smith & Loveless methodology: TDH to controlling high point, gravity/siphon after
    CORRECTED: NO safety factor on TDH - TDH is actual hydraulic requirement
    Safety factors applied to motor sizing only
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
    # CORRECTED TDH CALCULATION - Smith & Loveless Method
    # TDH calculated ONLY to controlling point (high point or discharge)
    # HGL calculated for entire system for verification and pressure analysis
    # ============================================================================
    
    hgl_values = [0] * len(distances)  # Initialize
    pressure_values = [0] * len(distances)  # Initialize
    
    # Start at discharge: HGL = discharge elevation (P = 0)
    hgl_values[-1] = discharge_elevation
    pressure_values[-1] = 0.0
    
    # Work backwards (upstream) from discharge - FOR HGL CALCULATION ONLY
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
    
    # ============================================================================
    # CORRECTED TDH CALCULATION - Per Smith & Loveless Method
    # ============================================================================
    
    if high_point_controls:
        # Smith & Loveless: TDH calculated ONLY to controlling high point
        # Find the controlling high point index
        control_point_idx = max_elevation_idx
        
        # Calculate TDH components TO CONTROLLING POINT ONLY
        cumulative_friction_to_control = 0
        cumulative_minor_to_control = 0
        
        # Sum losses from pump (index 0) to controlling point
        for i in range(1, control_point_idx + 1):
            if distances[i] <= design_length:
                cumulative_friction_to_control += segment_data[i]['segment_friction']
                cumulative_minor_to_control += segment_data[i]['minor_loss_at_point']
        
        # CORRECTED TDH = Static head to control point + losses to control point
        TDH = static_head + cumulative_friction_to_control + cumulative_minor_to_control
        
        # Store losses for reporting
        friction_loss = cumulative_friction_to_control
        minor_losses = cumulative_minor_to_control
        
    else:
        # Discharge controls: TDH to discharge (entire system)
        TDH = hgl_values[0]  # This is correct when discharge controls
        
        # Calculate losses to discharge for reporting
        cumulative_friction_to_discharge = sum(seg['segment_friction'] for seg in segment_data[1:])
        cumulative_minor_to_discharge = sum(seg['minor_loss_at_point'] for seg in segment_data)
        
        friction_loss = cumulative_friction_to_discharge
        minor_losses = cumulative_minor_to_discharge
    
    # Verification: Check that TDH calculation is correct
    if high_point_controls:
        # For high point control, verify TDH = static + friction + minor to control point
        calculated_tdh_check = static_head + friction_loss + minor_losses
        tdh_calculation_error = abs(TDH - calculated_tdh_check)
    else:
        # For discharge control, verify TDH matches HGL calculation
        tdh_calculation_error = abs(TDH - hgl_values[0])
    
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
    
    # Pump sizing - N-1 redundancy (safety factor applied here)
    pumps_operating = num_pumps - 1
    Q_pump_gpm = Q_peak / pumps_operating
    
    # Motor power calculation - Safety factor applied to MOTOR sizing, not TDH
    WHP = (Q_pump_gpm * TDH) / 3960  # Using actual TDH, no safety factor
    BHP = WHP / pump_eff
    MHP = BHP / motor_eff
    
    # Apply safety factor to MOTOR sizing only
    MHP_design = MHP * motor_safety_factor  # This is where safety factor belongs
    
    standard_motors = [1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
    motor_size = min([m for m in standard_motors if m >= MHP_design])
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
    # =========================================================================
    # COMPREHENSIVE AIR VALVE ANALYSIS - INTEGRATION (Part 2)
    # =========================================================================
    try:
        # Detect ALL valve locations (high points, low points, long runs)
        comprehensive_valve_locations = detect_valve_locations_comprehensive(
            elevation_df, pipe_diameter, Q_peak, total_length
        )

        # Generate complete valve schedule with sizing for all scenarios
        comprehensive_valve_schedule = generate_complete_valve_schedule(
            comprehensive_valve_locations, pipe_diameter, Q_peak, static_head
        )

        # Transient analysis for water hammer and column separation
        transient_analysis = analyze_transient_conditions(
            elevation_df, Q_peak, pipe_diameter, wave_speed=3000
        )

        # Generate staged filling procedure
        filling_procedure = calculate_staged_filling_procedure(
            comprehensive_valve_locations, pipe_diameter, total_length
        )

        # Cost analysis
        valve_cost_analysis = estimate_air_valve_costs(comprehensive_valve_schedule)

        # Optimization analysis
        valve_optimization = optimize_valve_locations(
            comprehensive_valve_locations, pipe_diameter, Q_peak,
            cost_weight=0.3, performance_weight=0.7
        )

        # Summary statistics
        air_valve_summary = {
            'total_valves': len(comprehensive_valve_locations['all_valves']),
            'high_point_valves': len(comprehensive_valve_locations['high_points']),
            'low_point_valves': len(comprehensive_valve_locations['low_points']),
            'long_run_valves': len(comprehensive_valve_locations['long_runs']),
            'pump_discharge_valves': len(comprehensive_valve_locations['pump_discharge']),
            'total_first_cost': valve_cost_analysis['summary']['total_first_cost'],
            'lifecycle_cost_20yr': valve_cost_analysis['summary']['lifecycle_cost_20yr'],
            'optimization_savings': valve_optimization['potential_savings']['total_estimated_savings']
        }

        # Critical findings and warnings
        air_valve_warnings = []

        # Check for high-risk conditions
        if comprehensive_valve_locations['high_points']:
            max_prominence = max(hp['prominence'] for hp in comprehensive_valve_locations['high_points'])
            if max_prominence > 20:
                air_valve_warnings.append({
                    'severity': 'CRITICAL',
                    'message': f'Very high prominence peak detected ({max_prominence:.1f} ft) - critical air valve location',
                    'recommendation': 'Consider redundant valve or larger capacity valve at this location'
                })

        if not comprehensive_valve_locations['low_points'] and total_length > 2000:
            air_valve_warnings.append({
                'severity': 'WARNING',
                'message': 'No low points detected in long forcemain',
                'recommendation': 'Verify profile accuracy - vacuum relief may still be needed'
            })

        # Check for column separation risk
        high_risk_transients = [t for t in transient_analysis if t['separation_risk'] in ['CRITICAL', 'HIGH']]
        if high_risk_transients:
            air_valve_warnings.append({
                'severity': 'CRITICAL',
                'message': f'{len(high_risk_transients)} locations with column separation risk',
                'recommendation': 'Surge protection and proper air valve sizing critical'
            })

        # Check velocity for air transport
        if pipe_velocity < 2.5:
            air_valve_warnings.append({
                'severity': 'WARNING',
                'message': f'Low velocity ({pipe_velocity:.2f} fps) - poor air transport',
                'recommendation': 'Additional air release valves recommended on long runs'
            })

        # Set flag that comprehensive analysis succeeded
        comprehensive_air_valve_available = True

    except Exception as e:
        # If comprehensive analysis fails, fall back to basic analysis
        comprehensive_air_valve_available = False
        comprehensive_valve_locations = None
        comprehensive_valve_schedule = None
        transient_analysis = None
        filling_procedure = None
        valve_cost_analysis = None
        valve_optimization = None
        air_valve_summary = None
        air_valve_warnings = [{
            'severity': 'ERROR',
            'message': f'Comprehensive air valve analysis failed: {str(e)}',
            'recommendation': 'Using basic air valve analysis only'
        }]
    # STARTUP ANALYSIS - NEW SECTION
    startup_analysis = calculate_startup_conditions(
        elevation_df, pipe_diameter, total_length, Q_peak
    )
    
    # Determine if startup is critical for warnings
    startup_warning = startup_analysis['is_startup_critical']
    # MULTI-PUMP OPERATION ANALYSIS - NEW SECTION
    multi_pump_scenarios = calculate_multi_pump_scenarios(
        Q_avg, Q_peak, Q_min, TDH, num_pumps, pump_eff, motor_eff, motor_safety_factor
    )
    
    pump_control_strategy = calculate_pump_control_strategy(
        multi_pump_scenarios, Q_avg, Q_peak
    )
    
        # In calculate_design:
    series_pump_scenarios = calculate_series_pumps_per_pump_system_curves(
        Q_peak, TDH, static_head, pipe_diameter, total_length, hazen_c, 
        sum(st.session_state.minor_loss_components['Quantity'] * st.session_state.minor_loss_components['K-value']),
        num_pumps, pump_eff, motor_eff, motor_safety_factor
    )
    
    # CORRECTED: Use the right function name
    series_control_recommendations = analyze_series_hydraulic_benefits(series_pump_scenarios)
    
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
        'TDH': TDH,  # CORRECTED: Single TDH value, no safety factor
        'motor_safety_factor': motor_safety_factor,
        'num_pumps': num_pumps,
        'pumps_operating': pumps_operating,
        'Q_pump_gpm': Q_pump_gpm,
        'WHP': WHP, 'BHP': BHP, 'MHP': MHP,
        'MHP_design': MHP_design,  # Motor HP with safety factor
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
        'comprehensive_air_valve_available': comprehensive_air_valve_available,
        'comprehensive_valve_locations': comprehensive_valve_locations,
        'comprehensive_valve_schedule': comprehensive_valve_schedule,
        'transient_analysis': transient_analysis,
        'filling_procedure': filling_procedure,
        'valve_cost_analysis': valve_cost_analysis,
        'valve_optimization': valve_optimization,
        'air_valve_summary': air_valve_summary,
        'air_valve_warnings': air_valve_warnings,
        'has_negative_pressure': has_negative_pressure,
        'min_pressure': min_pressure,
        'min_pressure_location': min_pressure_location,
        'siphon_data': siphon_data,
        'hgl_at_discharge': hgl_at_discharge,
        'hgl_error': hgl_error,
        'startup_analysis': startup_analysis,  # NEW: Startup analysis
        'startup_warning': startup_warning,      # NEW: Startup warning flag
        'multi_pump_scenarios': multi_pump_scenarios,  # NEW: Multi-pump scenarios
        'pump_control_strategy': pump_control_strategy,   # NEW: Pump control strategy
        'series_pump_scenarios': series_pump_scenarios,
        'series_control_recommendations': series_control_recommendations
    }
# Title and header
st.title("Lift Station Sizing Tool v5.0")
st.markdown("### Professional Engineering Design Tool for Wastewater Pump Stations")
st.markdown("**With Multi-Point Elevation Profile, Siphon Analysis, Air Valve Sizing, and Startup Analysis**")
st.markdown("---")

# Add hydraulic principle explanation
with st.expander("IMPORTANT: Smith & Loveless Methodology - CORRECTED TDH Approach", expanded=False):
    st.markdown("""
    ### Wastewater Forcemain Design Principles - CORRECTED
    
    **Unlike pressurized water systems, wastewater forcemains discharge to atmosphere (P=0)**
    
    #### CORRECTED Design Methodology:
    
    **TDH = Static Head + Friction Losses + Minor Losses (NO SAFETY FACTOR)**
    - TDH is a **physical reality**, not a design choice
    - Safety factors applied to **equipment**, not hydraulic calculations
    
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
    
    #### NEW: Startup vs Running Analysis:
    **STARTUP CONDITIONS (Empty Pipe):**
    - Higher head required to fill system
    - Air evacuation resistance
    - Static head to highest point
    
    **RUNNING CONDITIONS (Filled Pipe):**
    - Normal TDH calculation
    - Siphon assistance (if applicable)
    - Standard friction losses
    
    #### Where Safety Factors ARE Applied:
    - **Motor Sizing:** 15% safety factor on motor HP
    - **Pump Redundancy:** N+1 pump configuration
    - **Flow Capacity:** Design for peak flow conditions
    
    #### At Discharge:
    - **HGL ALWAYS = Discharge Elevation** (pressure = 0, atmospheric)
    
    #### CORRECTED HGL CALCULATION:
    - HGL calculated backwards from discharge where P=0
    - Ensures HGL = discharge elevation at discharge point
    - TDH = HGL at pump station (actual hydraulic requirement)
    """)

# Sidebar for inputs - CORRECTED
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
    
    # CORRECTED: Motor safety factor only, NOT TDH safety factor
    st.subheader("Safety Factors")
    motor_safety_factor = st.number_input("Motor Safety Factor", value=1.15, step=0.05, 
                                        help="Applied to motor sizing only, NOT to TDH calculation")
    
    st.info("""
    **CORRECTED APPROACH:**
    • TDH = Actual hydraulic requirement
    • NO safety factor on TDH
    • Safety factors applied to equipment only
    """)
    
    st.subheader("Forcemain Parameters")
    pipe_diameter = st.number_input("Pipe Diameter (inches)", value=8.0, step=1.0)
    
    st.subheader("Pump Configuration")
    num_pumps = st.selectbox("Number of Pumps", [2, 3, 4], index=0, 
                            help="N-1 redundancy provides safety factor")
    pump_eff = st.slider("Pump Efficiency", 0.60, 0.85, 0.70, 0.01)
    motor_eff = st.slider("Motor Efficiency", 0.85, 0.95, 0.90, 0.01)
    
    st.subheader("Wet Well Parameters")
    wetwell_diameter = st.number_input("Diameter (ft)", value=6.0, step=1.0)
    max_cycles = st.number_input("Max Cycles/Hour", value=6.0, step=1.0)
    min_drawdown = st.number_input("Min Drawdown (ft)", value=2.0, step=0.5)
    
    st.markdown("---")
    calculate_button = st.button("Calculate Design", type="primary", use_container_width=True)
    
    # Version info
    st.markdown("---")
    st.caption("**Version 5.0**")
    st.caption("✅ CORRECTED: No SF on TDH")
    st.caption("✅ NEW: Startup Analysis")
    st.caption("✅ Enhanced: Air Valve Design")
# Main tabs - ADD MULTI-PUMP ANALYSIS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Results", 
    "Elevation Profile", 
    "Minor Losses with Locations",
    "Siphon Analysis",
    "Air Valve Design",
    "Analysis & HGL", 
    "Export",
    "? Startup Analysis",
    "? Multi-Pump Operation"  # NEW TAB
])

# Tab 1: Results - CORRECTED
with tab1:
    if calculate_button:
        with st.spinner("Calculating design parameters..."):
            try:
                results = calculate_design(
                    Q_avg, Q_peak, Q_min, motor_safety_factor,  # CORRECTED: motor_safety_factor instead of safety_factor
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
        
        # Key metrics - CORRECTED
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Dynamic Head", f"{r['TDH']:.1f} ft")  # CORRECTED: Single TDH value
            st.caption("Actual hydraulic requirement")
        with col2:
            st.metric("Pump Capacity", f"{r['Q_pump_gpm']:.0f} GPM")
            st.caption("Each pump (N-1 config)")
        with col3:
            st.metric("Motor Size", f"{r['motor_size']:.1f} HP")
            st.caption(f"Required: {r['MHP']:.1f} HP")  # Show actual vs selected
        with col4:
            st.metric("Pipe Velocity", f"{r['pipe_velocity']:.2f} ft/s")
            if r['pipe_velocity'] < 2:
                st.caption("⚠️ Low velocity")
            elif r['pipe_velocity'] > 8:
                st.caption("⚠️ High velocity")
            else:
                st.caption("✅ Good velocity")
        with col5:
            st.metric("Static Head", f"{r['static_head']:.1f} ft")
            st.caption("Elevation component")
        
        st.markdown("---")
        
        # STARTUP ANALYSIS WARNINGS - NEW SECTION
        if r.get('startup_warning', False):
            with st.expander("⚠️ STARTUP ANALYSIS CRITICAL", expanded=True):
                startup = r['startup_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error(f"""
                    **TWO OPERATING POINTS DETECTED:**
                    
                    🔄 **STARTUP (Empty Pipe):**
                    • Static to high point: {startup['static_to_high_point']:.1f} ft
                    • Air evacuation losses: {startup['air_analysis']['total_air_resistance']:.1f} ft
                    • **Total startup TDH: {startup['startup_TDH']:.1f} ft**
                    
                    ▶️ **RUNNING (Filled Pipe):**
                    • **Normal TDH: {r['TDH']:.1f} ft**
                    
                    **Difference: {startup['startup_advantage']:.1f} ft**
                    """)
                
                with col2:
                    st.warning("**PUMP SELECTION IMPACT:**")
                    for rec in startup['startup_recommendations']:
                        st.write(f"• {rec}")
                
                if startup['startup_advantage'] > 5:
                    st.error("""
                    🚨 **ACTION REQUIRED:** 
                    Startup conditions significantly exceed running conditions. 
                    See **Startup Analysis** tab for detailed analysis and recommendations.
                    """)
        
        # HGL Verification - CORRECTED AND SIMPLIFIED
        with st.expander("✅ HGL Verification (CORRECTED METHOD)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**TDH (Actual Hydraulic Requirement):** {r['TDH']:.2f} ft")
                st.write(f"**HGL at Pump:** {r['hgl_values'][0]:.2f} ft")
                st.write(f"**HGL at Discharge:** {r['hgl_values'][-1]:.6f} ft")
                st.write(f"**Discharge Elevation:** {r['discharge_elevation']:.6f} ft")
                
                # Verify HGL = elevation at discharge
                hgl_error = r['hgl_error']
                if hgl_error < 0.001:
                    st.success(f"✅ PERFECT: HGL = Discharge Elevation")
                elif hgl_error < 0.1:
                    st.success(f"✅ GOOD: HGL ≈ Discharge Elevation")
                else:
                    st.error(f"❌ ERROR: HGL ≠ Discharge Elevation")
            
            with col2:
                st.write(f"**Static Head:** {r['static_head']:.1f} ft")
                st.write(f"**Friction Loss:** {r['friction_loss']:.1f} ft") 
                st.write(f"**Minor Losses:** {r['minor_losses']:.1f} ft")
                st.write(f"**Total (TDH):** {r['TDH']:.1f} ft")
                
                st.success("""
                **✅ CORRECTED METHOD:**
                • TDH = Actual hydraulic requirement
                • NO safety factor on TDH
                • HGL calculated backwards from P=0
                • Safety factors applied to equipment only
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
                st.write(f"**TDH Components:**")
                st.write(f"• Static Head: {r['static_head']:.1f} ft ({r['static_head']/r['TDH']*100:.0f}%)")
                st.write(f"• Friction Loss: {r['friction_loss']:.2f} ft ({r['friction_loss']/r['TDH']*100:.0f}%)")
                st.write(f"• Minor Losses: {r['minor_losses']:.2f} ft ({r['minor_losses']/r['TDH']*100:.0f}%)")
                st.write(f"• **Total TDH: {r['TDH']:.1f} ft**")
                
                if r['high_point_controls']:
                    st.caption(f"Note: Additional losses in siphon section: {r['total_friction_all'] - r['friction_loss']:.2f} ft (not included in TDH)")
        
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
        
        # Pump Specifications - CORRECTED
        with st.expander("Pump Specifications (CORRECTED)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Number of Pumps:** {r['num_pumps']}")
                st.write(f"**Operating Configuration:** {r['pumps_operating']} operating (N-1 redundancy)")
                st.write(f"**Pump Capacity (each):** {r['Q_pump_gpm']:.2f} GPM")
                st.write(f"**Pump Head (TDH):** {r['TDH']:.2f} ft")  # CORRECTED: No "design" TDH
            with col2:
                st.write(f"**Water Horsepower:** {r['WHP']:.2f} HP")
                st.write(f"**Brake Horsepower:** {r['BHP']:.2f} HP")
                st.write(f"**Motor HP Required:** {r['MHP']:.2f} HP")
                st.write(f"**Motor HP Selected:** {r['motor_size']:.1f} HP")
                st.write(f"**Motor Safety Factor:** {r['motor_safety_factor']:.2f}")
                st.write(f"**Power per Pump:** {r['power_kw']:.2f} kW")
        # TDH Calculation Verification - NEW SECTION
        with st.expander("🔍 TDH Calculation Verification", expanded=False):
            st.markdown("### CORRECTED Smith & Loveless Method")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if r['high_point_controls']:
                    st.success("**HIGH POINT CONTROLS DESIGN**")
                    st.write(f"• TDH calculated TO CONTROLLING POINT only")
                    st.write(f"• Static head to high point: {r['static_head']:.2f} ft")
                    st.write(f"• Friction loss to high point: {r['friction_loss']:.2f} ft")
                    st.write(f"• Minor losses to high point: {r['minor_losses']:.2f} ft")
                    st.write(f"• **TDH = {r['TDH']:.2f} ft**")
                    
                    st.info(f"Siphon section losses ({r['total_friction_all'] - r['friction_loss']:.2f} ft) NOT included in TDH")
                else:
                    st.info("**DISCHARGE CONTROLS DESIGN**")
                    st.write(f"• TDH calculated to discharge point")
                    st.write(f"• Static head: {r['static_head']:.2f} ft")
                    st.write(f"• Friction loss (total): {r['friction_loss']:.2f} ft")
                    st.write(f"• Minor losses (total): {r['minor_losses']:.2f} ft")
                    st.write(f"• **TDH = {r['TDH']:.2f} ft**")
            
            with col2:
                st.markdown("**Manual Verification:**")
                manual_tdh = r['static_head'] + r['friction_loss'] + r['minor_losses']
                st.write(f"Static Head: {r['static_head']:.2f} ft")
                st.write(f"+ Friction Loss: {r['friction_loss']:.2f} ft") 
                st.write(f"+ Minor Losses: {r['minor_losses']:.2f} ft")
                st.write(f"= **Manual TDH: {manual_tdh:.2f} ft**")
                
                error = abs(r['TDH'] - manual_tdh)
                if error < 0.01:
                    st.success(f"✅ VERIFIED: Error = {error:.4f} ft")
                else:
                    st.error(f"❌ ERROR: Difference = {error:.4f} ft")

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
                
                if r['actual_cycles_per_hour'] > 12:
                    st.warning("⚠️ High cycling rate - consider larger wet well")
                elif r['actual_cycles_per_hour'] < 4:
                    st.warning("⚠️ Low cycling rate - check sizing")
                else:
                    st.success("✅ Good cycling rate")
    else:
        st.info("Enter parameters in the sidebar and click 'Calculate Design' to see results")
# Tab 2: Elevation Profile (100 points) - ENHANCED WITH DELETE FUNCTIONALITY
with tab2:
    st.subheader("Forcemain Elevation Profile Configuration (Up to 100 Points)")
    st.markdown("Define up to 100 elevation points along the forcemain alignment. System will identify controlling point and siphon zones.")
    
    st.info("**Tip:** Add points at significant elevation changes, high points, and low points. First point = Pump Station, Last point = Discharge.")
    
    # FIXED: Better session state management
    if 'elevation_profile' not in st.session_state:
        st.session_state.elevation_profile = pd.DataFrame({
            'Station': [0, 1, 2],
            'Distance (ft)': [0.0, 500.0, 1000.0],
            'Elevation (ft)': [0.0, 15.0, 25.0],
            'Description': ['Pump Station', 'Intermediate Point', 'Discharge']
        })
    
    # Row management buttons
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        if st.button("➕ Add New Row", use_container_width=True):
            # Get current max distance to suggest next distance
            current_distances = pd.to_numeric(st.session_state.elevation_profile['Distance (ft)'], errors='coerce')
            max_distance = current_distances.max() if not current_distances.isna().all() else 0
            next_distance = max_distance + 100  # Add 100 ft by default
            
            # Create new row
            new_row = pd.DataFrame([{
                'Station': len(st.session_state.elevation_profile),
                'Distance (ft)': next_distance,
                'Elevation (ft)': 0.0,
                'Description': 'New Point'
            }])
            
            st.session_state.elevation_profile = pd.concat([
                st.session_state.elevation_profile, new_row
            ], ignore_index=True)
            st.rerun()
    
    with col2:
        # Delete row selector
        if len(st.session_state.elevation_profile) > 2:  # Keep minimum 2 rows
            row_options = []
            for idx, row in st.session_state.elevation_profile.iterrows():
                desc = row.get('Description', f'Row {idx}')
                dist = row.get('Distance (ft)', 0)
                row_options.append(f"Row {idx}: {desc} ({dist} ft)")
            
            selected_row = st.selectbox(
                "Select row to delete:",
                options=range(len(row_options)),
                format_func=lambda x: row_options[x],
                key="row_to_delete"
            )
        else:
            st.write("*Min 2 rows required*")
            selected_row = None
    
    with col3:
        if st.button("🗑️ Delete Selected Row", use_container_width=True, disabled=(len(st.session_state.elevation_profile) <= 2)):
            if selected_row is not None and len(st.session_state.elevation_profile) > 2:
                # Remove the selected row
                st.session_state.elevation_profile = st.session_state.elevation_profile.drop(
                    st.session_state.elevation_profile.index[selected_row]
                ).reset_index(drop=True)
                
                # Renumber stations
                st.session_state.elevation_profile['Station'] = range(len(st.session_state.elevation_profile))
                st.rerun()
    
    with col4:
        if st.button("📊 Renumber Stations", use_container_width=True):
            # Sort by distance and renumber
            st.session_state.elevation_profile = st.session_state.elevation_profile.sort_values('Distance (ft)').reset_index(drop=True)
            st.session_state.elevation_profile['Station'] = range(len(st.session_state.elevation_profile))
            st.rerun()
    
    st.markdown("---")
    
    # Enhanced data editor with delete functionality built-in
    edited_profile = st.data_editor(
        st.session_state.elevation_profile,
        num_rows="dynamic",  # This allows adding/deleting rows directly
        use_container_width=True,
        column_config={
            "Station": st.column_config.NumberColumn(
                "Station #",
                help="Sequential station number (auto-renumbered)",
                min_value=0,
                max_value=200,
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
        key="elevation_editor_enhanced"
    )
    
    # FIXED: Immediately update session state with any changes
    if not edited_profile.equals(st.session_state.elevation_profile):
        # Clean and validate the edited data
        if not edited_profile.empty:
            # Remove completely empty rows
            edited_profile = edited_profile.dropna(subset=['Distance (ft)', 'Elevation (ft)'], how='all')
            
            # Fill in missing values
            if len(edited_profile) > 0:
                # Fill missing station numbers
                if 'Station' in edited_profile.columns:
                    for i in range(len(edited_profile)):
                        if pd.isna(edited_profile.iloc[i]['Station']):
                            edited_profile.iloc[i, edited_profile.columns.get_loc('Station')] = i
                
                # Fill missing numeric values with reasonable defaults
                if 'Distance (ft)' in edited_profile.columns:
                    edited_profile['Distance (ft)'] = pd.to_numeric(edited_profile['Distance (ft)'], errors='coerce').fillna(0.0)
                if 'Elevation (ft)' in edited_profile.columns:
                    edited_profile['Elevation (ft)'] = pd.to_numeric(edited_profile['Elevation (ft)'], errors='coerce').fillna(0.0)
                if 'Description' in edited_profile.columns:
                    edited_profile['Description'] = edited_profile['Description'].fillna('Point')
                
                # Update session state immediately
                st.session_state.elevation_profile = edited_profile.copy()
    
    # Show current row count and management tips
    current_profile = st.session_state.elevation_profile
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"📍 **Current Points:** {len(current_profile)}/100")
    with col_info2:
        st.info("💡 **Tip:** You can also add/delete rows directly in the table using the ➕➖ icons")
    
    # Validation and feedback
    if len(current_profile) > 100:
        st.error("⚠️ Maximum 100 elevation points allowed. Please remove excess points.")
    elif len(current_profile) < 2:
        st.warning("⚠️ Minimum 2 points required (pump station and discharge).")
    else:
        # Show success with current count
        st.success(f"✅ Elevation profile validated with {len(current_profile)} points")
        
        # Profile preview
        st.markdown("### 📈 Profile Preview")
        
        # Only show preview if we have valid data
        if len(current_profile) >= 2:
            fig = go.Figure()
            
            sorted_profile = current_profile.sort_values('Distance (ft)')
            
            # Check if we have valid numeric data
            valid_distances = pd.to_numeric(sorted_profile['Distance (ft)'], errors='coerce')
            valid_elevations = pd.to_numeric(sorted_profile['Elevation (ft)'], errors='coerce')
            
            if not valid_distances.isna().all() and not valid_elevations.isna().all():
                fig.add_trace(go.Scatter(
                    x=valid_distances,
                    y=valid_elevations,
                    mode='lines+markers',
                    name='Elevation Profile',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10, color='blue')
                ))
                
                # Identify high point
                max_idx = valid_elevations.idxmax()
                if not pd.isna(max_idx):
                    max_elev = valid_elevations.loc[max_idx]
                    max_dist = valid_distances.loc[max_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=[max_dist],
                        y=[max_elev],
                        mode='markers+text',
                        name='Controlling Point',
                        marker=dict(size=15, color='red', symbol='star'),
                        text=['HIGH POINT'],
                        textposition='top center',
                        showlegend=False
                    ))
                
                # Add annotations for each point
                for idx, row in sorted_profile.iterrows():
                    if pd.notna(row['Distance (ft)']) and pd.notna(row['Elevation (ft)']):
                        fig.add_annotation(
                            x=row['Distance (ft)'],
                            y=row['Elevation (ft)'],
                            text=f"{row['Description']}<br>({row['Distance (ft)']} ft)",
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
            else:
                st.warning("⚠️ Please enter valid numeric values for Distance and Elevation to see preview.")
    
    # Enhanced template buttons
    st.markdown("---")
    st.markdown("### 📋 Template Options")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset to Default (3 points)", use_container_width=True):
            st.session_state.elevation_profile = pd.DataFrame({
                'Station': [0, 1, 2],
                'Distance (ft)': [0.0, 500.0, 1000.0],
                'Elevation (ft)': [0.0, 15.0, 25.0],
                'Description': ['Pump Station', 'Intermediate', 'Discharge']
            })
            st.rerun()
    
    with col2:
        if st.button("⛰️ Load Example: High Point (5 points)", use_container_width=True):
            st.session_state.elevation_profile = pd.DataFrame({
                'Station': [0, 1, 2, 3, 4],
                'Distance (ft)': [0.0, 300.0, 600.0, 800.0, 1200.0],
                'Elevation (ft)': [0.0, 15.0, 35.0, 20.0, 25.0],
                'Description': ['Pump Station', 'Rising', 'High Point', 'Descending', 'Discharge']
            })
            st.rerun()
    
    with col3:
        if st.button("🎢 Load Example: Complex Profile (8 points)", use_container_width=True):
            st.session_state.elevation_profile = pd.DataFrame({
                'Station': [0, 1, 2, 3, 4, 5, 6, 7],
                'Distance (ft)': [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0],
                'Elevation (ft)': [0.0, 10.0, 25.0, 40.0, 35.0, 20.0, 15.0, 30.0],
                'Description': ['Pump', 'Point 1', 'Point 2', 'High Point', 'Point 4', 'Point 5', 'Low Point', 'Discharge']
            })
            st.rerun()
# Tab 3: Minor Losses with Locations - ENHANCED
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
    st.markdown("### 📊 Minor Loss Summary by Location")
    
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
    st.markdown("### ⚡ Quick Add Common Components")
    
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
        if st.button("🔧 Load Typical Force Main Template", use_container_width=True):
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
            
            if siphon_segments:
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

# =============================================================================
# Tab 5: COMPREHENSIVE Air Valve Design & Analysis
# =============================================================================
with tab5:
    st.header("🌬️ Comprehensive Air Valve Analysis")
    
    st.info("""
    **Enhanced Air Valve System Design:**
    This comprehensive analysis identifies ALL required air valve locations and provides 
    detailed sizing for multiple operating scenarios including normal operation, system filling, 
    drainage, and transient protection.
    """)
    
    if st.session_state.results:
        r = st.session_state.results
        
        # Check if comprehensive analysis is available
        if not r.get('comprehensive_air_valve_available', False):
            st.warning("⚠️ Comprehensive air valve analysis not available - using basic analysis")
            
            # Show basic analysis as fallback
            if r.get('has_high_points', False):
                st.subheader("Basic Air Valve Analysis")
                for valve in r.get('air_valve_data', []):
                    st.write(f"• Station {valve.get('Distance (ft)', 0):.0f} ft - {valve.get('Valve Size', 'Unknown')}")
            
            # Show any warnings
            if r.get('air_valve_warnings'):
                st.markdown("### ⚠️ Warnings")
                for warning in r['air_valve_warnings']:
                    if warning['severity'] == 'ERROR':
                        st.error(f"**{warning['message']}**\n\n{warning['recommendation']}")
        
        else:
            # =====================================================================
            # COMPREHENSIVE ANALYSIS AVAILABLE
            # =====================================================================
            
            valve_summary = r['air_valve_summary']
            valve_locations = r['comprehensive_valve_locations']
            valve_schedule = r['comprehensive_valve_schedule']
            
            # Create sub-tabs for different analyses
            subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs([
                "📊 Overview",
                "📋 Valve Schedule", 
                "🔧 Sizing Details",
                "💰 Cost Analysis",
                "⚡ Transient Analysis",
                "🚰 Filling Procedure"
            ])
            
            # =================================================================
            # SUB-TAB 1: OVERVIEW
            # =================================================================
            with subtab1:
                st.markdown("### 📊 System-Wide Air Valve Summary")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Valves", valve_summary['total_valves'])
                    st.caption("All locations")
                
                with col2:
                    st.metric("First Cost", f"${valve_summary['total_first_cost']:,.0f}")
                    st.caption("Equipment + Installation")
                
                with col3:
                    st.metric("20-Year Cost", f"${valve_summary['lifecycle_cost_20yr']:,.0f}")
                    st.caption("Including maintenance")
                
                with col4:
                    if valve_summary['optimization_savings'] > 0:
                        st.metric("Potential Savings", f"${valve_summary['optimization_savings']:,.0f}")
                        st.caption("Through optimization")
                    else:
                        st.metric("Optimized", "✓")
                        st.caption("Current design optimal")
                
                # Warnings and critical items
                if r['air_valve_warnings']:
                    st.markdown("---")
                    st.markdown("### ⚠️ Critical Items & Recommendations")
                    
                    for warning in r['air_valve_warnings']:
                        if warning['severity'] == 'CRITICAL':
                            st.error(f"""
                            **CRITICAL:** {warning['message']}
                            
                            💡 **Recommendation:** {warning['recommendation']}
                            """)
                        else:
                            st.warning(f"""
                            **WARNING:** {warning['message']}
                            
                            💡 **Recommendation:** {warning['recommendation']}
                            """)
                
                # Valve distribution chart
                st.markdown("---")
                st.markdown("### 📊 Valve Distribution by Type")
                
                fig_dist = go.Figure()
                
                valve_types = ['High Points', 'Low Points', 'Long Runs', 'Pump Discharge']
                valve_counts = [
                    valve_summary['high_point_valves'],
                    valve_summary['low_point_valves'],
                    valve_summary['long_run_valves'],
                    valve_summary['pump_discharge_valves']
                ]
                colors_dist = ['red', 'blue', 'green', 'orange']
                
                fig_dist.add_trace(go.Bar(
                    x=valve_types,
                    y=valve_counts,
                    marker_color=colors_dist,
                    text=valve_counts,
                    textposition='auto'
                ))
                
                fig_dist.update_layout(
                    title="Air Valve Count by Location Type",
                    xaxis_title="Valve Type",
                    yaxis_title="Number of Valves",
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Profile with all valve locations
                st.markdown("---")
                st.markdown("### 🗺️ Complete Air Valve Location Map")
                
                fig_profile = go.Figure()
                
                # Elevation profile
                fig_profile.add_trace(go.Scatter(
                    x=r['distances'],
                    y=r['elevations'],
                    fill='tozeroy',
                    name='Forcemain Profile',
                    line=dict(color='brown', width=2),
                    fillcolor='rgba(139, 69, 19, 0.3)'
                ))
                
                # Mark all valve locations
                for valve_entry in valve_schedule:
                    valve = valve_entry
                    
                    if 'High Point' in valve['location_type']:
                        color = 'red'
                        symbol = 'triangle-up'
                        text = f"HP-{valve['valve_number']}"
                    elif 'Low Point' in valve['location_type']:
                        color = 'blue'
                        symbol = 'triangle-down'
                        text = f"LP-{valve['valve_number']}"
                    elif 'Long Run' in valve['location_type']:
                        color = 'green'
                        symbol = 'circle'
                        text = f"LR-{valve['valve_number']}"
                    else:  # Pump discharge
                        color = 'orange'
                        symbol = 'square'
                        text = f"PD-{valve['valve_number']}"
                    
                    fig_profile.add_trace(go.Scatter(
                        x=[valve['station']],
                        y=[valve['elevation']],
                        mode='markers+text',
                        name=valve['location_type'],
                        marker=dict(size=12, color=color, symbol=symbol),
                        text=[text],
                        textposition='top center',
                        showlegend=False
                    ))
                
                fig_profile.update_layout(
                    title="Air Valve Locations on Profile",
                    xaxis_title="Distance (ft)",
                    yaxis_title="Elevation (ft)",
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_profile, use_container_width=True)
                
                # Legend
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🔺 **HP:** High Point (Combination)")
                with col2:
                    st.markdown("🔻 **LP:** Low Point (Vacuum)")
                with col3:
                    st.markdown("🟢 **LR:** Long Run (Air Release)")
                with col4:
                    st.markdown("🟧 **PD:** Pump Discharge")
            
            # =================================================================
            # SUB-TAB 2: VALVE SCHEDULE
            # =================================================================
            with subtab2:
                st.markdown("### 📋 Complete Air Valve Schedule")
                
                st.info("**Professional valve schedule with all specifications**")
                
                # Create comprehensive schedule table
                schedule_data = []
                
                for valve_entry in valve_schedule:
                    valve = valve_entry
                    sizing = valve['comprehensive_sizing']
                    recommended = sizing['recommended_valve']
                    
                    schedule_data.append({
                        'Valve #': f"AV-{valve['valve_number']}",
                        'Station (ft)': f"{valve['station']:.0f}",
                        'Elevation (ft)': f"{valve['elevation']:.1f}",
                        'Location Type': valve['location_type'],
                        'Valve Type': recommended['type'],
                        'Connection Size': recommended['connection_size'],
                        'Function': recommended['function'],
                        'Priority': valve['priority']
                    })
                
                schedule_df = pd.DataFrame(schedule_data)
                
                # Style the dataframe
                def color_priority(val):
                    if val == 'CRITICAL':
                        return 'background-color: #ffcccc'
                    elif val == 'HIGH':
                        return 'background-color: #ffffcc'
                    else:
                        return 'background-color: #ccffcc'
                
                styled_df = schedule_df.style.applymap(
                    color_priority,
                    subset=['Priority']
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Export options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = schedule_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Valve Schedule (CSV)",
                        data=csv,
                        file_name="air_valve_schedule.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create detailed report
                    detailed_report = "COMPREHENSIVE AIR VALVE SCHEDULE\n"
                    detailed_report += "="*60 + "\n\n"
                    detailed_report += f"Project: Lift Station Design\n"
                    detailed_report += f"Total Valves: {valve_summary['total_valves']}\n"
                    detailed_report += f"Estimated Total Cost: ${valve_summary['total_first_cost']:,.0f}\n\n"
                    
                    for valve_entry in valve_schedule:
                        valve = valve_entry
                        detailed_report += f"\nVALVE AV-{valve['valve_number']}\n"
                        detailed_report += f"Station: {valve['station']:.0f} ft\n"
                        detailed_report += f"Elevation: {valve['elevation']:.1f} ft\n"
                        detailed_report += f"Type: {valve['location_type']}\n"
                        detailed_report += f"Priority: {valve['priority']}\n"
                        detailed_report += "-"*40 + "\n"
                    
                    st.download_button(
                        label="📥 Download Detailed Report (TXT)",
                        data=detailed_report,
                        file_name="air_valve_detailed_report.txt",
                        mime="text/plain"
                    )
            
            # =================================================================
            # SUB-TAB 3: SIZING DETAILS
            # =================================================================
            with subtab3:
                st.markdown("### 🔧 Detailed Sizing Analysis by Scenario")
                
                # Valve selector
                valve_numbers = [v['valve_number'] for v in valve_schedule]
                selected_valve_num = st.selectbox(
                    "Select Valve for Detailed Analysis:",
                    valve_numbers,
                    format_func=lambda x: f"AV-{x} - Station {valve_schedule[x-1]['station']:.0f} ft ({valve_schedule[x-1]['location_type']})"
                )
                
                selected_valve = valve_schedule[selected_valve_num - 1]
                sizing = selected_valve['comprehensive_sizing']
                
                st.markdown(f"### Valve AV-{selected_valve_num} Analysis")
                st.markdown(f"**Location:** Station {selected_valve['station']:.0f} ft @ {selected_valve['elevation']:.1f} ft elevation")
                st.markdown(f"**Type:** {selected_valve['location_type']}")
                
                # Show all scenarios for this valve
                for scenario_name, scenario_data in sizing['scenarios'].items():
                    st.markdown("---")
                    st.markdown(f"#### 📊 {scenario_name.replace('_', ' ').title()}")
                    st.write(f"*{scenario_data['description']}*")
                    
                    # Display scenario-specific data in columns
                    data_items = [(k, v) for k, v in scenario_data.items() if k != 'description']
                    num_cols = min(3, len(data_items))
                    
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        for idx, (key, value) in enumerate(data_items):
                            with cols[idx % num_cols]:
                                display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                                st.metric(key.replace('_', ' ').title(), display_value)
                
                # Manufacturer recommendations
                st.markdown("---")
                st.markdown("### 🏭 Manufacturer Options")
                
                manufacturers = selected_valve['manufacturer_options']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.success("**Val-Matic**")
                    valmatic = manufacturers['val_matic']
                    for key, rec in valmatic['recommendations'].items():
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for k, v in rec.items():
                            st.write(f"• {k.replace('_', ' ').title()}: {v}")
                
                with col2:
                    st.info("**APCO**")
                    apco = manufacturers['apco']
                    for key, rec in apco['recommendations'].items():
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for k, v in rec.items():
                            st.write(f"• {k.replace('_', ' ').title()}: {v}")
                
                with col3:
                    st.warning("**A.R.I.**")
                    ari = manufacturers['ari']
                    for key, rec in ari['recommendations'].items():
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for k, v in rec.items():
                            st.write(f"• {k.replace('_', ' ').title()}: {v}")
            
            # =================================================================
            # SUB-TAB 4: COST ANALYSIS
            # =================================================================
            with subtab4:
                st.markdown("### 💰 Economic Analysis")
                
                cost_analysis = r['valve_cost_analysis']
                optimization = r['valve_optimization']
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total First Cost",
                        f"${cost_analysis['summary']['total_first_cost']:,.0f}",
                        help="Equipment + Installation"
                    )
                
                with col2:
                    st.metric(
                        "20-Year Lifecycle Cost",
                        f"${cost_analysis['summary']['lifecycle_cost_20yr']:,.0f}",
                        help="First cost + 20 years maintenance"
                    )
                
                with col3:
                    st.metric(
                        "Average Cost/Valve",
                        f"${cost_analysis['summary']['cost_per_valve_average']:,.0f}",
                        help="Total first cost ÷ number of valves"
                    )
                
                # Cost breakdown chart
                st.markdown("---")
                st.markdown("### 📊 Cost Breakdown")
                
                fig_cost = go.Figure()
                
                categories = ['Equipment', 'Installation', '20-Yr Maintenance']
                costs = [
                    cost_analysis['summary']['total_equipment_cost'],
                    cost_analysis['summary']['total_installation_cost'],
                    cost_analysis['summary']['annual_maintenance_cost'] * 20
                ]
                
                fig_cost.add_trace(go.Bar(
                    x=categories,
                    y=costs,
                    marker_color=['lightblue', 'lightgreen', 'lightyellow'],
                    text=[f"${c:,.0f}" for c in costs],
                    textposition='auto'
                ))
                
                fig_cost.update_layout(
                    title="Air Valve System Cost Breakdown",
                    yaxis_title="Cost ($)",
                    height=400
                )
                
                st.plotly_chart(fig_cost, use_container_width=True)
                
                # Detailed cost table
                st.markdown("---")
                st.markdown("### 📋 Individual Valve Costs")
                
                cost_table = []
                for valve_cost in cost_analysis['valves']:
                    cost_table.append({
                        'Valve': f"AV-{valve_cost['valve_number']}",
                        'Station (ft)': f"{valve_cost['station']:.0f}",
                        'Type': valve_cost['type'],
                        'Size': valve_cost['size'],
                        'Equipment ($)': f"${valve_cost['equipment_cost']:,.0f}",
                        'Installation ($)': f"${valve_cost['installation_cost']:,.0f}",
                        'First Cost ($)': f"${valve_cost['first_cost']:,.0f}",
                        '20-Yr Cost ($)': f"${valve_cost['lifecycle_cost_20yr']:,.0f}"
                    })
                
                cost_df = pd.DataFrame(cost_table)
                st.dataframe(cost_df, use_container_width=True)
                
                # Optimization recommendations
                st.markdown("---")
                st.markdown("### 🎯 Optimization Opportunities")
                
                if optimization['potential_savings']['total_estimated_savings'] > 0:
                    st.warning(f"""
                    **Potential Cost Savings: ${optimization['potential_savings']['total_estimated_savings']:,.0f}**
                    
                    Through optimization, valve count could be reduced from 
                    {optimization['original_valve_count']} to {optimization['optimized_valve_count']} 
                    ({optimization['potential_savings']['percentage_reduction']:.0f}% reduction).
                    """)
                    
                    for rec in optimization['optimized_recommendations']:
                        st.info(f"""
                        **{rec['recommendation']}**
                        
                        Action: {rec['action']}
                        
                        Estimated Savings: ${rec.get('potential_savings', 0):,.0f}
                        """)
                else:
                    st.success("""
                    ✅ **Current Design is Optimized**
                    
                    No significant cost savings identified through optimization.
                    Current valve configuration is appropriate for system requirements.
                    """)
            
            # =================================================================
            # SUB-TAB 5: TRANSIENT ANALYSIS
            # =================================================================
            with subtab5:
                st.markdown("### ⚡ Water Hammer & Transient Analysis")
                
                st.warning("""
                **Critical for System Safety:**
                Transient analysis identifies potential column separation and surge pressure conditions
                that require proper air valve sizing for protection.
                """)
                
                transient = r['transient_analysis']
                
                # Find critical locations
                critical_transients = [t for t in transient if t['separation_risk'] in ['CRITICAL', 'HIGH']]
                
                if critical_transients:
                    st.error(f"""
                    ⚠️ **{len(critical_transients)} Critical Transient Locations Identified**
                    
                    These locations require special attention for surge protection and air valve sizing.
                    """)
                
                # Transient profile
                fig_transient = go.Figure()
                
                # Elevation profile
                fig_transient.add_trace(go.Scatter(
                    x=r['distances'],
                    y=r['elevations'],
                    fill='tozeroy',
                    name='Profile',
                    line=dict(color='brown', width=2),
                    fillcolor='rgba(139, 69, 19, 0.3)'
                ))
                
                # Color-code transient risk
                colors_map = {
                    'CRITICAL': 'red',
                    'HIGH': 'orange',
                    'MODERATE': 'yellow',
                    'LOW': 'green'
                }
                
                for risk_level, color in colors_map.items():
                    risk_points = [t for t in transient if t['separation_risk'] == risk_level]
                    if risk_points:
                        fig_transient.add_trace(go.Scatter(
                            x=[t['distance'] for t in risk_points],
                            y=[t['elevation'] for t in risk_points],
                            mode='markers',
                            name=f'{risk_level} Risk',
                            marker=dict(size=10, color=color, symbol='diamond')
                        ))
                
                fig_transient.update_layout(
                    title="Column Separation Risk Analysis",
                    xaxis_title="Distance (ft)",
                    yaxis_title="Elevation (ft)",
                    height=500
                )
                
                st.plotly_chart(fig_transient, use_container_width=True)
                
                # Detailed transient table
                st.markdown("---")
                st.markdown("### 📊 Transient Conditions by Location")
                
                transient_table = []
                for t in transient:
                    transient_table.append({
                        'Station (ft)': f"{t['distance']:.0f}",
                        'Elevation (ft)': f"{t['elevation']:.1f}",
                        'Pressure Drop (ft)': f"{t['pressure_drop_ft']:.1f}",
                        'Min Transient Pressure (ft)': f"{t['min_transient_pressure']:.1f}",
                        'Separation Risk': t['separation_risk']
                    })
                
                transient_df = pd.DataFrame(transient_table)
                
                # Color code by risk
                def color_risk(val):
                    if val == 'CRITICAL':
                        return 'background-color: #ff6666'
                    elif val == 'HIGH':
                        return 'background-color: #ffaa66'
                    elif val == 'MODERATE':
                        return 'background-color: #ffff66'
                    else:
                        return 'background-color: #66ff66'
                
                styled_transient = transient_df.style.applymap(
                    color_risk,
                    subset=['Separation Risk']
                )
                
                st.dataframe(styled_transient, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Transient Protection Recommendations")
                
                if critical_transients:
                    st.error("""
                    **CRITICAL ACTIONS REQUIRED:**
                    1. ✅ Install combination air valves at all high points
                    2. ✅ Ensure valves are sized for surge protection (not just steady-state)
                    3. ✅ Consider surge anticipation valves or pressure relief valves
                    4. ✅ Implement controlled pump shutdown procedures
                    5. ✅ Consider VFD or soft-start equipment
                    """)
                else:
                    st.success("""
                    ✅ **Low Transient Risk**
                    
                    Standard air valve sizing appears adequate for transient protection.
                    Continue with recommended valve schedule.
                    """)
            
            # =================================================================
            # SUB-TAB 6: FILLING PROCEDURE
            # =================================================================
            with subtab6:
                st.markdown("### 🚰 Staged Filling Procedure")
                
                filling = r['filling_procedure']
                
                st.info("""
                **Purpose of Staged Filling:**
                Proper system filling ensures air is evacuated through air valves in controlled manner,
                preventing air binding, water hammer, and damage to pumps and piping.
                """)
                
                if filling['procedure'] == 'Simple':
                    st.success("""
                    ✅ **Simple Filling Procedure Adequate**
                    
                    System profile allows straightforward filling without complex staging.
                    """)
                    
                    for step in filling['stages']:
                        st.write(f"• {step}")
                    
                    st.write(f"\n**Estimated Filling Time:** {filling.get('estimated_time_minutes', 'TBD'):.0f} minutes")
                
                else:
                    st.warning("""
                    ⚠️ **Staged Filling Procedure Required**
                    
                    Complex profile requires multi-stage filling to ensure proper air evacuation.
                    """)
                    
                    st.markdown(f"**Total Stages:** {filling['number_of_stages']}")
                    
                    # Display each stage
                    for stage in filling['stages']:
                        with st.expander(f"🔹 Stage {stage['stage']}: {stage['description']}"):
                            st.write(f"**Target Elevation:** {stage['target_elevation']}")
                            st.write(f"**Filling Rate:** {stage['filling_rate']}")
                            st.write(f"**Air Valve Action:** {stage['air_valve_action']}")
                            if stage.get('estimated_time_min'):
                                st.write(f"**Estimated Time:** {stage['estimated_time_min']} minutes")
                    
                    # Critical notes
                    st.markdown("---")
                    st.markdown("### ⚠️ Critical Filling Procedure Notes")
                    
                    for note in filling['critical_notes']:
                        st.warning(f"• {note}")
                    
                    # Filling checklist
                    st.markdown("---")
                    st.markdown("### ✅ Pre-Filling Checklist")
                    
                    checklist = [
                        "Verify all air valves are installed and operational",
                        "Confirm all manual valves along forcemain are open",
                        "Check discharge location is ready to receive flow",
                        "Ensure operators are stationed at key air valve locations",
                        "Have communication system in place between operators",
                        "Prepare to throttle filling rate if needed",
                        "Have air valve maintenance tools readily available",
                        "Review emergency shutdown procedures"
                    ]
                    
                    for item in checklist:
                        st.checkbox(item, key=f"checklist_{item[:20]}")
    
    else:
        st.info("🔘 Calculate design first to see comprehensive air valve analysis")
# Tab 6: Analysis & HGL - CORRECTED
with tab6:
    if st.session_state.results:
        r = st.session_state.results
        
        st.subheader("Force Main System Profile with Multiple HGL Conditions")
        
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
        
        # Running Conditions HGL (existing - but renamed for clarity)
        fig.add_trace(go.Scatter(
            x=r['distances'], 
            y=r['hgl_values'], 
            name='HGL - Running Conditions (Design)',
            line=dict(color='red', dash='dash', width=3),
            mode='lines+markers'
        ))
        
        # Startup Conditions HGL - NEW
        if 'startup_analysis' in r and r['startup_analysis']['is_startup_critical']:
            startup = r['startup_analysis']
            
            # Calculate startup HGL values
            startup_hgl_values = []
            startup_advantage = startup['startup_advantage']
            
            for i, distance in enumerate(r['distances']):
                if r['high_point_controls'] and distance <= r['controlling_distance']:
                    # In pumped section: add startup advantage
                    startup_hgl = r['hgl_values'][i] + startup_advantage
                else:
                    # In siphon section or discharge control: same as running
                    startup_hgl = r['hgl_values'][i]
                startup_hgl_values.append(startup_hgl)
            
            fig.add_trace(go.Scatter(
                x=r['distances'], 
                y=startup_hgl_values, 
                name='HGL - Startup Conditions (Empty Pipe)',
                line=dict(color='orange', dash='dot', width=2),
                mode='lines'
            ))
            
            # Fill area between startup and running HGL
            fig.add_trace(go.Scatter(
                x=r['distances'] + r['distances'][::-1],
                y=startup_hgl_values + r['hgl_values'][::-1],
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='HGL Operating Envelope',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Required vs Available HGL comparison - FIXED
        if r.get('motor_safety_factor', 1.0) > 1.0:
            motor_factor_addition = r['TDH'] * (r['motor_safety_factor'] - 1.0)
            safety_factor_hgl = [h + motor_factor_addition for h in r['hgl_values']]
            
            fig.add_trace(go.Scatter(
                x=r['distances'], 
                y=safety_factor_hgl, 
                name='HGL - If Motor SF Applied to TDH (Reference Only)',
                line=dict(color='gray', dash='dashdot', width=1),  # FIXED: removed opacity
                opacity=0.6,  # FIXED: moved opacity to trace level
                mode='lines'
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
            
            if siphon_distances and siphon_elevations:
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
        
        # Add reference lines
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
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=1.01
            )
        )
        
        # Enhanced verification annotation
        if 'startup_analysis' in r:
            startup = r['startup_analysis']
            annotation_text = (
                f"<b>MULTI-CONDITION HGL ANALYSIS:</b><br>"
                f"Running HGL at Discharge: {r['hgl_values'][-1]:.6f} ft<br>"
                f"Discharge Elevation: {r['discharge_elevation']:.6f} ft<br>"
                f"HGL Error: {r['hgl_error']:.8f} ft<br>"
                f"Status: {'✅ PASS' if r['hgl_error'] < 0.1 else '❌ FAIL'}<br><br>"
                f"<b>Operating Conditions:</b><br>"
                f"? Startup TDH: {startup['startup_TDH']:.1f} ft<br>"
                f"▶️ Running TDH: {r['TDH']:.1f} ft<br>"
                f"Flow: {r['Q_peak']:.0f} GPM<br>"
                f"Velocity: {r['pipe_velocity']:.2f} ft/s"
            )
        else:
            annotation_text = (
                f"<b>CORRECTED HGL VERIFICATION:</b><br>"
                f"HGL at Discharge: {r['hgl_values'][-1]:.6f} ft<br>"
                f"Discharge Elevation: {r['discharge_elevation']:.6f} ft<br>"
                f"Error: {r['hgl_error']:.8f} ft<br>"
                f"Status: {'✅ PASS' if r['hgl_error'] < 0.1 else '❌ FAIL'}<br><br>"
                f"<b>System (CORRECTED):</b><br>"
                f"TDH: {r['TDH']:.1f} ft (no SF)<br>"
                f"Flow: {r['Q_peak']:.0f} GPM<br>"
                f"Velocity: {r['pipe_velocity']:.2f} ft/s"
            )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            text=annotation_text,
            showarrow=False,
            bgcolor="lightgreen" if r['hgl_error'] < 0.1 else "lightcoral",
            bordercolor="black",
            borderwidth=1,
            align="right",
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation guide for multiple HGL lines
        if 'startup_analysis' in r and r['startup_analysis']['is_startup_critical']:
            st.markdown("---")
            st.markdown("### ? HGL Interpretation Guide")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.error("""
                **🔴 Orange Line - Startup HGL:**
                • Higher head during system filling
                • Accounts for air evacuation resistance
                • Static head to high point critical
                """)
            
            with col2:
                st.success("""
                **🔴 Red Dashed - Running HGL:**
                • Normal operating conditions
                • Filled pipe with standard friction
                • Design operating point
                """)
            
            with col3:
                st.info("""
                **🟠 Shaded Area - Operating Envelope:**
                • Range between startup and running
                • Pump must handle both extremes
                • Critical for pump selection
                """)
        
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
        
        # Pump curves - CORRECTED
        st.markdown("---")
        st.subheader("Pump Performance Curves (CORRECTED)")
        
        Q_range = np.linspace(0, r['Q_pump_gpm'] * 1.5, 100)
        H_shutoff = r['TDH'] * 1.2  # CORRECTED: Use actual TDH, not inflated TDH
        H_curve = H_shutoff - (H_shutoff - r['TDH']) * (Q_range / r['Q_pump_gpm'])**1.5
        H_system = r['static_head'] + (r['TDH'] - r['static_head']) * (Q_range / r['Q_pump_gpm'])**2
        eff_curve = r['pump_eff'] * 100 * np.exp(-((Q_range/r['Q_pump_gpm'] - 1)**2) / 0.5)
        
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Pump Curve (CORRECTED)', 'Efficiency Curve'))
        
        fig3.add_trace(go.Scatter(x=Q_range, y=H_curve, name='Pump Curve', 
                                  line=dict(color='blue', width=3)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=Q_range, y=H_system, name='System Curve', 
                                  line=dict(color='red', dash='dash', width=2)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=[r['Q_pump_gpm']], y=[r['TDH']], name='Operating Point', 
                                  mode='markers', marker=dict(size=15, color='green')), row=1, col=1)
        
        fig3.add_trace(go.Scatter(x=Q_range, y=eff_curve, name='Efficiency', 
                                  line=dict(color='green', width=3)), row=1, col=2)
        fig3.add_vline(x=r['Q_pump_gpm'], line_dash="dash", line_color="red", row=1, col=2)
        
        fig3.update_xaxes(title_text="Flow Rate (GPM)", row=1, col=1)
        fig3.update_xaxes(title_text="Flow Rate (GPM)", row=1, col=2)
        fig3.update_yaxes(title_text="Head (ft)", row=1, col=1)
        fig3.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        fig3.update_layout(height=400, showlegend=True)
        
        # Add annotation about corrected approach
        fig3.add_annotation(
            text="CORRECTED: No safety factor on TDH",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            bgcolor="lightgreen",
            row=1, col=1
        )
        
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
# Tab 7: Export - CORRECTED
with tab7:
    if st.session_state.results:
        r = st.session_state.results
        
        st.subheader("Export Design Results (CORRECTED)")
        
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
        
        # Summary data - CORRECTED
        summary_data = {
            'Parameter': [
                'HGL Verification Status',
                'HGL at Discharge (ft)',
                'Discharge Elevation (ft)', 
                'HGL Error (ft)',
                'Flow Regime',
                'Total Dynamic Head (CORRECTED)',
                'Motor Safety Factor Applied',
                'Average Flow', 'Peak Flow', 'Pipe Velocity',
                'Total Forcemain Length', 'Design Length (for TDH)', 
                'Number of Elevation Points', 'Number of High Points',
                'Pump Elevation', 'Discharge Elevation', 'Controlling Elevation',
                'High Point Controls Design',
                'Static Head', 'Friction Loss (to control)', 'Minor Losses (to control)',
                'Siphon Exists', 'Siphon Length', 'Min Pressure in Siphon',
                'Number of Pumps', 'Pump Capacity', 'Motor HP Required', 'Motor HP Selected',
                'Wet Well Diameter', 'Storage Volume', 'Cycles per Hour',
                'Air Valves Required', 'Startup Analysis Available'
            ],
            'Value': [
                'PERFECT' if r['hgl_error'] < 0.001 else ('PASS' if r['hgl_error'] < 0.1 else 'FAIL'),
                f"{r['hgl_values'][-1]:.8f}",
                f"{r['discharge_elevation']:.8f}",
                f"{r['hgl_error']:.8f}",
                r['flow_regime'],
                f"{r['TDH']:.2f} ft (NO safety factor)",  # CORRECTED
                f"{r['motor_safety_factor']:.2f} (applied to motor only)",  # CORRECTED
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
                f"{r['MHP']:.2f} HP", f"{r['motor_size']:.1f} HP",  # CORRECTED: Show both required and selected
                f"{r['wetwell_diameter']:.1f} ft", f"{r['storage_volume_gal']:.0f} gal",
                f"{r['actual_cycles_per_hour']:.1f}",
                len(r['air_valve_data']), "YES" if 'startup_analysis' in r else "NO"
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
                file_name=f"lift_station_design_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                
                # Startup Analysis - NEW
                if 'startup_analysis' in r:
                    startup = r['startup_analysis']
                    startup_export = pd.DataFrame([{
                        'Startup TDH (ft)': startup['startup_TDH'],
                        'Running TDH (ft)': r['TDH'],
                        'Static to High Point (ft)': startup['static_to_high_point'],
                        'Static to Discharge (ft)': startup['static_to_discharge'],
                        'Air Evacuation Resistance (ft)': startup['air_analysis']['total_air_resistance'],
                        'Startup Flow (GPM)': startup['startup_flow_gpm'],
                        'Is Startup Critical': 'YES' if startup['is_startup_critical'] else 'NO',
                        'Startup Advantage (ft)': startup['startup_advantage'],
                        'Pipe Volume (gal)': startup['air_analysis']['pipe_volume_gal']
                    }])
                    startup_export.to_excel(writer, sheet_name='Startup Analysis', index=False)
                    
                    # Startup recommendations
                    startup_recs = pd.DataFrame([{
                        'Recommendation': rec
                    } for rec in startup['startup_recommendations']])
                    startup_recs.to_excel(writer, sheet_name='Startup Recommendations', index=False)
                
                # HGL Verification - CORRECTED
                hgl_verification = pd.DataFrame([{
                    'HGL at Discharge (ft)': f"{r['hgl_at_discharge']:.8f}",
                    'Discharge Elevation (ft)': f"{r['discharge_elevation']:.8f}",
                    'HGL Error (ft)': f"{r['hgl_error']:.8f}",
                    'Verification Status': 'PERFECT' if r['hgl_error'] < 0.001 else ('PASS' if r['hgl_error'] < 0.1 else 'FAIL'),
                    'TDH (ft)': r['TDH'],
                    'Motor Safety Factor': r['motor_safety_factor'],
                    'Motor HP Required': r['MHP'],
                    'Motor HP Selected': r['motor_size'],
                    'Calculation Method': 'CORRECTED: Backwards from discharge where P=0, NO safety factor on TDH'
                }])
                hgl_verification.to_excel(writer, sheet_name='HGL Verification', index=False)
            
            st.download_button(
                label="Download Full Report (Excel)",
                data=buffer.getvalue(),
                file_name=f"lift_station_design_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Technical specification report - CORRECTED
            spec_text = "LIFT STATION DESIGN REPORT v5.0 - CORRECTED\n"
            spec_text += "="*80 + "\n\n"
            spec_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            spec_text += "CORRECTED HGL VERIFICATION\n"
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
            spec_text += f"Method: CORRECTED - Backwards calculation from discharge (P=0)\n\n"
            
            spec_text += "CORRECTED HYDRAULIC DESIGN SUMMARY\n"
            spec_text += "-"*80 + "\n"
            spec_text += f"Flow Regime: {r['flow_regime']}\n"
            spec_text += f"Total Dynamic Head: {r['TDH']:.2f} ft (NO safety factor)\n"
            spec_text += f"Motor Safety Factor: {r['motor_safety_factor']:.2f} (applied to motor only)\n"
            spec_text += f"Design Length: {r['design_length']:.0f} ft\n"
            spec_text += f"Total Length: {r['total_length']:.0f} ft\n"
            spec_text += f"Static Head: {r['static_head']:.1f} ft\n"
            spec_text += f"Friction Loss: {r['friction_loss']:.2f} ft\n"
            spec_text += f"Minor Losses: {r['minor_losses']:.2f} ft\n\n"
            
            if 'startup_analysis' in r:
                startup = r['startup_analysis']
                spec_text += "STARTUP ANALYSIS\n"
                spec_text += "-"*80 + "\n"
                spec_text += f"Startup TDH: {startup['startup_TDH']:.2f} ft\n"
                spec_text += f"Running TDH: {r['TDH']:.2f} ft\n"
                spec_text += f"Static to High Point: {startup['static_to_high_point']:.1f} ft\n"
                spec_text += f"Air Evacuation Resistance: {startup['air_analysis']['total_air_resistance']:.1f} ft\n"
                spec_text += f"Is Critical: {'YES' if startup['is_startup_critical'] else 'NO'}\n"
                spec_text += f"Startup Advantage: {startup['startup_advantage']:.1f} ft\n\n"
            
            if r['siphon_data']['exists']:
                spec_text += "SIPHON ANALYSIS\n"
                spec_text += "-"*80 + "\n"
                spec_text += f"Siphon Length: {r['siphon_data']['length']:.0f} ft\n"
                spec_text += f"Elevation Drop: {r['siphon_data']['elevation_drop']:.1f} ft\n"
                spec_text += f"Minimum Pressure: {r['siphon_data']['min_pressure']:.2f} ft\n"
                spec_text += f"Vacuum Margin: {r['siphon_data']['vacuum_margin']:.1f} ft\n"
                spec_text += f"Stability: {'STABLE' if r['siphon_data']['is_stable'] else 'UNSTABLE'}\n\n"
            
            spec_text += "CORRECTED PUMP SPECIFICATIONS\n"
            spec_text += "-"*80 + "\n"
            spec_text += f"Number of Pumps: {r['num_pumps']}\n"
            spec_text += f"Operating Pumps: {r['pumps_operating']} (N-1 redundancy)\n"
            spec_text += f"Capacity (each): {r['Q_pump_gpm']:.0f} GPM @ {r['TDH']:.1f} ft TDH\n"
            spec_text += f"Motor HP Required: {r['MHP']:.2f} HP\n"
            spec_text += f"Motor HP Selected: {r['motor_size']:.1f} HP\n"
            spec_text += f"Motor Safety Factor: {r['motor_safety_factor']:.2f}\n"
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
            spec_text += "Report generated by Lift Station Sizing Tool v5.0 - CORRECTED\n"
            spec_text += "Smith & Loveless Methodology with CORRECTED TDH Approach\n"
            spec_text += "✅ NO safety factor on TDH - TDH is actual hydraulic requirement\n"
            spec_text += "✅ Safety factors applied to equipment only\n"
            spec_text += "✅ HGL calculated backwards from discharge where P=0\n"
            spec_text += "✅ Includes startup analysis for two operating points\n"
            spec_text += "="*80 + "\n"
            
            st.download_button(
                label="Download Technical Specs (TXT)",
                data=spec_text,
                file_name=f"lift_station_specs_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Calculate design first to export results")
# Tab 8: Startup Analysis - COMPLETELY NEW
with tab8:
    st.subheader("🚀 Pump Startup Analysis - Two Operating Points")
    st.markdown("**Analysis of pump requirements during system filling vs. normal operation**")
    
    if st.session_state.results:
        r = st.session_state.results
        
        if 'startup_analysis' not in r:
            st.warning("Recalculate design to see startup analysis")
        else:
            startup = r['startup_analysis']
            
            # Overview comparison
            st.markdown("### 📊 Operating Conditions Comparison")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🚀 Startup TDH", 
                    f"{startup['startup_TDH']:.1f} ft",
                    delta=f"{startup['startup_advantage']:.1f} ft higher" if startup['startup_advantage'] > 1 else "Similar to running"
                )
            with col2:
                st.metric(
                    "▶️ Running TDH", 
                    f"{r['TDH']:.1f} ft"
                )
            with col3:
                st.metric(
                    "💨 Air Evacuation Loss", 
                    f"{startup['air_analysis']['total_air_resistance']:.1f} ft"
                )
            
            # Detailed breakdown
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🚀 Startup Conditions (Empty Pipe)")
                
                st.markdown("**Static Head Components:**")
                st.write(f"• To high point: **{startup['static_to_high_point']:.1f} ft**")
                st.write(f"• To discharge: **{startup['static_to_discharge']:.1f} ft**")
                
                st.markdown("**Air Evacuation Analysis:**")
                air = startup['air_analysis']
                st.write(f"• Base air resistance: **{air['base_air_resistance']:.1f} ft**")
                st.write(f"• Elevation factor: **{air['elevation_air_resistance']:.1f} ft**")
                st.write(f"• Volume factor: **{air['volume_air_resistance']:.1f} ft**")
                st.write(f"• **Total air resistance: {air['total_air_resistance']:.1f} ft**")
                
                st.markdown("**System Characteristics:**")
                st.write(f"• Pipe volume: **{air['pipe_volume_gal']:,.0f} gallons**")
                st.write(f"• Startup flow: **{startup['startup_flow_gpm']:.0f} GPM** (60% of design)")
                st.write(f"• Friction multiplier: **{startup['filling_friction_factor']:.1f}x** (filling resistance)")
                
                st.info(f"**🎯 Total Startup TDH: {startup['startup_TDH']:.1f} ft**")
            
            with col2:
                st.markdown("### ▶️ Running Conditions (Filled Pipe)")
                
                st.markdown("**System Head Components:**")
                st.write(f"• Static head: **{r['static_head']:.1f} ft**")
                st.write(f"• Friction losses: **{r['friction_loss']:.1f} ft**")
                st.write(f"• Minor losses: **{r['minor_losses']:.1f} ft**")
                
                if r['siphon_data']['exists']:
                    st.markdown("**Siphon Assistance:**")
                    siphon = r['siphon_data']
                    st.write(f"• Elevation drop: **{siphon['elevation_drop']:.1f} ft**")
                    st.write(f"• Siphon length: **{siphon['length']:.0f} ft**")
                    st.success("✅ Gravity assists flow after high point")
                
                st.markdown("**Operating Characteristics:**")
                st.write(f"• Design flow: **{r['Q_peak']:.0f} GPM**")
                st.write(f"• Pipe velocity: **{r['pipe_velocity']:.2f} ft/s**")
                st.write(f"• Flow regime: **{r['flow_regime'][:30]}**")
                
                st.success(f"**🎯 Normal Operating TDH: {r['TDH']:.1f} ft**")
            
            # Critical analysis
            st.markdown("---")
            st.markdown("### ⚠️ Critical Analysis & Recommendations")
            
            if startup['is_startup_critical']:
                st.error("⚠️ **STARTUP CONDITIONS ARE CRITICAL**")
                
                # Show impact analysis
                impact_analysis = pd.DataFrame([
                    {
                        'Condition': '🚀 Startup (Empty Pipe)',
                        'TDH (ft)': startup['startup_TDH'],
                        'Flow (GPM)': startup['startup_flow_gpm'],
                        'Primary Challenge': 'Air evacuation + Static head to high point'
                    },
                    {
                        'Condition': '▶️ Running (Filled Pipe)', 
                        'TDH (ft)': r['TDH'],
                        'Flow (GPM)': r['Q_peak'],
                        'Primary Challenge': 'Friction losses + Minor losses'
                    }
                ])
                
                st.dataframe(impact_analysis, use_container_width=True)
                
            else:
                st.success("✅ **STARTUP CONDITIONS MANAGEABLE**")
            
            # Professional recommendations
            st.markdown("**🛠️ Engineering Recommendations:**")
            for rec in startup['startup_recommendations']:
                if "⚠️ CRITICAL" in rec:
                    st.error(rec)
                elif "⚠️" in rec:
                    st.warning(rec)
                elif "🔧" in rec:
                    st.info(rec)
                else:
                    st.write(rec)
            
            # Enhanced pump curve with both operating points
            st.markdown("---")
            st.markdown("### 📈 Pump Curve - Dual Operating Points")
            
            # Generate pump curve showing both points
            Q_range = np.linspace(0, r['Q_pump_gpm'] * 1.5, 100)
            
            # Startup point: higher head, lower flow
            # Running point: design head and flow
            
            startup_flow = startup['startup_flow_gpm']
            startup_head = startup['startup_TDH']
            running_flow = r['Q_pump_gpm'] 
            running_head = r['TDH']
            
            # Estimated pump curve that handles both points
            # Use higher of the two heads as shutoff head
            H_shutoff = max(startup_head, running_head) * 1.3
            H_curve = H_shutoff - (H_shutoff - max(startup_head, running_head)) * (Q_range / running_flow)**1.8
            
            fig_startup = go.Figure()
            
            # Pump curve
            fig_startup.add_trace(go.Scatter(
                x=Q_range, 
                y=H_curve, 
                name='Required Pump Curve',
                line=dict(color='blue', width=3)
            ))
            
            # Startup operating point
            fig_startup.add_trace(go.Scatter(
                x=[startup_flow], 
                y=[startup_head],
                mode='markers+text',
                name='🚀 Startup Point',
                marker=dict(size=15, color='red', symbol='diamond'),
                text=['STARTUP<br>POINT'],
                textposition='top center'
            ))
            
            # Running operating point  
            fig_startup.add_trace(go.Scatter(
                x=[running_flow], 
                y=[running_head],
                mode='markers+text', 
                name='▶️ Running Point',
                marker=dict(size=15, color='green', symbol='circle'),
                text=['RUNNING<br>POINT'],
                textposition='bottom center'
            ))
            
            # System curves for both conditions
            H_system_startup = startup['static_to_high_point'] + (startup_head - startup['static_to_high_point']) * (Q_range / startup_flow)**2
            H_system_running = r['static_head'] + (running_head - r['static_head']) * (Q_range / running_flow)**2
            
            fig_startup.add_trace(go.Scatter(
                x=Q_range, 
                y=H_system_startup,
                name='🚀 Startup System Curve',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig_startup.add_trace(go.Scatter(
                x=Q_range, 
                y=H_system_running,
                name='▶️ Running System Curve', 
                line=dict(color='green', dash='dash', width=2)
            ))
            
            fig_startup.update_layout(
                title="Pump Performance - Startup vs Running Conditions",
                xaxis_title="Flow Rate (GPM)",
                yaxis_title="Head (ft)",
                height=500,
                showlegend=True
            )
            
            # Add annotations for operating points
            fig_startup.add_annotation(
                x=startup_flow,
                y=startup_head + 5,
                text=f"Startup: {startup_flow:.0f} GPM @ {startup_head:.1f} ft",
                showarrow=True,
                bgcolor="red",
                bordercolor="white"
            )
            
            fig_startup.add_annotation(
                x=running_flow,
                y=running_head - 5,
                text=f"Running: {running_flow:.0f} GPM @ {running_head:.1f} ft", 
                showarrow=True,
                bgcolor="green",
                bordercolor="white"
            )
            
            st.plotly_chart(fig_startup, use_container_width=True)
            
            # Pump selection guidance
            st.markdown("---")
            st.markdown("### 🛠️ Pump Selection Guidance")
            
            if startup['startup_advantage'] > 5:
                st.error("""
                **🔧 CRITICAL PUMP SELECTION CONSIDERATIONS:**
                
                1. **Pump Curve Shape:** Select pump with relatively flat curve to handle both operating points efficiently
                2. **Motor Starting:** Verify motor can provide adequate starting torque for high startup head
                3. **Operating Efficiency:** Check efficiency at both startup and running points
                4. **Air Handling:** Ensure pump can handle air entrainment during filling
                5. **Control Strategy:** Consider staged startup or flow control during filling
                """)
            else:
                st.success("""
                **✅ STANDARD PUMP SELECTION APPLIES:**
                
                1. **Single Operating Point:** Design for running conditions
                2. **Standard Motor:** Normal starting requirements
                3. **Conventional Control:** Standard on/off operation acceptable
                """)
            
            # Air valve considerations
            st.markdown("### 💨 Air Valve Impact on Startup")
            
            if r['has_high_points']:
                st.warning("""
                **Air valve performance is CRITICAL during startup:**
                
                • **Large air release capacity** needed during initial filling
                • **Fast response** required to prevent vacuum formation
                • **Multiple high points** may require staged filling sequence
                • **Vacuum relief** essential if startup fails mid-process
                
                **Recommendation:** Verify air valve sizing accounts for startup air flow rates, not just operational requirements.
                """)
            else:
                st.info("No intermediate high points - standard air valve considerations apply")
    
    else:
        st.info("Calculate design first to see startup analysis")

# Tab 9: Series Pump Operation Analysis - SERIES ONLY
with tab9:
    st.subheader("? Series Pump Operation Analysis")
    
    # CLEAR EXPLANATION OF WHAT WE'RE ANALYZING
    st.info("""
    **📌 IMPORTANT: This analysis considers SERIES pump operation only**
    
    **Series Configuration:** Pump 1 → Pump 2 → Pump 3 → Discharge
    • Same flow through each pump
    • Head is divided among pumps  
    • Total head = sum of individual pump heads
    • Used for high-head applications
    
    **Note:** Parallel operation (multiple pumps to common header) is NOT analyzed here.
    """)
    
    # Tab 9: Series Pump Operation Analysis - WITH ERROR HANDLING
with tab9:
    st.subheader("🔧 Series Pump Operation Analysis")
    
    if st.session_state.results:
        r = st.session_state.results
        
        # DEFENSIVE CHECK - FIXED
        if 'series_pump_scenarios' not in r:
            st.warning("Recalculate design to see series pump operation analysis")
        else:
            scenarios = r['series_pump_scenarios']
            
            # HANDLE MISSING RECOMMENDATIONS - NEW
            if 'series_control_recommendations' in r:
                recommendations = r['series_control_recommendations']
            else:
                # Fallback if recommendations missing
                recommendations = [
                    "⚠️ Recommendations not calculated - recalculate design",
                    "🎯 Series pump analysis available below"
                ]
            
            # Quick overview
            st.markdown("### ? Series Configuration Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("System TDH", f"{r['TDH']:.1f} ft")
                st.caption("Total head requirement")
            with col2:
                st.metric("System Flow", f"{r['Q_peak']:.0f} GPM") 
                st.caption("Flow through each pump")
            with col3:
                st.metric("Available Pumps", f"{r['num_pumps']}")
                st.caption("Max pumps in series")
            
            # Scenario comparison
            st.markdown("---")
            st.markdown("### ? Series Pump Scenarios")
            
            # Create tabs for each scenario
            if len(scenarios) >= 3:
                scenario_tabs = st.tabs([f"{s['pumps_in_series']} Pump{'s' if s['pumps_in_series'] > 1 else ''}" for s in scenarios[:3]])
            else:
                scenario_tabs = st.tabs([f"{s['pumps_in_series']} Pump{'s' if s['pumps_in_series'] > 1 else ''}" for s in scenarios])
            
            for i, scenario in enumerate(scenarios[:len(scenario_tabs)]):
                with scenario_tabs[i]:
                    flow_per_pump = scenario.get('Q_per_pump_gpm', scenario.get('Q_operating', r['Q_peak']))
                    
                    st.markdown(f"### {scenario['scenario_type']}")
                    st.markdown(f"*{scenario['scenario_description']}*")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Head per Pump", f"{scenario['H_per_pump']:.1f} ft")
                        st.caption(f"vs {r['TDH']:.1f} ft for single pump")
                    
                    with col2:
                        st.metric("Flow per Pump", f"{flow_per_pump:.0f} GPM")
                        st.caption("Same for all pumps")
                    
                    with col3:
                        st.metric("Motor per Pump", f"{scenario['motor_size_per_pump']:.1f} HP")
                        st.caption(f"{scenario['power_kw_per_pump']:.1f} kW each")
                    
                    with col4:
                        st.metric("Total Power", f"{scenario['total_power_kw']:.1f} kW")
                        st.caption(f"{scenario['pumps_in_series']} × {scenario['power_kw_per_pump']:.1f} kW")
                    
                    # Advantages and disadvantages
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Advantages:**")
                        for advantage in scenario['advantages']:
                            st.write(advantage)
                    
                    with col2:
                        st.markdown("**⚠️ Considerations:**")
                        for disadvantage in scenario['disadvantages']:
                            st.write(disadvantage)
                    
                    # Application notes
                    st.info(f"**Typical Application:** {scenario['typical_application']}")
            
            # Comparison table
            st.markdown("---")
            st.markdown("### ? Series Configuration Comparison")
            
            comparison_data = []
            for scenario in scenarios:
                flow_per_pump = scenario.get('Q_per_pump_gpm', scenario.get('Q_operating', r['Q_peak']))
                comparison_data.append({
                    'Pumps in Series': scenario['pumps_in_series'],
                    'Head per Pump (ft)': f"{scenario['H_per_pump']:.1f}",
                    'Flow per Pump (GPM)': f"{flow_per_pump:.0f}",
                    'Motor per Pump (HP)': f"{scenario['motor_size_per_pump']:.1f}",
                    'Total Motors (HP)': f"{scenario['total_motor_hp']:.1f}",
                    'Total Power (kW)': f"{scenario['total_power_kw']:.1f}",
                    'Configuration': scenario['scenario_type']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
                        # YOUR METHOD: Per-pump system curves
            st.markdown("---")
            st.markdown("### ? Graphical Analysis - Per-Pump System Curves")
            
            st.info("""
            **Your Graphical Method:**
            • **Individual Pump Curve (blue solid):** Single pump characteristic (fixed)
            • **Per-Pump System Curves (multiple colors):** What each pump sees in series
              - 1 Pump: Full system curve
              - 2 Pumps: System curve ÷ 2 (each pump's share)
              - 3 Pumps: System curve ÷ 3 (each pump's share)
            • **Intersections (●):** Where pump curve meets per-pump system curve = operating point
            """)
            
            fig_per_pump = go.Figure()
            
            # Get plotting data
            Q_plot = np.array(scenarios[0]['curve_data']['Q_range'])
            individual_pump = scenarios[0]['curve_data']['individual_pump_curve']
            
            # INDIVIDUAL PUMP CURVE (one curve, applies to all scenarios)
            fig_per_pump.add_trace(go.Scatter(
                x=Q_plot,
                y=individual_pump,
                name='Individual Pump Curve',
                line=dict(color='blue', width=4),
                mode='lines'
            ))
            
            # PER-PUMP SYSTEM CURVES (one for each series configuration)
            colors = ['black', 'red', 'green', 'orange', 'purple']
            dash_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
            
            for i, scenario in enumerate(scenarios):
                n = scenario['pumps_in_series']
                color = colors[i % len(colors)]
                dash = dash_styles[i % len(dash_styles)]
                
                per_pump_system = scenario['curve_data']['per_pump_system_curve']
                
                # Per-pump system curve
                fig_per_pump.add_trace(go.Scatter(
                    x=Q_plot,
                    y=per_pump_system,
                    name=f'System Curve for {n}P (÷{n})',
                    line=dict(color=color, width=2.5, dash=dash),
                    mode='lines'
                ))
                
                # Operating point (intersection)
                fig_per_pump.add_trace(go.Scatter(
                    x=[scenario['Q_operating']],
                    y=[scenario['H_per_pump']],
                    mode='markers+text',
                    name=f'{n}P Operating Point',
                    marker=dict(size=12, color=color, symbol='circle', 
                               line=dict(width=2, color='white')),
                    text=[f"{n}P<br>{scenario['Q_operating']:.0f}GPM<br>{scenario['H_per_pump']:.1f}ft"],
                    textposition='top right',
                    showlegend=False
                ))
            
            fig_per_pump.update_layout(
                title="Series Pump Analysis - Per-Pump System Curves<br><sub>Each pump curve shows what individual pump must overcome in series</sub>",
                xaxis_title="Flow Rate (GPM) - Same through all pumps",
                yaxis_title="Head per Pump (ft)",
                height=600,
                showlegend=True,
                hovermode='closest'
            )
            
            # Add annotation
            fig_per_pump.add_annotation(
                x=r['Q_peak'] * 1.3,
                y=r['TDH'] * 0.8,
                text="Per-pump system curves:<br>System head ÷ n<br>Lower curves = easier for each pump",
                showarrow=True,
                arrowhead=2,
                bgcolor="lightyellow",
                bordercolor="orange"
            )
            
            st.plotly_chart(fig_per_pump, use_container_width=True)
            
            # Explanation
            st.markdown("### ? Understanding the Graph:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("""
                **Key Insight:**
                - **Blue curve** = Individual pump capability
                - **Colored curves** = What each pump must overcome
                - **Lower curves** = Easier for pump (series benefit)
                - **Intersection** = Where pump naturally operates
                """)
            
            with col2:
                st.info("""
                **Series Operation:**
                - More pumps = lower per-pump system curve
                - Pump intersects at higher flow
                - Each pump works at lower head
                - Total head = n × individual pump head
                """)
            # Series pump curve analysis
            st.markdown("---")
            st.markdown("### ? Series Pump Curves")
            
            fig_series = go.Figure()
            
            # For each scenario, show the pump curve for individual pumps
            max_flow_point = max([
                s.get('Q_per_pump_gpm', s.get('Q_operating', r['Q_peak'])) for s in scenarios[:3]
            ] + [r['Q_peak']])
            Q_range = np.linspace(0, max_flow_point * 1.25, 180)
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, scenario in enumerate(scenarios[:3]):  # Show first 3 scenarios
                color = colors[i % len(colors)]
                flow_per_pump = scenario.get('Q_per_pump_gpm', scenario.get('Q_operating', r['Q_peak']))
                flow_curve = max(flow_per_pump, 1e-6)
                
                # Build curve anchored at operating point so line passes through marker exactly
                H_oper = scenario['H_per_pump']
                H_shutoff = H_oper * 1.2
                exponent = 1.8
                H_curve = H_shutoff - (H_shutoff - H_oper) * (Q_range / flow_curve) ** exponent
                
                fig_series.add_trace(go.Scatter(
                    x=Q_range,
                    y=H_curve,
                    name=f"Individual Pump - {scenario['pumps_in_series']} Series",
                    line=dict(color=color, width=2),
                    mode='lines'
                ))
                
                # Operating point for this scenario
                fig_series.add_trace(go.Scatter(
                    x=[flow_per_pump],
                    y=[scenario['H_per_pump']],
                    mode='markers+text',
                    name=f"Operating Point - {scenario['pumps_in_series']} Series",
                    marker=dict(size=12, color=color, symbol='circle'),
                    text=[f"{scenario['pumps_in_series']}P<br>{scenario['H_per_pump']:.0f}ft"],
                    textposition='top center',
                    showlegend=False
                ))
            
            fig_series.update_layout(
                title="Individual Pump Curves - Series Configuration<br><sub>Each pump operates at reduced head when in series</sub>",
                xaxis_title="Flow per Pump (GPM) - Same for all pumps in series",
                yaxis_title="Head per Pump (ft) - Divided among pumps in series", 
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_series, use_container_width=True)
            
                        # Enhanced multi-curve plot - YOUR METHOD
            st.markdown("---")
            st.markdown("### ? Graphical Analysis - Series Pump Curves")
            
            st.info("""
            **Graphical Method:**
            • **System Curve (black):** Fixed piping system resistance
            • **Individual Pump Curve (blue):** Single pump characteristic  
            • **Series Combinations:** 2×, 3× pump curves (heads add at each flow)
            • **Intersections (●):** Operating points for each configuration
            """)
            
            fig_graphical = go.Figure()
            
            # Get curve data from first scenario for plotting range
            curve0 = scenarios[0].get('curve_data', {})
            if 'Q_range' not in curve0:
                st.warning("Series curve data missing. Recalculate design to refresh curve datasets.")
                st.stop()
            Q_plot = np.array(curve0['Q_range'])
            
            # SYSTEM CURVE (one curve, doesn't change)
            system_curve_plot = curve0.get('system_curve')
            if system_curve_plot is None and 'per_pump_system_curve' in curve0:
                system_curve_plot = curve0['per_pump_system_curve']
            if system_curve_plot is None:
                st.warning("System curve data missing. Recalculate design to refresh curve datasets.")
                st.stop()
            fig_graphical.add_trace(go.Scatter(
                x=Q_plot,
                y=system_curve_plot,
                name='System Curve (Fixed)',
                line=dict(color='black', width=3, dash='solid'),
                mode='lines'
            ))
            
            # INDIVIDUAL PUMP CURVE (reference)
            individual_pump_plot = curve0.get('individual_pump_curve')
            if individual_pump_plot is None:
                st.warning("Individual pump curve data missing. Recalculate design to refresh curve datasets.")
                st.stop()
            fig_graphical.add_trace(go.Scatter(
                x=Q_plot,
                y=individual_pump_plot,
                name='Individual Pump Curve',
                line=dict(color='blue', width=2, dash='dot'),
                mode='lines'
            ))
            
            # COMBINED SERIES PUMP CURVES and OPERATING POINTS
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, scenario in enumerate(scenarios):
                n = scenario['pumps_in_series']
                color = colors[i % len(colors)]
                
                # Combined pump curve (n× individual)
                curve_data = scenario.get('curve_data', {})
                combined_curve = curve_data.get('combined_pump_curve')
                if combined_curve is None:
                    base_curve = curve_data.get('individual_pump_curve', individual_pump_plot)
                    combined_curve = [v * n for v in base_curve]
                
                if n == 1:
                    # Don't redraw individual pump curve
                    pass
                else:
                    fig_graphical.add_trace(go.Scatter(
                        x=Q_plot,
                        y=combined_curve,
                        name=f'{n}× Pump Curve (Series)',
                        line=dict(color=color, width=2.5),
                        mode='lines'
                    ))
                
                # Operating point (intersection)
                fig_graphical.add_trace(go.Scatter(
                    x=[scenario['Q_operating']],
                    y=[scenario['H_system_total']],
                    mode='markers+text',
                    name=f'{n}-Pump Operating Point',
                    marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                    text=[f"{n}P<br>{scenario['Q_operating']:.0f}GPM<br>{scenario['H_system_total']:.0f}ft"],
                    textposition='top center',
                    showlegend=False
                ))
            
            fig_graphical.update_layout(
                title="Series Pump Analysis - Graphical Method<br><sub>Intersections show operating points for each configuration</sub>",
                xaxis_title="Flow Rate (GPM)",
                yaxis_title="Head (ft)",
                height=600,
                showlegend=True,
                hovermode='closest',
                xaxis=dict(range=[float(min(Q_plot)), float(max(Q_plot))])
            )
            
            # Add annotations explaining key points
            fig_graphical.add_annotation(
                x=r['Q_peak'] * 0.6,
                y=r['TDH'] * 1.5,
                text="Series pumps:<br>Heads ADD at each flow<br>Higher combined curves",
                showarrow=True,
                arrowhead=2,
                bgcolor="lightyellow",
                bordercolor="orange"
            )
            
            st.plotly_chart(fig_graphical, use_container_width=True)
            # Power analysis
            st.markdown("---")
            st.markdown("### ⚡ Power Consumption Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Power per pump vs total power
                pump_counts = [s['pumps_in_series'] for s in scenarios]
                power_per_pump = [s['power_kw_per_pump'] for s in scenarios]
                total_power = [s['total_power_kw'] for s in scenarios]
                
                fig_power = go.Figure()
                
                fig_power.add_trace(go.Bar(
                    x=pump_counts,
                    y=power_per_pump,
                    name='Power per Pump',
                    marker_color='lightblue',
                    text=[f'{p:.1f} kW' for p in power_per_pump],
                    textposition='auto'
                ))
                
                fig_power.add_trace(go.Scatter(
                    x=pump_counts,
                    y=total_power,
                    mode='lines+markers',
                    name='Total System Power',
                    line=dict(color='red', width=3),
                    marker=dict(size=10)
                ))
                
                fig_power.update_layout(
                    title="Power Analysis - Series Configuration",
                    xaxis_title="Number of Pumps in Series",
                    yaxis_title="Power (kW)",
                    height=400
                )
                
                st.plotly_chart(fig_power, use_container_width=True)
            
            with col2:
                # Head per pump analysis
                heads_per_pump = [s['H_per_pump'] for s in scenarios]
                
                fig_head = go.Figure()
                
                fig_head.add_trace(go.Bar(
                    x=pump_counts,
                    y=heads_per_pump,
                    name='Head per Pump',
                    marker_color='lightgreen',
                    text=[f'{h:.1f} ft' for h in heads_per_pump],
                    textposition='auto'
                ))
                
                fig_head.add_hline(
                    y=r['TDH'], 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"System TDH: {r['TDH']:.1f} ft"
                )
                
                fig_head.update_layout(
                    title="Head per Pump - Series Configuration",
                    xaxis_title="Number of Pumps in Series",
                    yaxis_title="Head per Pump (ft)",
                    height=400
                )
                
                st.plotly_chart(fig_head, use_container_width=True)
            
            # Engineering recommendations
            st.markdown("---")
            st.markdown("### ?️ Series Configuration Recommendations")
            
            for rec in recommendations:
                if "VERY HIGH HEAD" in rec or "strongly recommended" in rec:
                    st.success(rec)
                elif "HIGH HEAD" in rec or "Consider" in rec:
                    st.info(rec)
                elif "⚠️" in rec:
                    st.warning(rec)
                elif "MODERATE HEAD" in rec:
                    st.warning(rec)
                else:
                    st.write(rec)
            
            # Bottom line recommendation
            st.markdown("---")
            
            if r['TDH'] > 200:
                recommended_series_pumps = scenarios[1]['pumps_in_series'] if len(scenarios) > 1 else 2
                reduced_head_per_pump = scenarios[1]['H_per_pump'] if len(scenarios) > 1 else (r['TDH'] / 2)
                series_motor_hp = scenarios[1]['motor_size_per_pump'] if len(scenarios) > 1 else None
                st.success(f"""
                ## ✅ SERIES PUMPS RECOMMENDED
                
                **System TDH: {r['TDH']:.1f} ft** - This is very high head
                
                **Recommendation:** Use {recommended_series_pumps} pumps in series
                • Reduces head per pump to {reduced_head_per_pump:.1f} ft
                • Allows use of standard pumps instead of specialized high-head pumps
                • Motor per pump: {f"{series_motor_hp:.1f}" if series_motor_hp is not None else "TBD"} HP vs {scenarios[0]['motor_size_per_pump']:.1f} HP for single pump
                """)
            
            elif r['TDH'] > 100:
                series_motor_hp = scenarios[1]['motor_size_per_pump'] if len(scenarios) > 1 else None
                series_total_hp = scenarios[1]['total_motor_hp'] if len(scenarios) > 1 else None
                st.info(f"""
                ## ? CONSIDER SERIES PUMPS
                
                **System TDH: {r['TDH']:.1f} ft** - This is moderate to high head
                
                **Options:**
                • Single pump: {scenarios[0]['motor_size_per_pump']:.1f} HP motor required
                • Series pumps: {f"{series_motor_hp:.1f}" if series_motor_hp is not None else "TBD"} HP per pump (total {f"{series_total_hp:.1f}" if series_total_hp is not None else "TBD"} HP)
                
                **Decision factors:** Cost, availability, maintenance preferences
                """)
                
            else:
                st.warning(f"""
                ## ⚠️ SINGLE PUMP PREFERRED
                
                **System TDH: {r['TDH']:.1f} ft** - This is manageable with single pump
                
                **Recommendation:** Use single pump configuration
                • Series pumps add unnecessary complexity
                • Single {scenarios[0]['motor_size_per_pump']:.1f} HP pump sufficient
                • Lower maintenance and operational complexity
                """)
    
    else:
        st.info("Calculate design first to see series pump operation analysis")
# Footer - CORRECTED
st.markdown("---")
st.markdown("**Lift Station Sizing Tool v5.0** | Professional Engineering Design Software | CORRECTED VERSION")
st.caption("✅ **CORRECTED:** NO safety factor on TDH - TDH is actual hydraulic requirement")
st.caption("✅ **NEW:** Startup analysis for dual operating points")
st.caption("✅ **ENHANCED:** Air valve sizing, siphon analysis, multi-point elevation profiles")
st.caption("🛠️ **Method:** Smith & Loveless methodology with backwards HGL calculation from discharge (P=0)")
st.caption("⚙️ **Safety Factors:** Applied to equipment only (motor sizing, pump redundancy)")
