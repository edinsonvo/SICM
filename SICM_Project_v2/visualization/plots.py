import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def plot_is_lm(model, equilibrium_before: Dict, equilibrium_after: Dict = None):
    """Graficar curvas IS-LM interactivas"""
    
    Y_range = np.linspace(50, 200, 100)
    i_is = []
    i_lm = []
    
    for Y in Y_range:
        C = model.params['C0'] + model.params['c'] * (Y - model.params['T'])
        i_is_val = (Y - C - model.params['G'] - model.params['I0']) / (-model.params['b'] * 100)
        i_is.append(max(0, min(0.15, i_is_val)))
        
        i_lm_val = (model.params['k'] * Y - model.params['M']/model.params['P']) / model.params['h']
        i_lm.append(max(0, min(0.15, i_lm_val)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=Y_range, y=i_is, mode='lines', name='IS', 
                             line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=Y_range, y=i_lm, mode='lines', name='LM', 
                             line=dict(color='red', width=3)))
    
    fig.add_trace(go.Scatter(x=[equilibrium_before['Y']], y=[equilibrium_before['i']], 
                            mode='markers', name=f'Equilibrio inicial (Y={equilibrium_before["Y"]:.1f}, i={equilibrium_before["i"]:.2%})',
                            marker=dict(size=15, color='green', symbol='star')))
    
    if equilibrium_after:
        fig.add_trace(go.Scatter(x=[equilibrium_after['Y']], y=[equilibrium_after['i']], 
                                mode='markers', name=f'Nuevo equilibrio (Y={equilibrium_after["Y"]:.1f}, i={equilibrium_after["i"]:.2%})',
                                marker=dict(size=15, color='orange', symbol='star')))
        
        # Flecha de desplazamiento
        fig.add_annotation(x=equilibrium_before['Y'], y=equilibrium_before['i'],
                          xref="x", yref="y",
                          ax=equilibrium_after['Y'], ay=equilibrium_after['i'],
                          axref="x", ayref="y",
                          showarrow=True, arrowhead=2, arrowsize=1, 
                          arrowwidth=2, arrowcolor="gray")
    
    fig.update_layout(title="Curvas IS-LM", xaxis_title="Producción (Y)", 
                     yaxis_title="Tasa de interés (i)", hovermode='closest',
                     template='plotly_white', height=500)
    return fig

def plot_ad_as(model, equilibrium_before: Dict, equilibrium_after: Dict = None):
    """Graficar modelo AD-AS"""
    
    P_range = np.linspace(0.5, 2, 100)
    Y_AD = model.params['M'] * model.params['V'] / P_range
    Y_AS = np.full_like(P_range, model.params['Y_potential'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=Y_AD, y=P_range, mode='lines', name='AD', 
                             line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=Y_AS, y=P_range, mode='lines', name='LRAS', 
                             line=dict(color='red', width=3, dash='dash')))
    
    fig.add_trace(go.Scatter(x=[equilibrium_before['Y']], y=[equilibrium_before.get('P', 1)], 
                            mode='markers', name='Equilibrio', marker=dict(size=15, color='green', symbol='star')))
    
    fig.update_layout(title="Modelo AD-AS Clásico", xaxis_title="Producción (Y)", 
                     yaxis_title="Nivel de Precios (P)", template='plotly_white', height=500)
    return fig

def plot_time_series_simulation(df: pd.DataFrame, variable: str):
    """Graficar simulación temporal"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['periodo'], y=df[variable], mode='lines+markers',
                            name=variable, line=dict(width=2)))
    
    # Añadir línea de equilibrio
    if len(df) > 1:
        steady = df[variable].iloc[-5:].mean()
        fig.add_hline(y=steady, line_dash="dash", line_color="green",
                     annotation_text=f"Equilibrio: {steady:.2f}")
    
    fig.update_layout(title=f"Evolución de {variable} en el tiempo",
                     xaxis_title="Periodo", yaxis_title=variable,
                     template='plotly_white', height=400)
    return fig

def plot_irf(irf_data: Dict, variables: List[str]):
    """Graficar funciones impulso-respuesta"""
    
    fig = sp.make_subplots(rows=len(variables), cols=1, subplot_titles=variables)
    
    for idx, var in enumerate(variables, 1):
        if var in irf_data:
            fig.add_trace(go.Scatter(x=list(range(len(irf_data[var]))), y=irf_data[var],
                                    mode='lines+markers', name=var), row=idx, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=idx, col=1)
    
    fig.update_layout(title="Funciones Impulso-Respuesta (IRF)", height=300*len(variables),
                     template='plotly_white', showlegend=False)
    return fig

def plot_solow_dynamics(solow_model):
    """Graficar dinámica del modelo de Solow"""
    
    ss = solow_model.steady_state()
    path = solow_model.time_path(periods=100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=path['periods'], y=path['capital'], 
                            mode='lines', name='Capital (k)', line=dict(width=2)))
    fig.add_hline(y=ss['k_ss'], line_dash="dash", line_color="red",
                 annotation_text=f"Estado estacionario: k*={ss['k_ss']:.2f}")
    
    fig.update_layout(title="Convergencia al Estado Estacionario - Modelo de Solow",
                     xaxis_title="Tiempo", yaxis_title="Capital por trabajador efectivo",
                     template='plotly_white', height=400)
    return fig

def plot_comparative_schools(results: Dict, variable: str):
    """Comparar resultados entre escuelas económicas"""
    
    schools = list(results.keys())
    values_cp = [results[s][f'{variable}_CP'] for s in schools]
    values_lp = [results[s][f'{variable}_LP'] for s in schools]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Corto Plazo', x=schools, y=values_cp, text=values_cp, textposition='auto'))
    fig.add_trace(go.Bar(name='Largo Plazo', x=schools, y=values_lp, text=values_lp, textposition='auto'))
    
    fig.update_layout(title=f"Comparación de {variable} entre Escuelas", barmode='group',
                     xaxis_title="Escuela", yaxis_title=variable, template='plotly_white', height=400)
    return fig

def plot_forecast(forecast_results: Dict, variable: str):
    """Graficar pronósticos econométricos"""
    
    fig = go.Figure()
    
    # Datos históricos
    actual = forecast_results.get('model', None)
    if actual and hasattr(actual, 'data'):
        fig.add_trace(go.Scatter(x=actual.data.index, y=actual.data.values,
                                mode='lines', name='Histórico'))
    
    # Pronóstico
    forecast = forecast_results['forecast']
    forecast_idx = pd.date_range(start=actual.data.index[-1], periods=len(forecast)+1, freq='M')[1:]
    
    fig.add_trace(go.Scatter(x=forecast_idx, y=forecast, mode='lines+markers',
                            name='Pronóstico', line=dict(color='red', dash='dot')))
    
    # Intervalos de confianza
    if 'forecast_ci' in forecast_results:
        ci = forecast_results['forecast_ci']
        fig.add_trace(go.Scatter(x=forecast_idx, y=ci.iloc[:, 0], fill=None,
                                mode='lines', line=dict(color='gray', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_idx, y=ci.iloc[:, 1], fill='tonexty',
                                mode='lines', line=dict(color='gray', width=0),
                                name='Intervalo 95%'))
    
    fig.update_layout(title=f"Pronóstico de {variable}", xaxis_title="Fecha",
                     yaxis_title=variable, template='plotly_white', height=400)
    return fig