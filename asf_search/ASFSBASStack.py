from datetime import datetime, timedelta

import asf_search as asf
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape


### TODOs:
# - support add/remove pairs
# - optionally connect start and ends of each season with 1-year pairs
# - optional arg for target date from which to create bridge pairs
# - optional automatic disconnected-stack correction 
#     - If there is an interruption in data at the end of a season, a disconnection can occur
#         - work backwards from the end of the season until a scene is found that can connect to the following year
#     - if the final season in a stack contains no scene within the temporal baseline threshold of the end of the season, it will be disconnected
#         - solution: perform backwards search to connect end of stack to previous year (maintain earlier date as ref scene)

class ASFSBASStack:
    def __init__(self, **kwargs):
        self.plot_available = False
        try:
            import plotly.graph_objects as go
            import networkx as nx
            self.plot_available = True
        except ImportError:
            print('Warning: ASFSBASStack.plot() requires additional dependencies: plotly and/or networkx')
        
        self._ref_scene_id = kwargs.get('refSceneName', None)  
        self._season = kwargs.get('season', None)
        self._start = kwargs.get('start', None)
        self._end = kwargs.get('end', None)
        self._needs_ref_stack_update = True
        self._sbas_stack = None

        self.perp_baseline = kwargs.get('perpendicularBaseline', 400)
        self.temporal_baseline = kwargs.get('temporalBaseline', 36)
        self.repeat_pass_freq = kwargs.get('repeatPassFrequency', 12)

        # build stack upon initialization if required args were passed
        if self._ref_scene_id and self._season and self._end:
            self._sbas_stack = self.build_sbas_stack()
            self._needs_ref_stack_update = False

    @property
    def ref_scene_id(self):
        return self._ref_scene_id

    @ref_scene_id.setter
    def ref_scene_id(self, scene_id: str):
        self._ref_scene_id = scene_id
        self._needs_ref_stack_update = True

    @property
    def season(self):
        return self._season

    @season.setter
    def season(self, season: tuple[str,str]):
        self._season = season
        self._needs_ref_stack_update = True

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, stack_start_date: str):
        if (
            hasattr(self, '_sbas_stack') and 
            not (
                pd.to_datetime(self._sbas_stack['stopTime']).min() <=
                pd.to_datetime(stack_start_date) <=
                pd.to_datetime(self._sbas_stack['stopTime']).max()
            )
        ):
            self._needs_ref_stack_update = True
        self._start = stack_start_date

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, stack_end_date: str):
        if (
            hasattr(self, '_sbas_stack') and 
            not (
                pd.to_datetime(self._sbas_stack['stopTime']).min() <=
                pd.to_datetime(stack_end_date, utc=True) <=
                pd.to_datetime(self._sbas_stack['stopTime']).max()
            )
        ):
            self._needs_ref_stack_update = True
        self._end = stack_end_date

    @property
    def sbas_stack(self):
        if self._needs_ref_stack_update:
            self._sbas_stack = self.build_sbas_stack()
            self._needs_ref_stack_update = False
        else:
            self._sbas_stack['insarNeighbors'] = self._sbas_stack.apply(
            lambda row: self.get_seasonal_nearest_neighbors(self._sbas_stack, row['sceneName']),
            axis=1
        )
        return self._sbas_stack

    @property
    def plot(self):
        if not self.plot_available:
            raise Exception('ASFSBASStack.plot() requires additional dependencies: plotly and/or networkx')
        return self._plot
            
    
    def get_ref_stacks(self, stack_gdf, season_bounds):
        if self._sbas_stack is not None: 
            # merge updated stack_gdf with existing _sbas_stack so we only need to make API calls on added scenes 
            stack_gdf = self._sbas_stack.merge(stack_gdf, on=['sceneName'], how='right')

            # clean up after the merge
            x_to_copy = ['insarNeighbors_x', 'stack_x']
            y_to_copy = [i for i in stack_gdf.columns if '_y' in i and 'insarNeighbors' not in i and 'stack' not in i]
            
            for y in y_to_copy:
                stack_gdf[y.split('_y')[0]] = stack_gdf[y]
            for x in x_to_copy:
                stack_gdf[x.split('_x')[0]] = stack_gdf[x]
                
            to_drop = [i for i in stack_gdf.columns if '_x' in i or '_y' in i]
            stack_gdf = stack_gdf.drop(columns=to_drop)

        args = asf.ASFSearchOptions(
            **{
                'start': self._start,
                'end': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'season': season_bounds
            }
        )

        # update 'stack' with stack search results for ref scenes not previously searched
        stack_gdf['stack'] = stack_gdf.apply(
            lambda row: list(asf.stack_from_id(
                f"{row['sceneName']}-SLC" if 'BURST' not in row['sceneName'] else row['sceneName'], 
                args
            )) if (not isinstance(row['stack'], list) and pd.isna(row['stack'])) else row['stack'],
            axis=1
        )
        return stack_gdf

        
    def centered_sublist(self, stack, target, length):
        temp_baselines = [i.properties['temporalBaseline'] for i in stack]
        if len(temp_baselines) == 0:
            return []
        closest_value = min(temp_baselines, key=lambda x: abs(x - target))
        target_index = temp_baselines.index(closest_value)
        half_length = length // 2
        
        start_index = max(0, target_index - half_length)
        end_index = min(len(stack), start_index + length)
        
        if end_index - start_index < length:
            start_index = max(0, end_index - length)
    
        return [stack[i] for i in range(start_index, end_index)]
    
    
    def baseline_stack_filter(self, stack, temporal_baseline_range):
        return [
            i for i in stack 
                if (
                    temporal_baseline_range[0] < i.properties['temporalBaseline'] <= temporal_baseline_range[1]
                    and
                    i.properties['perpendicularBaseline']
                    and
                    np.abs(i.properties['perpendicularBaseline']) <= self.perp_baseline
                    and
                    pd.Timestamp(i.properties['stopTime'], tz='UTC') <= pd.Timestamp(self._end, tz='UTC')
                )
            ]
    
    
    def get_seasonal_nearest_neighbors(self, stack_gdf, ref_scene_name):
        stack = stack_gdf.loc[stack_gdf.sceneName == ref_scene_name]['stack'].iloc[0]
    
        # create stack of in-season scenes, within baseline thresholds
        cur_season_stack = self.baseline_stack_filter(stack, (0, self.temporal_baseline))
    
        # find the number of expected neighbors 
        neighbor_len = self.temporal_baseline // self.repeat_pass_freq
    
        # return the current season's stack if enough neighbors were found
        if len(cur_season_stack) >= neighbor_len:
            return cur_season_stack
    
        # create stack of next-season scenes
        next_season_stack = self.baseline_stack_filter(stack, (365-self.temporal_baseline, 365+self.temporal_baseline))
    
        # find number remaining neighbors to identify
        neighbor_len = neighbor_len - len(cur_season_stack)
    
        # return any current season and next season scenes found
        return cur_season_stack + self.centered_sublist(next_season_stack, 365, neighbor_len)

    def get_insar_pairs(self):
        
        return [
            (
                row['sceneName'], 
                neighbor.properties['sceneName']
            ) 
            for _, row in self.sbas_stack.iterrows() 
            for neighbor in row['insarNeighbors']
        ]

    def _plot(self):
        G = nx.Graph()

        insar_node_pairs = [
            (
                row['stopTime'].split('T')[0], 
                neighbor.properties['stopTime'].split('T')[0], 
                {
                    'perp_bs': neighbor.properties['perpendicularBaseline'],
                    'temp_bs': neighbor.properties['temporalBaseline']
                }
            ) 
            for _, row in self.sbas_stack.iterrows() 
            for neighbor in row['insarNeighbors']
        ]
        
        G.add_edges_from(insar_node_pairs, data=True)
        
        scene_dates = {row['stopTime'].split('T')[0]: row['stopTime'].split('T')[0] for _, row in self._sbas_stack.iterrows()}
        nx.set_node_attributes(G, scene_dates, 'date')
        
        perp_bs = {row['stopTime'].split('T')[0]: row['perpendicularBaseline'] for _, row in self._sbas_stack.iterrows()}
        nx.set_node_attributes(G, perp_bs, 'perp_bs')
        
        node_positions = {node: datetime.strptime(data['date'], '%Y-%m-%d').timestamp() for node, data in G.nodes(data=True)}
        node_y_positions = {node: data['perp_bs'] for node, data in G.nodes(data=True)}
        
        node_x = [node_positions[node] for node in G.nodes()]
        node_y = [node_y_positions[node] for node in G.nodes()]
        node_text = [G.nodes[node]['date'] for node in G.nodes()]
        
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0 = node_positions[edge[0]]
            y0 = node_y_positions[edge[0]]
            x1 = node_positions[edge[1]]
            y1 = node_y_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[0]} - {edge[1]}, perp baseline: {edge[2]['perp_bs']}, temp baseline: {edge[2]['temp_bs']}")
        
        start_date = min(scene_dates.values())
        end_date = max(scene_dates.values())
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
        date_range_ts = [datetime.strptime(date, '%Y-%m').timestamp() for date in date_range]
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=4, color='rgba(52, 114, 168, 0.7)'),
            mode='lines'
        )
        
        edge_hover_trace = go.Scatter(
            x=[(node_positions[edge[0]] + node_positions[edge[1]]) / 2 for edge in G.edges()],
            y=[(node_y_positions[edge[0]] + node_y_positions[edge[1]]) / 2 for edge in G.edges()],
            mode='markers',
            marker=dict(size=20, color='rgba(255, 255, 255, 0)'),
            hoverinfo='text',
            text=edge_text
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition="top center",
            marker=dict(size=15, color='rgba(32, 33, 32, 0.7)', line_width=0),
            hoverinfo='text',
            hovertext=node_text
        )
        
        def f_date(dash_date_str):
            return dash_date_str.replace("-", "/")
        
        fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            height=800,
                            margin=dict(t=180),
                            xaxis=dict(
                                title='Acquisition Date',
                                tickvals=date_range_ts,
                                ticktext=date_range,
                                gridcolor='gray',
                                zerolinecolor='gray'
                            ),
                            yaxis=dict(
                                title='Perpendicular Baseline (m)',
                                gridcolor='gray',
                                zerolinecolor='gray'
                            ),
                            title=dict(
                                text=(
                                    f'Sentinel-1 Seasonal SBAS Stack from Reference SLC: {self._ref_scene_id}<br>'
                                    f'Temporal Bounds: {f_date(self._start)} - {f_date(self._end)}<br>Seasonal Bounds {f_date(self._season[0])} - {f_date(self._season[1])}<br>'
                                    f'Max Temporal Baseline: {self.temporal_baseline} days, Max Perpendicular Baseline: {self.perp_baseline}m<br>'
                                    f'Stack Size: {len(insar_node_pairs)} pairs from {len(scene_dates)} scenes<br>'
                                ),
                                y=0.95,
                                x=0.5,
                                xanchor='center',
                                yanchor='top',
                                font=dict(
                                    family="Helvetica, monospace",
                                    size=24,
                                )
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='lightgrey',
                        ))   
        fig.show()

    def build_sbas_stack(self):
        if not self._ref_scene_id:
            raise(Exception('Stack "refSceneName" not set'))
        if not self._start:
            raise(Exception('Stack "start" not set'))
        if not self._end:
            raise(Exception('Stack "end" not set'))
        if pd.to_datetime(self._end) <= pd.to_datetime(self._start):
            raise(Exception(f'Stack end date is earlier than its start date. Correct to proceed.'))


        
        if self._season:
            season_start_ts = pd.Timestamp(datetime.strptime(self._season[0], '%m-%d'), tz='UTC')
            season_start_day = season_start_ts.timetuple().tm_yday
            season_end_ts = pd.Timestamp(datetime.strptime(self._season[1], '%m-%d'), tz='UTC')
            season_end_day = season_end_ts.timetuple().tm_yday
            season = (season_start_day, season_end_day)
        else:
            season = None
        
        args = asf.ASFSearchOptions(
            **{
                'start': self._start,
                'end': self._end,
                'season': season
            }
        )
        
        # create stack
        stack = asf.stack_from_id(self._ref_scene_id, args)

        gdf = gpd.GeoDataFrame([s.properties|s.baseline for s in stack], geometry=[shape(s.geometry) for s in stack])
        gdf['insarNeighbors'] = [float('nan') for _ in range(len(gdf))]
        gdf['stack'] = [float('nan') for _ in range(len(gdf))]

        gdf = self.get_ref_stacks(gdf, season)

        gdf['insarNeighbors'] = gdf.apply(
            lambda row: self.get_seasonal_nearest_neighbors(gdf, row['sceneName']),
            axis=1
        )
        return gdf

 